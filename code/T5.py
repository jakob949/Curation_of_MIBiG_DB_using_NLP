#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer
from torch.optim import AdamW

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logfile', type=str, help='name of the log file')
parser.add_argument('-tr', '--trainfile', type=str, help='name of the training file')
parser.add_argument('-te', '--testfile', type=str, help='name of the test file')
args = parser.parse_args()

class TextClassificationDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []
        for file_path in file_paths:
            with open(file_path, "r") as f:
                for line in f:
                    text, label = line.strip().split("\t")
                    self.data.append((text, int(label)))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the pre-trained model
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base").to(device)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

# Define the dataloader
train_file_path = [args.trainfile]
dataset = TextClassificationDataset(train_file_path)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Fine-tune the model
model.train()
num_of_epochs = 4
optimizer = AdamW(model.parameters(), lr=1e-5)

with open(args.logfile, "w") as f:
    for epoch in range(num_of_epochs):
        print(f"Epoch {epoch + 1}/{num_of_epochs}", file=f)
        for i, batch in enumerate(dataloader):
            print(f"Batch {i + 1}/{len(dataloader)}", file=f)
            texts, labels = batch
            input_texts = ["classify: " + text for text in texts]
            inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
            labels = labels.to(device)
            loss = model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), labels=labels).loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Define the test dataloader
    test_file_path = [args.testfile]
    test_dataset = TextClassificationDataset(test_file_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # Evaluate the model on the test dataset
    model.eval()
    total_correct_preds = 0
    total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            print(f"Batch {i + 1}/{len(test_dataloader)}", file=f)
            abstract_text, labels = batch
            input_text = "classify: " + abstract_text[0]
            inputs = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt")
            outputs = model.generate(inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), num_return_sequences=1)
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            predicted_label = int(prediction)
            print(f"Prediction class: {predicted_label}\tCorrect label: {labels.item()}", file=f)
            total_correct_preds += (predicted_label == labels.item())
            total_samples += 1
    accuracy = total_correct_preds / total_samples
    print(f"Accuracy: {accuracy * 100:.2f}%", file=f)
