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
def train_model(model, dataloader, num_of_epochs, optimizer):
    model.train()
    for epoch in range(num_of_epochs):
        for i, batch in enumerate(dataloader):
            texts, labels = batch
            input_texts = ["classify: " + text for text in texts]
            inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            labels = labels.unsqueeze(-1).to(device)  # Add an extra dimension to the labels tensor
            loss = model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), labels=labels).loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


# Evaluate the model on the test dataset
def evaluate_model(model, test_dataloader):
    model.eval()
    total_correct_preds = 0
    total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            texts, labels = batch
            labels = labels.to(device)
            input_texts = ["classify: " + text for text in texts]
            inputs = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            decoder_input_ids = torch.zeros_like(inputs["input_ids"])
            outputs = model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), decoder_input_ids=decoder_input_ids.to(device), return_dict=True)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=-1).squeeze()

            # Compare the single predicted label to the corresponding label
            total_correct_preds += (predicted_labels.item() == labels.item())
            total_samples += 1

    accuracy = total_correct_preds / total_samples
    return accuracy



optimizer = AdamW(model.parameters(), lr=1e-5)
num_of_epochs = 4
with open(args.logfile, "w") as f:
    # Train the model
    train_model(model, dataloader, num_of_epochs, optimizer)

    # Define the test dataloader
    test_file_path = [args.testfile]
    test_dataset = TextClassificationDataset(test_file_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # Evaluate the model on the test dataset
    accuracy = evaluate_model(model, test_dataloader)

    # Log the results
    print(f"Accuracy: {accuracy * 100:.2f}%", file=f)