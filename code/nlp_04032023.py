#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, RobertaForSequenceClassification, RobertaTokenizer, BertTokenizer, BertForSequenceClassification, AdamW
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logfile', type=str, help='name of the log file')
parser.add_argument('-tr', '--trainfile', type=str, help='name of the training file')
parser.add_argument('-te', '--testfile', type=str, help='name of the test file')
args = parser.parse_args()

# Define the dataset
class Dataset(Dataset):
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

time_start = time.time()

# Load the pre-trained model
# model = RobertaForSequenceClassification.from_pretrained("allenai/biomed_roberta_base")
# tokenizer = RobertaTokenizer.from_pretrained("allenai/biomed_roberta_base")

# Loading pre-trained-fine-tuned model
# model = RobertaForSequenceClassification.from_pretrained('finetuned_model_roberta_4')
# tokenizer = RobertaTokenizer.from_pretrained('finetuned_model_roberta_4')

model = RobertaForSequenceClassification.from_pretrained("roberta-base")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Define the dataloader
file_paths = [args.trainfile]
dataset = Dataset(file_paths)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)



## Fine-tune the model ##
save_model = False
model.train()
num_of_epochs = 1
optimizer = AdamW(model.parameters(), lr=1e-5) # weight_decay=0.01
with open(args.logfile, 'w') as f:
    print(f"Training for {num_of_epochs} epochs", file=f)
    print(f"Training for {num_of_epochs} epochs")
    for epoch in range(num_of_epochs):
        print(f"Epoch {epoch+1}/{num_of_epochs}", file=f)
        for i, batch in enumerate(dataloader):
            print(f"Batch {i+1}/{len(dataloader)}", file=f)
            texts, labels = batch
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            outputs = model(inputs["input_ids"], inputs["attention_mask"], labels=labels)

            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

    # Define the test dataloader, re-using
    test_file_path = [args.testfile]
    test_dataset = Dataset(test_file_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # Evaluate the model on the test dataset
    model.eval()
    total_correct_preds = 0
    total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            print(f"Batch {i+1}/{len(test_dataloader)}", file=f)
            abstract_text, labels = batch
            inputs = tokenizer(abstract_text, padding=True, truncation=True, return_tensors="pt")
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            predictions = torch.argmax(outputs.logits, dim=1)
            print('Prediction class:', predictions.item(), '\tCorrect label:', labels.item(), '\tprobs',torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0], file=f)
            total_correct_preds += torch.sum(predictions == labels).item()
            total_samples += 1

    accuracy = total_correct_preds / total_samples
    print("Accuracy: {:.2f}%".format(accuracy * 100), file=f)

# Save the fine-tuned model
if save_model:
  model_dir = f"finetuned_model_roberta_{num_of_epochs}"
  model.save_pretrained(model_dir)
  tokenizer.save_pretrained(model_dir)
time_end = time.time()
print(f"Time elapsed in this session: {round(time_end - time_start, 2)/ 60} minutes", file=f)