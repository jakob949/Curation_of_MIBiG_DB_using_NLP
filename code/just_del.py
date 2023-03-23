#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, RobertaForSequenceClassification, RobertaTokenizer, BertTokenizer, BertForSequenceClassification, AdamW
import time

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
model = RobertaForSequenceClassification.from_pretrained('finetuned_model_roberta_4')
tokenizer = RobertaTokenizer.from_pretrained('finetuned_model_roberta_4')



test_file_path = ['hard_dataset.txt']
test_dataset = Dataset(test_file_path)
test_dataloader = DataLoader(test_dataset, batch_size=1)


preds = []
model.eval()
total_correct_preds = 0
total_samples = 0
with torch.no_grad():
    for i, batch in enumerate(test_dataloader):
        print(f"Batch {i+1}/{len(test_dataloader)}")
        abstract_text, labels = batch
        inputs = tokenizer(abstract_text, padding=True, truncation=True, return_tensors="pt")
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        predictions = torch.argmax(outputs.logits, dim=1)
        print('Prediction class:', predictions.item(), '\tCorrect label:', labels.item(), '\tprobs',torch.nn.functional.softmax(outputs.logits, dim=1).tolist()[0])
        total_correct_preds += torch.sum(predictions == labels).item()
        total_samples += 1
        preds.append(predictions.item())
    accuracy = total_correct_preds / total_samples


import pickle
with open('hard_preds.pickle', 'wb') as f:
    pickle.dump(preds, f)

