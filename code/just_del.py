#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import time
import argparse
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datafile', type=str, help='name of the data file')
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

# Loading pre-trained model
model = RobertaModel.from_pretrained('finetuned_model_roberta_4')
tokenizer = RobertaTokenizer.from_pretrained('finetuned_model_roberta_4')

# Define the dataloader
file_paths = ['dataset_positives_titles_abstracts.txt', 'dataset_negatives_titles_abstracts.txt']
dataset = Dataset(file_paths)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Extract embeddings from the pre-trained model
model.eval()
embeddings = []
labels = []

with torch.no_grad():
    for i, batch in enumerate(dataloader):
        texts, batch_labels = batch
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        outputs = model(inputs["input_ids"], inputs["attention_mask"])
        embeddings.extend(outputs.last_hidden_state[:, 0, :].detach().numpy())
        labels.extend(batch_labels.numpy())

# Apply PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# Plot PCA results
plt.figure(figsize=(10, 10))
for label in set(labels):
    idx = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label)

plt.legend()
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.title("PCA Plot")
plt.savefig('PCA_embeddings.pdf', bbox_inches='tight', dpi=300, format='pdf')

time_end = time.time()
print(f"Time elapsed in this session: {round(time_end - time_start, 2) / 60} minutes")

