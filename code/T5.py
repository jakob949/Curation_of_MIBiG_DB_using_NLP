import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config

class SpacyDataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = []
        with open(filename, "r", encoding="utf-8") as f:
            for line in f.readlines():
                text, label = line.strip().split("\t")
                self.data.append((text, label))

        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        input_encoding = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        target_encoding = self.tokenizer(label, return_tensors="pt", max_length=2, padding="max_length", truncation=True)

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }

model_name = "google/flan-t5-base"
tokenizer = T5TokenizerFast.from_pretrained(model_name)
config = T5Config.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)

train_dataset = SpacyDataset("spacy_train.txt", tokenizer)
test_dataset = SpacyDataset("spacy_test.txt", tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask)
            predicted_labels = [tokenizer.decode(pred, skip_special_tokens=True) for pred in outputs]
            true_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

        for pred, true in zip(predicted_labels, true_labels):
            total_predictions += 1
            if pred == true:
                correct_predictions += 1

    print(f"Epoch {epoch + 1}/{epochs}")
    print(f"Accuracy: {correct_predictions / total_predictions:.2f}")

model.save_pretrained("fine_tuned_flan-t5-base")

