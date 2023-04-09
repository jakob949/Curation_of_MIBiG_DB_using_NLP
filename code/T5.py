import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config
import argparse
import time


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logfile', type=str, help='name of the log file')
parser.add_argument('-tr', '--trainfile', type=str, help='name of the training file')
parser.add_argument('-te', '--testfile', type=str, help='name of the test file')
args = parser.parse_args()

class Dataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = []
        with open(filename, "r") as f:
            for line in f:
                text, label = line.strip().split("\t")
                # label = "1" if label == "1" else "0"
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

start_time = time.time()

model_name = "google/flan-t5-base"
tokenizer = T5TokenizerFast.from_pretrained(model_name)
config = T5Config.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)

train_dataset = Dataset(args.trainfile, tokenizer)
test_dataset = Dataset(args.testfile, tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 4
epochs = 7
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

with open(args.logfile, 'w') as f:
    f.write(f"Model name: {model_name}, Train file: {args.trainfile}, Test file: {args.testfile}, Batch size: {batch_size}, Epochs: {epochs}, Device: {device}\n\n")

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
            print('outputs: ', outputs)
            print('Predicted labels: ', predicted_labels)
            true_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            print('True labels: ', true_labels)

        for pred, true in zip(predicted_labels, true_labels):
            total_predictions += 1
            if pred == true:
                print('\npred: ',pred,'\ntrue: ', true)
                correct_predictions += 1

    with open(args.logfile, 'a') as f:
        print(f"Epoch {epoch + 1}/{epochs}", file=f)
        print(f"Accuracy: {round(correct_predictions / total_predictions, 3)}", file=f)
model.save_pretrained("fine_tuned_flan-t5-base")
end_time = time.time()
with open(args.logfile, 'a') as f:
    print(f"Total time: {round((end_time - start_time)/60, 2)} minutes", file=f)
