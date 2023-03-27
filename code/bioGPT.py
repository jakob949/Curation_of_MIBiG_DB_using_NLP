import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, AdamW

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logfile', type=str, help='name of the log file')
parser.add_argument('-tr', '--trainfile', type=str, help='name of the training file')
parser.add_argument('-te', '--testfile', type=str, help='name of the test file')
args = parser.parse_args()

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = []
        self.max_length = max_length

        with open(filename, "r", encoding="utf-8") as f:
            for line in f.readlines():
                text, label = line.strip().split("\t")
                binary_label = 1 if label == "1" else 0
                self.data.append((text, binary_label))


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, binary_label = self.data[idx]
        input_encoding = self.tokenizer("classify: " + text, return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        target_encoding = torch.tensor(binary_label, dtype=torch.long)

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding,
        }

start_time = time.time()
model_name = 'gpt2'

tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = GPT2Tokenizer.eos_token
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=2)
model.config.pad_token_id = model.config.eos_token_id

train_dataset = Dataset(args.trainfile, tokenizer)
test_dataset = Dataset(args.testfile, tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 6
epochs = 4
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
            outputs = model(input_ids, attention_mask = attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=-1)

        correct_predictions += (predicted_labels == labels).sum().item()
        total_predictions += labels.size(0)

    with open(args.logfile, 'a') as f:
        print(f"Epoch {epoch + 1}/{epochs}", file=f)
        print(f"Accuracy: {round(correct_predictions / total_predictions, 3)}", file=f)

if args.trainfile == "train_fold_0.txt":
    model.save_pretrained(f"fine_tuned_{model_name}_epoch{epochs}")

end_time = time.time()
with open(args.logfile, 'a') as f:
    print(f"Total time: {round((end_time - start_time)/60, 2)} minutes", file=f)

