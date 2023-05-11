import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config
import time
from rdkit import Chem
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse as arg

parser = arg.ArgumentParser()
parser.add_argument("-o", "--output_file_name", type=str, default="unknown", )
args = parser.parse_args()


class Dataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=1750):
        self.tokenizer = tokenizer
        self.data = []
        with open(filename, "r") as f:
            for line in f:
                if len(line.strip().split("\t")) == 2:

                    text = line.split('\t')[0]
                    task = text.split(':')[0]
                    label = line.split('\t')[1].strip('\n')
                    # label = "1" if label == "1" else "0"
                    if len(text) < 1750:
                        self.data.append((text, label, task))
        print(len(self.data))
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        input_encoding = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding="max_length")
        target_encoding = self.tokenizer(label, return_tensors="pt", max_length=400, padding="max_length",
                                         truncation=True)

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }


def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

# [Dataset class definition remains the same]

start_time = time.time()

model_name = "google/flan-t5-base"
tokenizer = T5TokenizerFast.from_pretrained(model_name)
config = T5Config.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)

train_dataset = Dataset("train_combined_multitask.txt", tokenizer)
test_dataset = Dataset("test_combined_multitask.txt", tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#
batch_size = 6
epochs = 20
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

with open("log.txt", 'w') as f:
    f.write(f"Model name: {model_name}, Batch size: {batch_size}, Epochs: {epochs}, Device: {device}\n\n")

for epoch in range(epochs):
    model.train()
    train_accuracy_accumulated = 0.0
    train_f1_accumulated = 0.0
    num_train_batches = 0

    for batch in train_loader:
        num_train_batches += 1
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        with torch.no_grad():
            train_outputs = model.generate(input_ids, attention_mask=attention_mask, num_beams=12)
            train_predicted_labels = [tokenizer.decode(pred, skip_special_tokens=True) for pred in train_outputs]
            train_true_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

            # Calculate accuracy and f1 score
            train_accuracy = accuracy_score(train_true_labels, train_predicted_labels)
            train_f1 = f1_score(train_true_labels, train_predicted_labels, average='weighted')

            # Accumulate the values of these metrics in separate variables
            train_accuracy_accumulated += train_accuracy
            train_f1_accumulated += train_f1

    scheduler.step()

    model.eval()

    test_accuracy_accumulated = 0.0
    test_f1_accumulated = 0.0
    num_test_batches = 0

    for batch in test_loader:
        num_test_batches += 1
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, num_beams=6)
            test_predicted_labels = [tokenizer.decode(pred, skip_special_tokens=True) for pred in outputs]
            test_true_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

            # Calculate accuracy and f1 score
            test_accuracy = accuracy_score(test_true_labels, test_predicted_labels)
            test_f1 = f1_score(test_true_labels, test_predicted_labels, average='weighted')

            # Accumulate the values of these metrics in separate variables
            test_accuracy_accumulated += test_accuracy
            test_f1_accumulated += test_f1
            #
            with open(f"predictions_{args.output_file_name}.txt", "a") as predictions_file:
                print(f"Epoch {epoch + 1}/{epochs}\tTrue: {test_true_labels}\tPred: {test_predicted_labels}",
                      file=predictions_file)

    with open(f"scores_{args.output_file_name}.txt", "a") as scores_file:
        print(
            f"Epoch {epoch + 1}/{epochs}\tAvg Train Accuracy\t {train_accuracy_accumulated / num_train_batches}\tAvg Train F1 Score\t {train_f1_accumulated / num_train_batches}",
            file=scores_file)

        print(
            f"Epoch {epoch + 1}/{epochs}\tAvg Test Accuracy\t {test_accuracy_accumulated / num_test_batches}\tAvg Test F1 Score\t {test_f1_accumulated / num_test_batches}",
            file=scores_file)
