import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config
import time
from torchmetrics import BLEUScore
from torchmetrics.text.rouge import ROUGEScore
from torch.optim.lr_scheduler import CosineAnnealingLR


class Dataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=1750):
        self.tokenizer = tokenizer
        self.data = []
        with open(filename, "r") as f:
            for line in f:
                if len(line.strip().split("\t")) == 3:

                    text = line.split('\t')[1]
                    label = line.split('\t')[2].strip('\n')
                    # label = "1" if label == "1" else "0"
                    if len(text) < 1750:
                        self.data.append((text, label))
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


start_time = time.time()

model_name = "google/flan-t5-base"
tokenizer = T5TokenizerFast.from_pretrained(model_name)
config = T5Config.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)

train_dataset = Dataset("train_dataset_protein_text_v2_shorten_0.txt", tokenizer)
test_dataset = Dataset("test_dataset_protein_text_v2_shorten_0.txt", tokenizer)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 12

epochs = 35
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

rouge_accumulated = 0.0
num_batches = 0
rouge = ROUGEScore()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

with open("log.txt", 'w') as f:
    f.write(f"Model name: {model_name}, Batch size: {batch_size}, Epochs: {epochs}, Device: {device}\n\n")

for epoch in range(epochs):
    model.train()
    rouge_train_accumulated = 0.0
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

        # Clip gradients to avoid exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        with torch.no_grad():
            train_outputs = model.generate(input_ids, attention_mask=attention_mask, num_beams=6)
            train_predicted_labels = [tokenizer.decode(pred, skip_special_tokens=True) for pred in train_outputs]
            train_true_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            train_rouge_score = rouge(train_predicted_labels, train_true_labels)["rouge1_fmeasure"]
            rouge_train_accumulated += train_rouge_score

    # Update learning rate using scheduler
    scheduler.step()

    model.eval()
    correct_predictions = 0
    total_predictions = 0
    rouge_accumulated = 0.0
    num_batches = 0
    for batch in test_loader:
        num_batches += 1
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, num_beams=6)
            predicted_labels = [tokenizer.decode(pred, skip_special_tokens=True) for pred in outputs]
            print('Predicted labels: ', predicted_labels, end='\t')
            true_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            print('True labels: ', true_labels, end='\t')
            rouge_score = rouge(predicted_labels, true_labels)["rouge1_fmeasure"]
            print('Rouge: ', rouge_score)
            rouge_accumulated += rouge_score

    avg_rouge1_fmeasure_train = rouge_train_accumulated / num_train_batches
    avg_rouge1_fmeasure_test = rouge_accumulated / num_batches

    with open("log.txt", 'a') as f:
        print(f"Epoch {epoch + 1}/{epochs}", file=f)
        print(f"The avg rouge1_fmeasure for training data: {avg_rouge1_fmeasure_train}", file=f)
        print(f"The avg rouge1_fmeasure for testing data: {avg_rouge1_fmeasure_test}", file=f)

model.save_pretrained("fine_tuned_flan-t5-base")
end_time = time.time()
with open("log.txt", 'a') as f:
    print(f"Total time: {round((end_time - start_time) / 60, 2)} minutes", file=f)
