import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config, AutoModel, AutoTokenizer
import argparse
import time
from torch.cuda.amp import GradScaler, autocast

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logfile', type=str, help='name of the log file')
parser.add_argument('-t1tr', '--task1_trainfile', type=str, help='name of the task 1 training file')
parser.add_argument('-t1te', '--task1_testfile', type=str, help='name of the task 1 test file')
parser.add_argument('-t2tr', '--task2_trainfile', type=str, help='name of the task 2 training file')
parser.add_argument('-t2te', '--task2_testfile', type=str, help='name of the task 2 test file')
args = parser.parse_args()

class Task1Dataset(Dataset):
    def __init__(self, filename, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.data = []
        with open(filename, "r") as f:
            for line in f:
                text, label = line.strip().split("\t")
                label = "1" if label == "1" else "0"
                self.data.append((text, label))

        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        input_encoding = self.tokenizer(f"AbstractClassification: {text}", return_tensors="pt", max_length=self.max_length, padding="max_length", truncation=True)
        target_encoding = self.tokenizer(label, return_tensors="pt", max_length=2, padding="max_length", truncation=True)

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }


class Task2Dataset(Dataset):
    def __init__(self, filename, tokenizer, esm_tokenizer, esm_model, max_length=8000):
        self.tokenizer = tokenizer
        self.esm_tokenizer = esm_tokenizer
        self.esm_model = esm_model
        self.data = []
        with open(filename, "r") as f:
            for line in f:
                text, label = line.strip().split("\t")
                self.data.append((text, label))

        self.max_length = max_length

    def _process_esm_output(self, text, max_chunk_length=4096):
        # Tokenize the input text and get the number of tokens
        esm_input_encoding = self.esm_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
        num_tokens = esm_input_encoding["input_ids"].size(1)

        # Calculate the number of chunks needed
        num_chunks = (num_tokens + max_chunk_length - 1) // max_chunk_length

        # Process each chunk and aggregate the results
        esm_output_reprs = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * max_chunk_length
            end_idx = min((chunk_idx + 1) * max_chunk_length, num_tokens)

            # Get the input encoding for the current chunk
            chunk_input_encoding = {k: v[:, start_idx:end_idx] for k, v in esm_input_encoding.items()}

            # Process the chunk using the ESM model
            esm_output = self.esm_model(**chunk_input_encoding)

            # Compute the mean-pooling for the current chunk
            chunk_esm_output_repr = torch.mean(esm_output[0], dim=1)
            esm_output_reprs.append(chunk_esm_output_repr)

        # Concatenate and average the mean-pooled representations of all chunks
        esm_output_repr = torch.mean(torch.cat(esm_output_reprs, dim=0), dim=0, keepdim=True)
        return esm_output_repr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        esm_output_repr = self._process_esm_output(text)
        input_encoding = self.tokenizer(f"protein2Smile: {esm_output_repr}", return_tensors="pt", max_length=self.max_length,
                                        padding="max_length", truncation=True)
        target_encoding = self.tokenizer(label, return_tensors="pt", max_length=200, padding="max_length",
                                         truncation=True)

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }

class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, idx):
        for d in self.datasets:
            if idx < len(d):
                return d[idx]
            idx -= len(d)
        raise IndexError("Index out of range")


start_time = time.time()

# Load ESM model and tokenizer
esm_model_name = "facebook/esm2_t6_8M_UR50D" # smallest model
esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
esm_model = AutoModel.from_pretrained(esm_model_name)


model_name = "google/flan-t5-base"
tokenizer = T5TokenizerFast.from_pretrained(model_name)
config = T5Config.from_pretrained(model_name)
config.n_positions = 26000 # max length needed for protein sequences > 25,000
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)

task1_train_dataset = Task1Dataset(args.task1_trainfile, tokenizer)
task1_test_dataset = Task1Dataset(args.task1_testfile, tokenizer)
task2_train_dataset = Task2Dataset(args.task2_trainfile, tokenizer, esm_tokenizer, esm_model)
task2_test_dataset = Task2Dataset(args.task2_testfile, tokenizer, esm_tokenizer, esm_model)

train_dataset = ConcatDataset(task1_train_dataset, task2_train_dataset)
test_dataset = ConcatDataset(task1_test_dataset, task2_test_dataset)

if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 1
epochs = 5
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
scaler = GradScaler()

with open(args.logfile, 'w') as f:
    f.write(f"Model name: {model_name}, Task 1 Train file: {args.task1_trainfile}, Task 1 Test file: {args.task1_testfile}, Task 2 Train file: {args.task2_trainfile}, Task 2 Test file: {args.task2_testfile}, Batch size: {batch_size}, Epochs: {epochs}, Device: {device}\n\n")



accumulation_steps = 4  # Update the model every 4 steps

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

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

