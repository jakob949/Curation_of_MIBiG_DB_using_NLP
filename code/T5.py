import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config
import time
from rdkit import Chem
from torchmetrics.text import BLEUScore, ROUGEScore
from torchmetrics import CharErrorRate, SacreBLEUScore
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse as arg

parser = arg.ArgumentParser()
parser.add_argument("-o", "--output_file_name", type=str, default="unknown", )
args = parser.parse_args()


def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

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

batch_size = 6

epochs = 50
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


rouge = ROUGEScore()
bleu = BLEUScore()
char_error_rate = CharErrorRate()
sacre_bleu = SacreBLEUScore()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

with open("log.txt", 'w') as f:
    f.write(f"Model name: {model_name}, Batch size: {batch_size}, Epochs: {epochs}, Device: {device}\n\n")

for epoch in range(epochs):
    model.train()
    rouge_train_accumulated = 0.0
    bleu_train_accumulated = 0.0
    num_train_batches = 0
    Num_correct_val_mols_train = 0
    char_error_rate_train_accumulated = 0.0
    sacre_bleu_train_accumulated = 0.0

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
            train_outputs = model.generate(input_ids, attention_mask=attention_mask, num_beams=12)
            train_predicted_labels = [tokenizer.decode(pred, skip_special_tokens=True) for pred in train_outputs]
            train_true_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]

            # Evaluate metrics
            # Inside the training loop, after calculating train_rouge_score and train_bleu_score
            train_rouge_score = rouge(train_predicted_labels, train_true_labels)["rouge1_fmeasure"]
            train_char_error_rate_score = char_error_rate(train_predicted_labels, train_true_labels).item()
            train_sacre_bleu_score = sacre_bleu(train_predicted_labels, train_true_labels).item()
            train_bleu_score = bleu(train_predicted_labels, train_true_labels).item()

            # Accumulate the values of these metrics in separate variables
            char_error_rate_train_accumulated += train_char_error_rate_score
            sacre_bleu_train_accumulated += train_sacre_bleu_score
            rouge_train_accumulated += train_rouge_score
            bleu_train_accumulated += train_bleu_score

    # Update learning rate using scheduler
    scheduler.step()

    model.eval()

    rouge_test_accumulated = 0.0
    num_test_batches = 0
    bleu_test_accumulated = 0.0
    char_error_rate_test_accumulated = 0.0
    sacre_bleu_test_accumulated = 0.0
    Num_correct_val_mols_test = 0

    for batch in test_loader:
        num_test_batches += 1
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask, num_beams=6)
            test_predicted_labels = [tokenizer.decode(pred, skip_special_tokens=True) for pred in outputs]
            print('Predicted labels: ', test_predicted_labels, end='\t')
            test_true_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            print('True labels: ', test_true_labels, end='\t')
            rouge_score = rouge(test_predicted_labels, test_true_labels)["rouge1_fmeasure"]
            print('Rouge: ', rouge_score)

            test_rouge_score = rouge(test_predicted_labels, test_true_labels)["rouge1_fmeasure"]
            test_char_error_rate_score = char_error_rate(test_predicted_labels, test_true_labels).item()
            test_sacre_bleu_score = sacre_bleu(test_predicted_labels, test_true_labels).item()

            # Accumulate the values of these metrics in separate variables
            char_error_rate_test_accumulated += test_char_error_rate_score
            sacre_bleu_test_accumulated += test_sacre_bleu_score
            rouge_test_accumulated += test_rouge_score

            # print(f"test_true_labels: {test_true_labels}, test_predicted_labels: {test_predicted_labels}, test_rouge_score: {test_rouge_score}")
            if is_valid_smiles(test_predicted_labels):
                Num_correct_val_mols_test += 1
    with open(f"predictions_{args.output_file_name}.txt", "a") as predictions_file:
        print(f"Epoch {epoch + 1}/{epochs}\tTrue: {test_true_labels}\tPred: {test_predicted_labels}",
              file=predictions_file)

    with open(f"scores_{args.output_file_name}.txt", "a") as scores_file:
        print(
            f"Epoch {epoch + 1}/{epochs}\t Avg Train ROUGE-1 F1 Score\t {rouge_train_accumulated / num_train_batches}\tAvg Train BLEU Score\t {bleu_train_accumulated / num_train_batches}\tAvg Train Char Error Rate\t {char_error_rate_train_accumulated / num_train_batches}\tAvg Train SacreBLEU Score\t {sacre_bleu_train_accumulated / num_train_batches}\tNum correct val mols train: {Num_correct_val_mols_train}",
            file=scores_file)

        print(
            f"Epoch {epoch + 1}/{epochs}\t Avg Test ROUGE-1 F1 Score\t {rouge_test_accumulated / num_test_batches}\tAvg Test BLEU Score\t {bleu_test_accumulated / num_test_batches}\tAvg Test Char Error Rate\t {char_error_rate_test_accumulated / num_test_batches}\tAvg Test SacreBLEU Score\t {sacre_bleu_test_accumulated / num_test_batches}\tNum correct val mols test: {Num_correct_val_mols_test}",
            file=scores_file)