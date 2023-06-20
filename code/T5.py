import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel, AdamW
import time
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse as arg
from torchmetrics.text import BLEUScore, ROUGEScore
from torchmetrics import CharErrorRate, SacreBLEUScore
from rdkit import Chem

def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

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
                    print(len(text), len(label))
                    if len(text) < 500:
                        self.data.append((text, label, task))
                    else:
                        print('something wrong')
        print(len(self.data))
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label, task = self.data[idx]
        input_encoding = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding="max_length")
        target_encoding = self.tokenizer(label, return_tensors="pt", max_length=400, padding="max_length",
                                         truncation=True)

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
            "task": task
        }


start_time = time.time()

# Assume you have a T5 model and tokenizer already
T5_model_name = 'GT4SD/multitask-text-and-chemistry-t5-base-augm'
t5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)


train_dataset = Dataset("dataset/invalid2validSMILE/train_invalid2validSMILE.txt", t5_tokenizer)
test_dataset = Dataset("dataset/invalid2validSMILE/test_invalid2validSMILE.txt", t5_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


learning_rate = 5e-5
optimizer = AdamW(list(t5_model.parameters()), lr=learning_rate)

rouge = ROUGEScore()
bleu = BLEUScore()
char_error_rate = CharErrorRate()
sacre_bleu = SacreBLEUScore()

num_epochs = 50

# Training loop
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    t5_model.train()
    rouge_train_accumulated = 0.0
    bleu_train_accumulated = 0.0
    char_error_rate_train_accumulated = 0.0
    sacre_bleu_train_accumulated = 0.0
    num_train_batches = 0
    Num_correct_val_mols_train = 0


    for batch in train_loader:
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        num_train_batches += 1

        # compute the model output
        outputs = t5_model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # compute the metrics for each output
        with torch.no_grad():
            train_predicted_labels = t5_tokenizer.decode(outputs.logits[0].argmax(dim=-1).tolist(),
                                                         skip_special_tokens=True)
            train_true_labels = [t5_tokenizer.decode(label.tolist(), skip_special_tokens=True) for label in batch["labels"]]
            if is_valid_smiles(train_predicted_labels):
                Num_correct_val_mols_train += 1

            with open(f"predictions_{args.output_file_name}.txt", "a") as predictions_file:
                print(f"Epoch {epoch + 1}/{num_epochs}\tTrue: {train_true_labels}\tPred: {train_predicted_labels}", file=predictions_file)

            train_rouge_score = rouge(train_predicted_labels, train_true_labels)["rouge1_fmeasure"]
            train_bleu_score = bleu(train_predicted_labels.split(), [train_true_labels[0].split()])
            train_char_error_rate_score = char_error_rate(train_predicted_labels, train_true_labels).item()
            train_sacre_bleu_score = sacre_bleu([train_predicted_labels], [train_true_labels]).item()

            rouge_train_accumulated += train_rouge_score
            bleu_train_accumulated += train_bleu_score
            char_error_rate_train_accumulated += train_char_error_rate_score
            sacre_bleu_train_accumulated += train_sacre_bleu_score

    # Similar loop for testing
    t5_model.eval()
    rouge_test_accumulated = 0.0
    bleu_test_accumulated = 0.0
    char_error_rate_test_accumulated = 0.0
    sacre_bleu_test_accumulated = 0.0
    num_test_batches = 0
    Num_correct_val_mols_test = 0
    test_outputs = []

    for batch in test_loader:
        num_test_batches += 1
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = t5_model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            test_predicted_labels = t5_tokenizer.decode(outputs.logits[0].argmax(dim=-1).tolist(),
                                                        skip_special_tokens=True)
            test_true_labels = [t5_tokenizer.decode(label.tolist(), skip_special_tokens=True) for label in batch["labels"]]

            test_outputs.append({"predicted_label": test_predicted_labels, "true_label": test_true_labels[0]})

            if is_valid_smiles(test_predicted_labels):
                Num_correct_val_mols_test += 1

            with open(f"predictions_{args.output_file_name}.txt", "a") as predictions_file:
                print(f"Epoch {epoch + 1}/{num_epochs}\tTrue: {test_true_labels}\tPred: {test_predicted_labels}",
                      file=predictions_file)

            test_rouge_score = rouge(test_predicted_labels, test_true_labels)["rouge1_fmeasure"]
            test_bleu_score = bleu(test_predicted_labels.split(), [test_true_labels[0].split()])
            test_char_error_rate_score = char_error_rate(test_predicted_labels, test_true_labels).item()
            test_sacre_bleu_score = sacre_bleu([test_predicted_labels], [test_true_labels]).item()

            rouge_test_accumulated += test_rouge_score
            bleu_test_accumulated += test_bleu_score
            char_error_rate_test_accumulated += test_char_error_rate_score
            sacre_bleu_test_accumulated += test_sacre_bleu_score

    # Print and save results for this epoch
    with open(f"scores_{args.output_file_name}.txt", "a") as scores_file:
        print(
            f"Epoch {epoch + 1}/{num_epochs}\t Avg Train ROUGE-1 F1 Score\t {rouge_train_accumulated / num_train_batches}\tAvg Train BLEU Score\t {bleu_train_accumulated / num_train_batches}\tAvg Train Char Error Rate\t {char_error_rate_train_accumulated / num_train_batches}\tAvg Train SacreBLEU Score\t {sacre_bleu_train_accumulated / num_train_batches}\tNum correct val mols train: {Num_correct_val_mols_train}",
            file=scores_file)

        print(
            f"Epoch {epoch + 1}/{num_epochs}\t Avg Test ROUGE-1 F1 Score\t {rouge_test_accumulated / num_test_batches}\tAvg Test BLEU Score\t {bleu_test_accumulated / num_test_batches}\tAvg Test Char Error Rate\t {char_error_rate_test_accumulated / num_test_batches}\tAvg Test SacreBLEU Score\t {sacre_bleu_test_accumulated / num_test_batches}\tNum correct val mols test: {Num_correct_val_mols_test}",
            file=scores_file)
