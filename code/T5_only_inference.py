import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel, AdamW
from sklearn.metrics import accuracy_score, f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse as arg
from torchmetrics.text import BLEUScore, ROUGEScore
from torchmetrics import CharErrorRate, SacreBLEUScore
from rdkit import Chem


parser = arg.ArgumentParser()
parser.add_argument("-o", "--output_file_name", type=str, default="unknown", )
args = parser.parse_args()
class Dataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=1050):
        self.file_path = file_path
        self.data = self.load_data()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self):
        data = []
        num_of_truncs = 0
        with open(self.file_path, 'r') as f:
            for line in f:
                text = line.split('\t')[0]
                label = line.split('\t')[1].strip('\n')
                text_list = text.split(': ')[1].split('_')
                task = text.split(': ')[0]
                if task == 'GeneName2SMILE' or task == 'pfam2SMILES':
                    if len(text_list) < 3:
                        num_of_truncs += 1
                        continue
                if task == 'ProteinSeqs2SMILE' or task == 'SMILE2Biosynclass':
                    print('ProteinSeqs2SMILE skipped')
                    continue
                    # possibly implement ESM2 encoder here
                if task == 'invalid2validSMILE':
                    data.append((text, label))

                if len(text) > 80000:
                    data.append((text[:800], label))
                    num_of_truncs += 1
                else:
                    data.append((text, label))

        print('Num of seqs: ', len(data), 'truncated seqs: ', num_of_truncs)
        return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        input_encoding = self.tokenizer(text, return_tensors="pt", max_length=self.max_length, padding="max_length")
        target_encoding = self.tokenizer(label, return_tensors="pt", max_length=1000, padding="max_length",
                                         truncation=True)

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
            "text": text,
        }


# T5_model_name = 'google/t5-efficient-tiny'
T5_model_name = 'GT4SD/multitask-text-and-chemistry-t5-base-augm'
t5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)
T5_model_name = 'models/model_020124_i2v_pfam2SMILES_0.pt'

t5_model = torch.load(T5_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)

# Check if the model is a DataParallel object
if isinstance(t5_model, torch.nn.DataParallel):
    model_to_use = t5_model.module
else:
    model_to_use = t5_model

#load data
dataset = Dataset("test_text2SMILES_I2V_gio_method_base_correct_format.txt", t5_tokenizer)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

print("starting inference")
with open("test_i2v_gios_data_using_a_saved_BGC2SMM_model_for_inference.txt", "w") as predictions_file:
    for batch in loader:
        with torch.no_grad():
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = t5_model.generate(input_ids=inputs, attention_mask=attention_mask, num_beams=10, max_length=512)
            true_labels = [t5_tokenizer.decode(label.tolist(), skip_special_tokens=True) for label in batch["labels"]]
            # Decode the predictions
            decoded_predictions = [t5_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
            print()
            print("text2SMILES_I2V: ", decoded_predictions[0], "\t", true_labels[0], file=predictions_file, sep="")
            print("text2SMILES_I2V: ", decoded_predictions[0], "\t", true_labels[0], sep="")
