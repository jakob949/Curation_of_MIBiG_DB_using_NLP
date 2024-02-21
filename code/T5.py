import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel, AdamW
import time
# from sklearn.metrics import accuracy_score, f1_score
# from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse as arg
# from torchmetrics.text import BLEUScore, ROUGEScore
# from torchmetrics import CharErrorRate, SacreBLEUScore
# from rdkit import Chem
from peft import get_peft_model, LoraConfig, TaskType


def count_valid_smiles(smiles_list: list) -> int:
    valid_count = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_count += 1
    return valid_count

def validate_smiles(smiles_list: list) -> float:
    loss_updata_factor = 1
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            loss_updata_factor *= 0.5
        else:
            loss_updata_factor *= 1.75
    return loss_updata_factor

parser = arg.ArgumentParser()
parser.add_argument("-o", "--output_file_name", type=str, default="unknown")
parser.add_argument("-i", "--information", type=str, default="None")
parser.add_argument("-s", "--sampling", type=bool, default=False)
parser.add_argument("-loss", "--loss", type=bool, default=False)
parser.add_argument("-tr", "--train", type=str, default="None")
parser.add_argument("-te", "--test", type=str, default="None")
parser.add_argument("-mo", "--model", type=str, default="None")
args = parser.parse_args()


class Dataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=2500):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_data()

    def load_data(self):
        data = []
        num_of_truncs = 0
        excluded_seqs = 0
        with open(self.file_path, 'r', encoding='ISO-8859-1') as f:
            for line in f:
                text = line.split('\t')[0]
                label = line.split('\t')[1].strip('\n')
                text_list = text.split(': ')[1].split('_')
                task = text.split(': ')[0]
                if task == 'GeneName2SMILE' or task == 'pfam2SMILES':
                    if len(text_list) < 6:
                        excluded_seqs += 1
                        continue

                if task == 'invalid2validSMILE':
                    data.append((text, label))

                if len(text) > self.max_length:
                    data.append((text[:self.max_length], label))
                    num_of_truncs += 1
                else:
                    data.append((text, label))


        print('Num of seqs: ', len(data), 'truncated seqs: ', num_of_truncs, 'excluded seqs: ', excluded_seqs)
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
            "label": label,
        }


start_time = time.time()

# Model configuration
if args.model == "None":
    T5_model_name = "GT4SD/multitask-text-and-chemistry-t5-base-augm"
    t5_tokenizer = T5Tokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")
else:
    T5_model_name = args.model
    try:
        t5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)
    except:
        t5_tokenizer = T5Tokenizer.from_pretrained("GT4SD/multitask-text-and-chemistry-t5-base-augm")

t5_model = T5ForConditionalGeneration.from_pretrained(T5_model_name, torch_dtype="auto")

# Lora peft model
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.3
)
t5_model = get_peft_model(t5_model, peft_config)

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)
t5_model = torch.nn.DataParallel(t5_model)

print('Model:', type(t5_model))
print('Devices:', t5_model.device_ids)

#load data

if args.train == "None" or args.test == "None":
    train_dataset = Dataset("dataset/Text2SMILES_Gio/i2v/train_i2v_text2smiles.txt", t5_tokenizer)
    test_dataset = Dataset("dataset/Text2SMILES_Gio/i2v/test_i2v_text2smiles.txt", t5_tokenizer)
else:
    train_dataset = Dataset(args.train, t5_tokenizer)
    test_dataset = Dataset(args.test, t5_tokenizer)

batch_size_train = 15
num_workers = 1

# Modify the DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

# Optimizer
learning_rate = 3e-4
optimizer = AdamW(list(t5_model.parameters()), lr=learning_rate, weight_decay = 0.0015, correct_bias = True, no_deprecation_warning = True)

num_epochs = 6

train_sampling_predictions = []
test_sampling_predictions = []
loss = args.loss

num_gen_seqs = 1
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
torch.cuda.empty_cache()
with open(f"information_{args.output_file_name}.txt", "w") as predictions_file:
    print(f">Learning rate: {learning_rate}, num of epoch: {num_epochs}, train batch size: {batch_size_train}", file=predictions_file)
    print(">T5 model: ", T5_model_name, " Cuda available:", device, file=predictions_file)
    print(f">Dataset: {train_dataset.file_path}", file=predictions_file)
    print(f">Information: {args.information}", file=predictions_file)

# Training loop
for epoch in range(num_epochs):
    # print(f"Epoch {epoch + 1}/{num_epochs}")
    t5_model.train()
    num_train_batches = 0

    for batch in train_loader:
        task_train = []
        for item in batch["text"]:
            task_train.append(item.split(': ')[0])
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        num_train_batches += 1
        outputs = t5_model(input_ids=inputs, attention_mask=attention_mask, labels=labels)

        if epoch == 15 and sampling:
            # Generate predictions for each input
            generated_ids = t5_model.module.generate(inputs, attention_mask=attention_mask, num_beams=5,
                                                     num_return_sequences=num_gen_seqs, temperature=0.7, max_new_tokens=500)

            # Reshape generated_ids to match (num_inputs, num_gen_seqs, sequence_length)
            generated_ids = generated_ids.view(inputs.size(0), num_gen_seqs, -1)

            # Decode generated ids to text and save them
            for i in range(inputs.size(0)):
                generated_texts = [t5_tokenizer.decode(generated_id, skip_special_tokens=True)
                                   for generated_id in generated_ids[i]]
                input_text = t5_tokenizer.decode(batch["input_ids"][i].tolist(), skip_special_tokens=True)
                true_label = t5_tokenizer.decode(batch["labels"][i].tolist(), skip_special_tokens=True)

                # Saving predictions with corresponding true labels
                with open(f'train_sampling_{num_gen_seqs}_for_iv2_{args.output_file_name}.txt', 'a') as file:
                    for generated_text in generated_texts:
                        line = f"iv2_sampling_{num_gen_seqs}: {generated_text}\t{true_label}\n"  # Pair each prediction with the true label
                        file.write(line)
                        # print(f"iv2_sampling_{num_gen_seqs}: {generated_text}\t{true_label}")


        with torch.no_grad():
            # use the generate function to get the predictions
            generated_ids = t5_model.module.generate(input_ids=inputs, attention_mask=attention_mask, max_length=1000)
            
            train_predicted_labels = [t5_tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]
            
            train_true_labels = batch["label"]
            print("train_true_labels: ", train_true_labels)
            train_true_labels = [t5_tokenizer.decode(label.tolist(), skip_special_tokens=True) for label in batch["labels"]]
            
            with open(f"predictions_train_{args.output_file_name}.txt", "a") as predictions_file:
                print(f"Epoch\t{epoch + 1}\tTrue:\t{train_true_labels}\tPred:\t{train_predicted_labels}\task\t{task_train}", file=predictions_file)

        loss = outputs.loss
        if args.loss:
            print("Alternating the loss: ", loss)
            loss *= validate_smiles(train_predicted_labels)
        loss.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()
    t5_model.eval()

    num_test_batches = 0


    for batch in test_loader:
        task_test = []
        for item in batch["text"]:
            task_test.append(item.split(': ')[0])
        num_test_batches += 1
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if epoch == 15 and sampling:
            # Generate predictions for each input
            generated_ids = t5_model.module.generate(inputs, attention_mask=attention_mask, num_beams=5,
                                                     num_return_sequences=num_gen_seqs, temperature=0.7, max_new_tokens=500)

            # Reshape generated_ids to match (num_inputs, num_gen_seqs, sequence_length)
            generated_ids = generated_ids.view(inputs.size(0), num_gen_seqs, -1)

            # Decode generated ids to text and save them
            for i in range(inputs.size(0)):
                generated_texts = [t5_tokenizer.decode(generated_id, skip_special_tokens=True)
                                   for generated_id in generated_ids[i]]
                true_label = t5_tokenizer.decode(batch["labels"][i].tolist(), skip_special_tokens=True)

                # Saving predictions with corresponding true labels
                with open(f'test_sampling_{num_gen_seqs}_for_iv2_{args.output_file_name}.txt', 'a') as file:
                        for generated_text in generated_texts:
                            line = f"iv2_sampling_{num_gen_seqs}: {generated_text}\t{true_label}\n"  # Pair each prediction with the true label
                            file.write(line)

        with torch.no_grad():
            # use the generate function to get the predictions
            outputs = t5_model.module.generate(input_ids=inputs, attention_mask=attention_mask, max_length=1000)
            test_true_labels = [t5_tokenizer.decode(label.tolist(), skip_special_tokens=True) for label in batch["labels"]]
            generated_ids = outputs.view(inputs.size(0), num_gen_seqs, -1)
            generated_texts = [t5_tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids[0]]

            with open(f"predictions_test_{args.output_file_name}.txt", "a") as predictions_file:
                print(f"Epoch\t{epoch + 1}\tTrue:\t{test_true_labels}\tPred:\t{generated_texts}\ttask\t{task_test}",file=predictions_file)
    t5_model.module.save_pretrained(f"models/model_{args.output_file_name}_{epoch}.pt")

