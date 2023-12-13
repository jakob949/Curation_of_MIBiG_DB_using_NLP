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
# from peft import get_peft_model, LoraConfig, TaskType

def count_valid_smiles(smiles_list: list) -> int:
    valid_count = 0
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_count += 1
    return valid_count


parser = arg.ArgumentParser()
parser.add_argument("-o", "--output_file_name", type=str, default="unknown")
parser.add_argument("-i", "--information", type=str, default="None")
parser.add_argument("-s", "--sampling", type=bool, default=False)
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

                if len(text) > 800:
                    data.append((text[:800], label))
                    num_of_truncs += 1
                else:
                    data.append((text, label))


                # # Check if any element in text_list is longer than 2000 characters
                # if all(len(element) <= 800 for element in text_list):
                #     data.append((text, label))
                # else:
                #     truncated_text_list = [element[:100] for element in text_list]
                #     for item in truncated_text_list:
                #         data.append((item, label))
                #         if len(item) > 99:
                #             num_of_truncs += 1

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


start_time = time.time()


# T5_model_name = 'google/t5-efficient-tiny'
T5_model_name = 'GT4SD/multitask-text-and-chemistry-t5-base-augm'
# T5_model_name = 'model_020623_geneProduct2SMILES_v3.pt'
t5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)
# t5_model = torch.load(T5_model_name)

t5_model = T5ForConditionalGeneration.from_pretrained(T5_model_name)
# Lora peft
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.3
# )
# t5_model = get_peft_model(t5_model, peft_config)
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)
t5_model = torch.nn.DataParallel(t5_model)


#load data
# dataset/train_i2v_BGC2SMM_250923.txt => no bias set
# train_dataset = Dataset("train_text2SMILES_I2V_gio_method_base_correct_format.txt", t5_tokenizer)
train_dataset = Dataset("dataset/pfam2SMILES/train_pfam_v2.txt", t5_tokenizer)
# test_dataset = Dataset("test_text2SMILES_I2V_gio_method_base_correct_format.txt", t5_tokenizer)
test_dataset = Dataset("dataset/pfam2SMILES/test_pfam_v2.txt", t5_tokenizer)

num_workers = 4
batch_size_train = 1

# Modify the DataLoader instances
train_loader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False, num_workers=num_workers)

# Optimizer
learning_rate = 4e-4
optimizer = AdamW(list(t5_model.parameters()), lr=learning_rate, weight_decay = 0.05)

rouge = ROUGEScore()
bleu = BLEUScore()
char_error_rate = CharErrorRate()
sacre_bleu = SacreBLEUScore()

num_epochs = 20
train_sampling_predictions = []
test_sampling_predictions = []
sampling = args.sampling
print(sampling, "type: ", type(sampling))
num_gen_seqs = 5

with open(f"information_{args.output_file_name}.txt", "w") as predictions_file:
    print(">T5 model: ", T5_model_name, " Cuda available:", device, file=predictions_file)
    print(f">Learning rate: {learning_rate}, num of epoch: {num_epochs}, train batch size: {batch_size_train}", file=predictions_file)
    print(f">Dataset: {train_dataset.file_path}", file=predictions_file)
    print(f">Information: {args.information}", file=predictions_file)
    # print(">peft: ", t5_model.print_trainable_parameters(), file=predictions_file)

# Training loop
for epoch in range(num_epochs):
    # print(f"Epoch {epoch + 1}/{num_epochs}")
    t5_model.train()
    rouge_train_accumulated, bleu_train_accumulated, char_error_rate_train_accumulated, sacre_bleu_train_accumulated = 0.0, 0.0, 0.0, 0.0
    num_train_batches = 0
    Num_correct_val_mols_train = 0
    train_accuracy_accumulated = 0.0

    for batch in train_loader:
        task_train = []
        for item in batch["text"]:
            task_train.append(item.split(': ')[0])
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        num_train_batches += 1
        train_true_labels = [t5_tokenizer.decode(label.tolist(), skip_special_tokens=True) for label in batch["labels"]]
        outputs = t5_model(input_ids=inputs, attention_mask=attention_mask, labels=labels)

        if epoch == 0 and sampling:
            # Generate predictions for each input
            generated_ids = t5_model.module.generate(inputs, attention_mask=attention_mask, num_beams=5,
                                                     num_return_sequences=num_gen_seqs, temperature=0.7)

            # Reshape generated_ids to match (num_inputs, num_gen_seqs, sequence_length)
            generated_ids = generated_ids.view(inputs.size(0), num_gen_seqs, -1)

            # Decode generated ids to text and save them
            for i in range(inputs.size(0)):
                generated_texts = [t5_tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in
                                   generated_ids[i]]
                input_text = t5_tokenizer.decode(batch["input_ids"][i].tolist(), skip_special_tokens=True)

                # Saving predictions
                with open(f'train_sampling_{num_gen_seqs}_for_iv2_{args.output_file_name}.txt', 'a') as file:
                    for generated_text in generated_texts:
                        line = f"Input: {input_text}\tGenerated: {generated_text}\n"
                        file.write(line)

        loss = outputs.loss
        loss = loss.mean()
        print(loss)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # compute the metrics for each output
        with torch.no_grad():
            train_predicted_labels = [t5_tokenizer.decode(logits.argmax(dim=-1).tolist(), skip_special_tokens=True) for logits in outputs.logits]

            train_true_labels = [t5_tokenizer.decode(label.tolist(), skip_special_tokens=True) for label in batch["labels"]]
            Num_correct_val_mols_train += count_valid_smiles(train_predicted_labels)

            with open(f"predictions_train_{args.output_file_name}.txt", "a") as predictions_file:
                print(f"Epoch\t{epoch + 1}\tTrue:\t{train_true_labels}\tPred:\t{train_predicted_labels}\task\t{task_train}", file=predictions_file)

            train_rouge_score = rouge(train_predicted_labels, train_true_labels)["rouge1_fmeasure"]
            train_bleu_score = bleu(train_predicted_labels, train_true_labels)
            train_char_error_rate_score = char_error_rate(train_predicted_labels, train_true_labels)
            train_sacre_bleu_scores = [sacre_bleu([pred], [[true]]) for pred, true in zip(train_predicted_labels, train_true_labels)]
            train_sacre_bleu_score = sum(train_sacre_bleu_scores) / len(train_sacre_bleu_scores)

            print(train_rouge_score, train_bleu_score, train_char_error_rate_score, train_sacre_bleu_score, Num_correct_val_mols_train)

            batch_train_accuracy = accuracy_score(train_true_labels, train_predicted_labels)
            train_accuracy_accumulated += batch_train_accuracy
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
    test_accuracy_accumulated = 0.0

    for batch in test_loader:
        task_test = []
        for item in batch["text"]:
            task_test.append(item.split(': ')[0])
        num_test_batches += 1
        inputs = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        if epoch == 0 and sampling:
            # Generate predictions
            generated_ids = t5_model.module.generate(inputs, attention_mask=attention_mask, num_beams=5, num_return_sequences=num_gen_seqs, temperature=0.7)
            # Decode generated ids to text and save them
            generated_texts = [t5_tokenizer.decode(generated_id, skip_special_tokens=True) for generated_id in generated_ids]
            # saving predictions
            test_sampling_predictions.append((generated_texts, train_true_labels))

            with open(f'test_sampling_{num_gen_seqs}_for_iv2_{args.output_file_name}.txt', 'w') as file:
                for batch in test_sampling_predictions:
                    generated_texts, true_labels = batch
                    for i, true_label in enumerate(true_labels):
                        for j in range(num_gen_seqs):
                            line = f"iv2_sampling_{num_gen_seqs}: {generated_texts[i * num_gen_seqs + j]}\t{true_label}\n"
                            file.write(line)

        with torch.no_grad():
            outputs = t5_model(input_ids=inputs, attention_mask=attention_mask, labels=labels)
            test_predicted_labels = [t5_tokenizer.decode(logits.argmax(dim=-1).tolist(), skip_special_tokens=True) for logits in outputs.logits]
            test_true_labels = [t5_tokenizer.decode(label.tolist(), skip_special_tokens=True) for label in batch["labels"]]

            Num_correct_val_mols_test += count_valid_smiles(test_predicted_labels)

            with open(f"predictions_test_{args.output_file_name}.txt", "a") as predictions_file:
                print(f"Epoch {epoch + 1}\tTrue:\t{test_true_labels}\tPred:\t{test_predicted_labels}\ttask\t{task_test}",
                      file=predictions_file)

            test_rouge_score = rouge(test_predicted_labels, test_true_labels)["rouge1_fmeasure"]
            test_bleu_score = bleu(test_predicted_labels, test_true_labels)
            test_char_error_rate_score = char_error_rate(test_predicted_labels, test_true_labels)
            test_sacre_bleu_scores = [sacre_bleu([pred], [[true]]) for pred, true in zip(test_predicted_labels, test_true_labels)]
            test_sacre_bleu_score = sum(test_sacre_bleu_scores) / len(test_sacre_bleu_scores)
            batch_test_accuracy = accuracy_score(test_true_labels, test_predicted_labels)

            test_accuracy_accumulated += batch_test_accuracy
            rouge_test_accumulated += test_rouge_score
            bleu_test_accumulated += test_bleu_score
            char_error_rate_test_accumulated += test_char_error_rate_score
            sacre_bleu_test_accumulated += test_sacre_bleu_score

            print(test_rouge_score, test_bleu_score, test_char_error_rate_score, test_sacre_bleu_score, batch_test_accuracy, Num_correct_val_mols_test)

    # Print and save results for this epoch
    with open(f"scores_{args.output_file_name}.txt", "a") as scores_file:
        print(
            f"Epoch\t{epoch + 1}\tTrain Accuracy:\t{train_accuracy_accumulated / num_train_batches}\tAvg Train ROUGE-1 F1 Score\t{rouge_train_accumulated / num_train_batches}\tAvg Train BLEU Score\t{bleu_train_accumulated / num_train_batches}\tAvg Train Char Error Rate\t{char_error_rate_train_accumulated / num_train_batches}\tAvg Train SacreBLEU Score\t{sacre_bleu_train_accumulated / num_train_batches}\tNum correct val mols train\t{Num_correct_val_mols_train}",
            file=scores_file)

        print(
            f"Epoch\t{epoch + 1}\tTest Accuracy:\t{test_accuracy_accumulated / num_test_batches}\tAvg Test ROUGE-1 F1 Score\t{rouge_test_accumulated / num_test_batches}\tAvg Test BLEU Score\t{bleu_test_accumulated / num_test_batches}\tAvg Test Char Error Rate\t{char_error_rate_test_accumulated / num_test_batches}\tAvg Test SacreBLEU Score\t{sacre_bleu_test_accumulated / num_test_batches}\tNum correct val mols test\t{Num_correct_val_mols_test}",
            file=scores_file)
    # save the model
    # if epoch == 17:
    # if epoch > 2:
    torch.save(t5_model, f"models/model_{args.output_file_name}_{epoch}.pt")
    t5_model.module.config.to_json_file(f"models/config_{args.output_file_name}_{epoch}.json")
