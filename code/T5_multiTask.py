from torch.utils.data import DataLoader
import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel, AdamW
from rdkit import Chem
from torchmetrics.text import BLEUScore, ROUGEScore
from torchmetrics import CharErrorRate, SacreBLEUScore
import argparse as arg
from sklearn.metrics import accuracy_score, f1_score
from peft import get_peft_model, LoraConfig, TaskType



parser = arg.ArgumentParser()
parser.add_argument("-o", "--output_file_name", type=str, default="unknown", )
args = parser.parse_args()
def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, T5_tokenizer, esm_tokenizer, max_length=850):
        self.file_path = file_path
        self.data = self.load_data()
        self.T5_tokenizer = T5_tokenizer
        self.esm_tokenizer = esm_tokenizer
        self.max_length = max_length

    def load_data(self):
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                if line != '\n':
                    text = line.split('\t')[0]
                    task = text.split(':')[0]
                    label = line.split('\t')[1].strip('\n')

                    if task == 'ProteinSeqs2SMILE':
                        text_list = text.split('_')[1:]
                        if all(len(element) <= 850 for element in text_list):
                            data.append((text_list, label, task))
                    else:
                        text_list = text
                        # Check if any element in text_list is longer than 2000 characters
                        if all(len(element) <= 1750 for element in text_list):
                            data.append((text_list, label, task))

        print(len(data))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label, task = self.data[idx]
        input_encoding = self.T5_tokenizer(text, return_tensors="pt", max_length=self.max_length, padding="max_length",
                                           truncation=True)
        target_encoding = self.T5_tokenizer(label, return_tensors="pt", max_length=250, padding="max_length",
                                            truncation=True)

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
            "text_list": text,
            "label": label,
            "task": task
        }


# Function to process input sequences with the ESM2 model and return hidden states
def get_esm_hidden_states(input_text):
    esm_input_tokens = esm_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    esm_input_ids = esm_input_tokens['input_ids'].to(device)  # Move to device
    esm_attention_mask = esm_input_tokens['attention_mask'].to(device)  # Move to device
    esm_outputs = esm_model(input_ids=esm_input_ids, attention_mask=esm_attention_mask)
    return esm_outputs[0]


def pad_to_match(tensor1, tensor2, dim):
    size1 = tensor1.size(dim)
    size2 = tensor2.size(dim)
    diff = size1 - size2

    if diff > 0:
        padding = (0, 0, 0, diff)  # (left, right, top, bottom) padding
        tensor2 = nn.functional.pad(tensor2, padding)
    elif diff < 0:
        padding = (0, 0, 0, -diff)
        tensor1 = nn.functional.pad(tensor1, padding)

    return tensor1, tensor2


def concat_seqs(text):
    concat_hidden_states = None
    hidden_states_list = []
    # Pad all items to the maximum length and concatenate
    for item in text:
        # gets the ESM encoding/hidden_states for each protein sequence in the input list 'text'
        hidden_states_list.append(get_esm_hidden_states(item))

    max_length = max(item_hidden_states.size(1) for item_hidden_states in hidden_states_list)

    padded_hidden_states_list = [
        pad_to_match(item_hidden_states, torch.zeros(1, max_length, item_hidden_states.size(2)), dim=1)[0] for
        item_hidden_states in hidden_states_list]
    concat_hidden_states = torch.cat(padded_hidden_states_list, dim=1)
    return concat_hidden_states


def evaluate(pred, true):
    if type(true) != list:
        true = [true]
        print("\n\ntrue", true, type(true))
    if type(pred) != list:
        pred = [pred]
        print("\n\npred", pred, type(pred))

    if len(pred) == 0:
        pred = " "
    rouge_score = rouge(pred, true)["rouge1_fmeasure"]
    char_error_rate_score = char_error_rate(pred, true).item()

    # Assign a default value to sacre_bleu_score
    sacre_bleu_score = None
    try:
        sacre_bleu_score = sacre_bleu(pred, true).item()
    except:
        print("pred", pred, type(pred), "true", true, type(true))
    print("pred", pred, type(pred), "true", true, type(true))

    accuracy = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='weighted')
    bleu_score = bleu(pred, true)  # WHY index 0?
    print("rouge_score", rouge_score, "bleu_score", bleu_score, "char_error_rate_score", char_error_rate_score, "sacre_bleu_score", sacre_bleu_score, "accuracy", accuracy, "f1", f1)
    return rouge_score, bleu_score, char_error_rate_score, sacre_bleu_score, accuracy, f1


load_model_continue_training = False

# Set up the training parameters
num_epochs = 8
learning_rate = 5e-5
batch_size = 1

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)
peft_config_esm = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)
T5_model_name = 'GT4SD/multitask-text-and-chemistry-t5-base-augm'
t5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if load_model_continue_training:
    checkpoint = torch.load(f"t5_model_{args.output_file_name}.pt")

    t5_model = T5ForConditionalGeneration.from_pretrained(T5_model_name)  # Initialize a new model
    t5_model.load_state_dict(checkpoint['model_state_dict'])  # Load the model state dict

    optimizer = torch.optim.Adam(t5_model.parameters())  # Initialize a new optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Load the optimizer state dict

    epoch = checkpoint['epoch']  # Load the number of epochs and loss completed so far
    loss = checkpoint['loss']

    checkpoint = torch.load(f"t5_model_{args.output_file_name}.pt", map_location=torch.device(device))
else:

    t5_config = T5Config.from_pretrained(T5_model_name)
    t5_model = T5ForConditionalGeneration.from_pretrained(T5_model_name, config=t5_config)


esm_model_name = "facebook/esm2_t6_8M_UR50D"
esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
esm_model = AutoModel.from_pretrained(esm_model_name)

projection = nn.Linear(esm_model.config.hidden_size, t5_config.d_model)


t5_model.to(device)
esm_model.to(device)
projection.to(device)
print(device)

train_dataset = ProteinDataset("train_combined_multitask_incl_protein_seq_v3_0.txt", t5_tokenizer, esm_tokenizer)
test_dataset = ProteinDataset("test_combined_multitask_incl_protein_seq_v3_0.txt", t5_tokenizer, esm_tokenizer)
valid_dataset = ProteinDataset("validation_combined_multitask_incl_protein_seq_v3_0.txt", t5_tokenizer, esm_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

# optimizer = AdamW(list(t5_model.parameters()) + list(esm_model.parameters()) + list(projection.parameters()), lr=learning_rate)
optimizer = torch.optim.AdamW(t5_model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=1, factor=0.1)


rouge = ROUGEScore()
bleu = BLEUScore()
char_error_rate = CharErrorRate()
sacre_bleu = SacreBLEUScore()

file = 'lolol'
output_lenght = 350
evaluation_results = {}
evaluation_results_train = {}
names_of_eval = ["rouge_test_accumulated", "char_error_rate_test_accumulated", "sacre_bleu_test_accumulated",
                 "test_accuracy_accumulated", "test_f1_accumulated"]
train_eval_name = ["rouge_train_accumulated", "char_error_rate_train_accumulated", "sacre_bleu_train_accumulated",
                   "train_accuracy_accumulated", "train_f1_accumulated"]

t5_model = get_peft_model(t5_model, peft_config)
esm_model = get_peft_model(esm_model, peft_config_esm)

# Training loop
for epoch in range(num_epochs):
    print("Epoch: ", epoch)

    t5_model.train()

    num_train_batches = 0
    for batch in train_loader:
        num_train_batches += len(batch["labels"])
        rouge_train_accumulated, bleu_train_accumulated, Num_correct_val_mols_train, char_error_rate_train_accumulated, sacre_bleu_train_accumulated, train_accuracy_accumulated, train_f1_accumulated = 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0

        task = batch["task"][0]

        if task not in evaluation_results:
            evaluation_results[task] = {}
        if epoch not in evaluation_results[task]:
            evaluation_results[task][epoch] = {}
        if task not in evaluation_results_train:
            evaluation_results_train[task] = {}
        if epoch not in evaluation_results_train[task]:
            evaluation_results_train[task][epoch] = {}

        if str(task) == 'ProteinSeqs2SMILE':
            text = batch["text_list"]
            labels = batch["labels"].to(device)

            concat_hidden_states = concat_seqs(text)

            projected_hidden_states = projection(concat_hidden_states)
            optimizer.zero_grad()

            decoder_input_ids = torch.cat(
                (torch.full((labels.size(0), 1), 0, dtype=torch.long, device=device), labels[:, :-1]), dim=-1)

            t5_outputs = t5_model(
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=(projected_hidden_states, None),
                labels=labels,
            )

            with torch.no_grad():
                train_predicted_labels = t5_tokenizer.decode(t5_outputs.logits[0].argmax(dim=-1).tolist(),
                                                             skip_special_tokens=True, num_of_beams=5, max_new_tokens=output_lenght)

                train_rouge_score, train_bleu_score, train_char_error_rate_score, train_sacre_bleu_score, train_accuracy, train_f1 = evaluate(train_predicted_labels, train_true_labels)
                train_eval_value = [train_rouge_score, train_bleu_score, train_char_error_rate_score,
                                    train_sacre_bleu_score, train_accuracy, train_f1]

                for name, value in zip(train_eval_name, train_eval_value):
                    if value is None:
                        value = 0.0
                    if name not in evaluation_results_train[task][epoch]:
                        evaluation_results_train[task][epoch][name] = value
                    else:
                        evaluation_results_train[task][epoch][name] += value

            loss = t5_outputs.loss
            loss.backward()
            optimizer.step()

        else:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            train_true_labels = [batch["label"][0]]
            if len(input_ids.shape) == 2:
                outputs = t5_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()

                with torch.no_grad():
                    train_outputs = t5_model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=12, max_new_tokens=output_lenght)
                    train_predicted_labels = [t5_tokenizer.decode(pred, skip_special_tokens=True) for pred in train_outputs]
                    train_true_labels = [t5_tokenizer.decode(label, skip_special_tokens=True) for label in labels]

                    train_rouge_score, train_bleu_score, train_char_error_rate_score, train_sacre_bleu_score, train_accuracy, train_f1 = evaluate(train_predicted_labels, train_true_labels)
                    train_eval_value = [train_rouge_score, train_bleu_score, train_char_error_rate_score, train_sacre_bleu_score, train_accuracy, train_f1]

                    for name, value in zip(train_eval_name, train_eval_value):
                        if name not in evaluation_results_train[task][epoch]:
                            evaluation_results_train[task][epoch][name] = value
                        else:
                            evaluation_results_train[task][epoch][name] += value

    for task in evaluation_results_train:
        evaluation_results_train[task][epoch] = {name: value / num_train_batches for name, value in evaluation_results_train[task][epoch].items()}

    ### validation loop
    t5_model.eval()
    esm_model.eval()
    projection.eval()

    valid_loss = 0.0
    valid_batches = 0
    for batch in valid_loader:
        # Similar to your training and test loops, calculate the loss for the validation set
        # Be sure to call loss.item() to get a Python number, and not a one-element tensor
        valid_loss += loss.item()
        valid_batches += len(batch["labels"])

    # After the validation loop, calculate the average validation loss
    valid_loss /= valid_batches

    # Update the learning rate based on the validation loss
    scheduler.step(valid_loss)

    num_test_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            rouge_test_accumulated, bleu_test_accumulated, char_error_rate_test_accumulated, sacre_bleu_test_accumulated, Num_correct_val_mols_test, test_accuracy_accumulated, test_f1_accumulated = 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0
            num_test_batches += len(batch["labels"])

            text = batch["text_list"]
            task = batch["task"][0]

            labels = batch["labels"].to(device)
            if str(task) == 'ProteinSeqs2SMILE':
                concat_hidden_states = concat_seqs(text)

                projected_hidden_states = projection(concat_hidden_states)

                decoder_input_ids = torch.cat(
                    (torch.full((labels.size(0), 1), 0, dtype=torch.long, device=device), labels[:, :-1]), dim=-1)

                test_outputs = t5_model(
                    input_ids=None,
                    attention_mask=None,
                    decoder_input_ids=decoder_input_ids,
                    encoder_outputs=(projected_hidden_states, None),
                    labels=labels,
                )

                test_predicted_labels = t5_tokenizer.decode(test_outputs.logits[0].argmax(dim=-1).tolist(),
                                                            skip_special_tokens=True, max_new_tokens=output_lenght)
                test_true_labels = [batch["label"][0]]

                test_rouge_score, test_bleu_score, test_char_error_rate_score, test_sacre_bleu_score, test_accuracy, test_f1 = evaluate(
                    test_predicted_labels, test_true_labels)

                value_of_eval = [test_rouge_score, test_bleu_score, test_char_error_rate_score, test_sacre_bleu_score,
                                 test_accuracy, test_f1]

                for name, value in zip(names_of_eval, value_of_eval):
                    if value is None:
                        value = 0.0
                    if name not in evaluation_results[task][epoch]:
                        evaluation_results[task][epoch][name] = value
                    else:
                        evaluation_results[task][epoch][name] += value

            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                with torch.no_grad():
                    outputs = t5_model.generate(input_ids=input_ids, attention_mask=attention_mask, num_beams=6,
                                                max_new_tokens=output_lenght)
                    test_predicted_labels = [t5_tokenizer.decode(pred, skip_special_tokens=True) for pred in outputs]
                    test_true_labels = [t5_tokenizer.decode(label, skip_special_tokens=True) for label in labels]

                    test_rouge_score, test_bleu_score, test_char_error_rate_score, test_sacre_bleu_score, test_accuracy, test_f1 = evaluate(test_predicted_labels, test_true_labels)
                    value_of_eval = [test_rouge_score, test_bleu_score, test_char_error_rate_score, test_sacre_bleu_score, test_accuracy, test_f1]

                    for name, value in zip(names_of_eval, value_of_eval):
                        if name not in evaluation_results[task][epoch]:
                            evaluation_results[task][epoch][name] = value
                        else:
                            evaluation_results[task][epoch][name] += value

            with open(f"predictions_{args.output_file_name}.txt", "a") as predictions_file:
                print(f"Epoch {epoch + 1}/{num_epochs}\ttask: {task}\tTrue: {test_true_labels}\tPred: {test_predicted_labels}",
                      file=predictions_file)
    # divede by num_test_batches for each task and each name
    for task in evaluation_results:
        evaluation_results[task][epoch] = {name: value / num_test_batches for name, value in evaluation_results[task][epoch].items()}


    with open(f"score_{args.output_file_name}.txt", "a") as scores_file:
        print(f"Epoch {epoch + 1}/{num_epochs}\tTrain: {evaluation_results_train}\nTest: {evaluation_results}", file=scores_file)
    # save model
    torch.save({
        'epoch': epoch,
        'model_state_dict': t5_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f"t5_model_{args.output_file_name}.pt")

# save evaluation_results as a json file
import json
with open(f"evaluation_results_{args.output_file_name}.json", "w") as evaluation_results_file:
    json.dump(evaluation_results, evaluation_results_file)
with open(f"evaluation_results_train_{args.output_file_name}.json", "w") as evaluation_results_train_file:
    json.dump(evaluation_results_train, evaluation_results_train_file)