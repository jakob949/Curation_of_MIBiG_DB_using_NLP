from torch.utils.data import DataLoader
import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from rdkit import Chem
from torchmetrics.text import BLEUScore, ROUGEScore
from torchmetrics import CharErrorRate, SacreBLEUScore
import argparse as arg
from peft import get_peft_model, LoraConfig, TaskType

parser = arg.ArgumentParser()
parser.add_argument("-o", "--output_file_name", type=str, default="unknown", )
args = parser.parse_args()

def is_valid_smiles(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, T5_tokenizer, esm_tokenizer, max_length=851):
        self.file_path = file_path
        self.data = self.load_data()
        self.T5_tokenizer = T5_tokenizer
        self.esm_tokenizer = esm_tokenizer
        self.max_length = max_length

    def load_data(self):
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                text = line.split(': ')[1].split('\t')[0]
                label = line.split('\t')[1].strip('\n')
                text_list = text.split('_')

                # Check if any element in text_list is longer than 2000 characters
                if all(len(element) <= 851 for element in text_list):
                    data.append((text_list, label))
        print(len(data))
        return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        input_encoding = self.T5_tokenizer(text, return_tensors="pt", max_length=self.max_length, padding="max_length",
                                           truncation=True)
        target_encoding = self.T5_tokenizer(label, return_tensors="pt", max_length=250, padding="max_length",
                                            truncation=True)

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
            "text_list": text,
            "label": label
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

    padded_hidden_states_list = [pad_to_match(item_hidden_states, torch.zeros(1, max_length, item_hidden_states.size(2)), dim=1)[0] for item_hidden_states in hidden_states_list]
    concat_hidden_states = torch.cat(padded_hidden_states_list, dim=1)
    return concat_hidden_states

def calculate_metrics(true_labels, predicted_labels, rouge_accumulated, bleu_accumulated, Num_correct_val_mols, char_error_rate_accumulated, sacre_bleu_accumulated):


    rouge_score = rouge(predicted_labels, true_labels)["rouge1_fmeasure"]
    char_error_rate_score = char_error_rate(predicted_labels, true_labels).item()
    sacre_bleu_score = sacre_bleu([predicted_labels], [true_labels]).item()
    bleu_score = bleu(predicted_labels.split(), [true_labels[0].split()])

    # Accumulate the values of these metrics in separate variables
    char_error_rate_accumulated += char_error_rate_score
    sacre_bleu_accumulated += sacre_bleu_score
    rouge_accumulated += rouge_score
    bleu_accumulated += bleu_score
    if is_valid_smiles(predicted_labels):
        Num_correct_val_mols += 1

    return char_error_rate_accumulated, sacre_bleu_accumulated, rouge_accumulated, bleu_accumulated, Num_correct_val_mols

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

T5_model_name = 'GT4SD/multitask-text-and-chemistry-t5-base-augm'
t5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)
t5_config = T5Config.from_pretrained(T5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_model_name, config=t5_config)
t5_model = get_peft_model(t5_model, peft_config)

esm_model_name = "facebook/esm2_t6_8M_UR50D"
esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
esm_model = AutoModel.from_pretrained(esm_model_name)

projection = nn.Linear(esm_model.config.hidden_size, t5_config.d_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up the training parameters
num_epochs = 50
learning_rate = 1e-4
optimizer = AdamW(list(t5_model.parameters()), lr=learning_rate, weight_decay=0.07)

t5_model.to(device)
esm_model.to(device)
projection.to(device)

print(device)
validation_set = False
train_dataset = ProteinDataset("dataset/protein_SMILE/train_protein_peptides_complete_v3_3_shorten_0.txt", t5_tokenizer, esm_tokenizer)
test_dataset = ProteinDataset("dataset/protein_SMILE/test_protein_peptides_complete_v3_3_shorten_0.txt", t5_tokenizer, esm_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

if validation_set:
    validation_dataset = ProteinDataset("dataset/protein_SMILE/validation_protein_peptides_complete_v3_4_shorten_0.txt", t5_tokenizer, esm_tokenizer)
    validation_loader = DataLoader(validation_dataset, batch_size=1, shuffle=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience = 3, factor = 0.5)


rouge = ROUGEScore()
bleu = BLEUScore()
char_error_rate = CharErrorRate()
sacre_bleu = SacreBLEUScore()

t5_model = get_peft_model(t5_model, peft_config)
esm_model.eval()
projection.eval()

# Training loop
for epoch in range(num_epochs):

    t5_model.train()
    esm_model.train()
    # projection.train()
    num_train_batches = 0

    rouge_accumulated_train, bleu_accumulated_train, Num_correct_val_mols_train, char_error_rate_accumulated_train, sacre_bleu_accumulated_train = 0.0, 0.0, 0, 0.0, 0.0

    for batch in train_loader:
        optimizer.zero_grad()
        # Should be fixed - This only works for batch size 1...
        num_train_batches += 1

        text = batch["text_list"]
        labels = batch["labels"].to(device)
        concat_hidden_states = concat_seqs(text)
        projected_hidden_states = projection(concat_hidden_states)
        decoder_input_ids = torch.cat((torch.full((labels.size(0), 1), 0, dtype=torch.long, device=device), labels[:, :-1]), dim=-1)

        t5_outputs = t5_model(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(projected_hidden_states, None),
            labels=labels,
        )

        with torch.no_grad():
            train_predicted_labels = t5_tokenizer.decode(t5_outputs.logits[0].argmax(dim=-1).tolist(), skip_special_tokens=True, num_of_beams=10)
            train_true_labels = [batch["label"][0]]
            char_error_rate_accumulated_train, sacre_bleu_accumulated_train, rouge_accumulated_train, bleu_accumulated_train, Num_correct_val_mols_train = calculate_metrics(train_true_labels, train_predicted_labels, rouge_accumulated_train, bleu_accumulated_train, Num_correct_val_mols_train, char_error_rate_accumulated_train, sacre_bleu_accumulated_train)

        loss = t5_outputs.loss
        loss.backward()
        print('loss ', loss)
        optimizer.step()

    if validation_set:
        ### validation loop
        t5_model.eval()

        valid_loss = 0.0
        valid_batches = 0
        rouge_accumulated_valid, bleu_accumulated_valid, Num_correct_val_mols_valid, char_error_rate_accumulated_valid, sacre_bleu_accumulated_valid = 0.0, 0.0, 0, 0.0, 0.0
        with torch.no_grad():
            for batch in validation_loader:
                valid_batches += 1

                text = batch["text_list"]
                labels = batch["labels"].to(device)
                concat_hidden_states = concat_seqs(text)
                projected_hidden_states = projection(concat_hidden_states)
                decoder_input_ids = torch.cat(
                    (torch.full((labels.size(0), 1), 0, dtype=torch.long, device=device), labels[:, :-1]), dim=-1)

                valid_outputs = t5_model(
                    input_ids=None,
                    attention_mask=None,
                    decoder_input_ids=decoder_input_ids,
                    encoder_outputs=(projected_hidden_states, None),
                    labels=labels,
                )

                valid_loss += valid_outputs.loss.item()
                valid_predicted_labels = t5_tokenizer.decode(valid_outputs.logits[0].argmax(dim=-1).tolist(),
                                                             skip_special_tokens=True, num_of_beams=5)
                valid_true_labels = [batch["label"][0]]
                char_error_rate_accumulated_valid, sacre_bleu_accumulated_valid, rouge_accumulated_valid, bleu_accumulated_valid, Num_correct_val_mols_valid = calculate_metrics(valid_true_labels, valid_true_labels, rouge_accumulated_valid, bleu_accumulated_valid, Num_correct_val_mols_valid, char_error_rate_accumulated_valid, sacre_bleu_accumulated_valid)

            valid_loss /= valid_batches

            # Update the learning rate based on the validation loss??
            scheduler.step(valid_loss)

    ### test loop
    t5_model.eval()
    esm_model.eval()
    num_test_batches = 0
    rouge_accumulated_test, bleu_accumulated_test, Num_correct_val_mols_test, char_error_rate_accumulated_test, sacre_bleu_accumulated_test = 0.0, 0.0, 0, 0.0, 0.0

    with torch.no_grad():
        for batch in test_loader:
            num_test_batches += 1

            text = batch["text_list"]
            labels = batch["labels"].to(device)

            concat_hidden_states = concat_seqs(text)
            projected_hidden_states = projection(concat_hidden_states)
            decoder_input_ids = torch.cat((torch.full((labels.size(0), 1), 0, dtype=torch.long, device=device), labels[:, :-1]), dim=-1)

            test_outputs = t5_model(
                input_ids=None,
                attention_mask=None,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=(projected_hidden_states, None),
                labels=labels)

            test_predicted_labels = t5_tokenizer.decode(test_outputs.logits[0].argmax(dim=-1).tolist(), skip_special_tokens=True, num_of_beams=10)
            test_true_labels = [batch["label"][0]]
            char_error_rate_accumulated_test, sacre_bleu_accumulated_test, rouge_accumulated_test, bleu_accumulated_test, Num_correct_val_mols_test = calculate_metrics(test_true_labels, test_predicted_labels, rouge_accumulated_test, bleu_accumulated_test, Num_correct_val_mols_test, char_error_rate_accumulated_test, sacre_bleu_accumulated_test)

            with open(f"predictions_{args.output_file_name}.txt", "a") as predictions_file:
                print(f"Epoch {epoch + 1}/{num_epochs}\tTrue: {test_true_labels}\tPred: {test_predicted_labels}", file=predictions_file)

            # save pridictions for training set
            with open(f"predictions_train_{args.output_file_name}.txt", "a") as predictions_file:
                print(f"Epoch {epoch + 1}/{num_epochs}\tTrue: {train_true_labels}\tPred: {train_predicted_labels}", file=predictions_file)

        with open(f"scores_{args.output_file_name}.txt", "a") as scores_file:
            print(
                f"Epoch {epoch + 1}/{num_epochs}\t Avg Train ROUGE-1 F1 Score\t {rouge_accumulated_train / num_train_batches}\tAvg Train BLEU Score\t {bleu_accumulated_train / num_train_batches}\tAvg Train Char Error Rate\t {char_error_rate_accumulated_train / num_train_batches}\tAvg Train SacreBLEU Score\t {sacre_bleu_accumulated_train / num_train_batches}\tNum correct val mols train: {Num_correct_val_mols_train}",
                file=scores_file)

            print(
                f"Epoch {epoch + 1}/{num_epochs}\t Avg Test ROUGE-1 F1 Score\t {rouge_accumulated_test / num_test_batches}\tAvg Test BLEU Score\t {bleu_accumulated_test / num_test_batches}\tAvg Test Char Error Rate\t {char_error_rate_accumulated_test / num_test_batches}\tAvg Test SacreBLEU Score\t {sacre_bleu_accumulated_test / num_test_batches}\tNum correct val mols test: {Num_correct_val_mols_test}",
                file=scores_file)
