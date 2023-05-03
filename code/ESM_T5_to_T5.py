from torch.utils.data import DataLoader
import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel, AdamW
from torchmetrics.text.rouge import ROUGEScore
import os

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, file_paths_1, file_paths_2, T5_tokenizer, esm_tokenizer, max_length=250):
        self.file_paths_1 = file_paths_1
        self.file_paths_2 = file_paths_2
        self.data = self.load_data()
        self.T5_tokenizer = T5_tokenizer
        self.esm_tokenizer = esm_tokenizer
        self.max_length = max_length

    def load_data(self):
        data = []
        for file_path_1, file_path_2 in zip(self.file_paths_1, self.file_paths_2):
            with open(file_path_1, 'r') as f1, open(file_path_2, 'r') as f2:
                for line1, line2 in zip(f1, f2):
                    BCG1 = line1.split('\t')[0]
                    BCG2 = line2.split('\t')[0]
                    BCG_all = (BCG1, BCG2)
                    text1 = line1.split('\t')[1]
                    text2 = line2.split('\t')[1]
                    label = line1.split('\t')[2].strip('\n')
                    text_list_1 = text1.split('_')
                    text_list_2 = text2.split('_')

                    # Check if any element in text_list is longer than 2000 characters
                    if all(len(element) <= 250 for element in text_list_1) and all(
                            len(element) <= 750 for element in text_list_2):
                        data.append(((text_list_1, text_list_2), label, BCG_all))
        print(len(data))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        (text1, text2), label, BCG = self.data[idx]
        input_encoding1 = self.T5_tokenizer(text1, return_tensors="pt", max_length=self.max_length,
                                            padding="max_length",
                                            truncation=True)
        input_encoding2 = self.T5_tokenizer(text2, return_tensors="pt", max_length=self.max_length,
                                            padding="max_length",
                                            truncation=True)
        target_encoding = self.T5_tokenizer(label, return_tensors="pt", max_length=200, padding="max_length",
                                            truncation=True)

        return {
            "input_ids_1": input_encoding1["input_ids"].squeeze(),
            "attention_mask_1": input_encoding1["attention_mask"].squeeze(),
            "input_ids_2": input_encoding2["input_ids"].squeeze(),
            "attention_mask_2": input_encoding2["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
            "text_list_1": text1,
            "text_list_2": text2,
            "label": label,
            "BCG": BCG
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


def concat_seqs_ESM(text):
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


def concat_seqs_T5(text, t5_encoder, projection):
    concat_hidden_states = None
    hidden_states_list = []
    # Pad all items to the maximum length and concatenate
    for item in text:
        # Tokenize the input text with the T5 tokenizer
        t5_input_tokens = t5_tokenizer(item, return_tensors='pt', padding=True, truncation=True)
        t5_input_ids = t5_input_tokens['input_ids'].to(device)
        t5_attention_mask = t5_input_tokens['attention_mask'].to(device)

        # Get the T5 encoder hidden states
        t5_outputs = t5_encoder(input_ids=t5_input_ids, attention_mask=t5_attention_mask)
        t5_hidden_states = t5_outputs[0]

        # Project the T5 hidden states to match the ESM hidden states size
        projected_hidden_states = projection(t5_hidden_states.view(-1, t5_hidden_states.size(-1)))
        projected_hidden_states = projected_hidden_states.view(t5_hidden_states.size(0), t5_hidden_states.size(1), -1)
        hidden_states_list.append(projected_hidden_states)

    max_length = max(item_hidden_states.size(1) for item_hidden_states in hidden_states_list)

    padded_hidden_states_list = [
        pad_to_match(item_hidden_states, torch.zeros(1, max_length, item_hidden_states.size(2)), dim=1)[0] for
        item_hidden_states in hidden_states_list]
    concat_hidden_states = torch.cat(padded_hidden_states_list, dim=1)
    return concat_hidden_states




os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Set up the training parameters
num_epochs = 12
learning_rate = 5e-5

T5_model_name = 'google/flan-t5-small'
t5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)
t5_config = T5Config.from_pretrained(T5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_model_name, config=t5_config)

esm_model_name = "facebook/esm2_t6_8M_UR50D"
esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
esm_model = AutoModel.from_pretrained(esm_model_name)

projection = nn.Linear(t5_config.d_model, esm_model.config.hidden_size)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t5_model.to(device)
esm_model.to(device)
projection.to(device)

t5_encoder = t5_model.get_encoder()

print(device)

train_file_paths_1 = ["train_dataset_protein_v2_0.txt"]
train_file_paths_2 = ["train_dataset_protein_text_v2_shorten_0.txt"]
test_file_paths_1 = ["test_dataset_protein_v2_0.txt"]
test_file_paths_2 = ["test_dataset_protein_text_v2_shorten_0.txt"]
train_dataset = ProteinDataset(train_file_paths_1, train_file_paths_2, t5_tokenizer, esm_tokenizer)
test_dataset = ProteinDataset(test_file_paths_1, test_file_paths_2, t5_tokenizer, esm_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
optimizer = AdamW(list(t5_model.parameters()) + list(esm_model.parameters()) + list(projection.parameters()),
                  lr=learning_rate)

rouge = ROUGEScore()

# Training loop
for epoch in range(num_epochs):
    t5_model.train()
    esm_model.train()
    projection.train()

    rouge_train_accumulated = 0.0
    num_train_batches = 0

    for batch in train_loader:
        num_train_batches += 1

        text_ESM = batch["text_list_1"]
        text_T5 = batch["text_list_2"]
        labels = batch["labels"].to(device)

        concat_seq_ESM = concat_seqs_ESM(text_ESM)
        concat_seq_T5 = concat_seqs_T5(text_T5, t5_encoder, projection)

        concat_hidden_states = torch.cat([concat_seq_ESM, concat_seq_T5], dim=1)

        print(batch["BCG"])
        print(concat_seq_ESM.shape, concat_seq_T5.shape)
        optimizer.zero_grad()
        # Pass the concatenated hidden states to the T5 decoder
        outputs = t5_model(input_ids=None, labels=labels, decoder_inputs_embeds=concat_hidden_states, return_dict=True)
        loss = outputs.loss
        loss.backward()

        optimizer.step()

        # Calculate the Rouge-L score
        generated_text = t5_tokenizer.batch_decode(outputs.logits.argmax(dim=-1), skip_special_tokens=True)
        references = [batch["label"]]
        rouge_score = rouge(references, generated_text)
        rouge_train_accumulated += rouge_score["rougeL"].item()
        print("Rouge-L score: ", rouge_score["rougeL"].item())
        break
    break