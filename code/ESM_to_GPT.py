from torch.utils.data import DataLoader
import torch
from torch import nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModel, AdamW
from torchmetrics.text.rouge import ROUGEScore
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, gpt2_tokenizer, esm_tokenizer, max_length=850):
        self.file_path = file_path
        self.data = self.load_data()
        self.gpt2_tokenizer = gpt2_tokenizer
        self.esm_tokenizer = esm_tokenizer
        self.max_length = max_length

    def load_data(self):
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                text = line.split('\t')[1]
                label = line.split('\t')[2].strip('\n')
                text_list = text.split('_')

                if all(len(element) <= 850 for element in text_list):
                    data.append((text_list, label))
        print(len(data))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        input_encoding = self.gpt2_tokenizer(text, return_tensors="pt", max_length=self.max_length,
                                             padding="max_length",
                                             truncation=True)
        target_encoding = self.gpt2_tokenizer(label, return_tensors="pt", max_length=200, padding="max_length",
                                              truncation=True)

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
            "text_list": text,
            "label": label
        }


def get_esm_hidden_states(input_text):
    esm_input_tokens = esm_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    esm_input_ids = esm_input_tokens['input_ids'].to(device)
    esm_attention_mask = esm_input_tokens['attention_mask'].to(device)
    esm_outputs = esm_model(input_ids=esm_input_ids, attention_mask=esm_attention_mask)
    return esm_outputs[0]


def pad_to_match(tensor1, tensor2, dim):
    size1 = tensor1.size(dim)
    size2 = tensor2.size(dim)
    diff = size1 - size2

    if diff > 0:
        padding = (0, 0, 0, diff)
        tensor2 = nn.functional.pad(tensor2, padding)
    elif diff < 0:
        padding = (0, 0, 0, -diff)
        tensor1 = nn.functional.pad(tensor1, padding)

    return tensor1, tensor2


def concat_seqs(text):
    concat_hidden_states = None
    hidden_states_list = []

    for item in text:
        hidden_states_list.append(get_esm_hidden_states(item))

    max_length = max(item_hidden_states.size(1) for item_hidden_states in hidden_states_list)

    padded_hidden_states_list = [
        pad_to_match(item_hidden_states, torch.zeros(1, max_length, item_hidden_states.size(2)), dim=1)[0] for
        item_hidden_states in hidden_states_list]
    concat_hidden_states = torch.cat(padded_hidden_states_list, dim=1)
    return concat_hidden_states


def resize_position_embeddings(model, new_max_position_embeddings):
    old_position_embeddings = model.transformer.wpe.weight.data
    new_position_embeddings = torch.zeros(new_max_position_embeddings, model.config.n_embd)
    new_position_embeddings[: model.config.max_position_embeddings] = old_position_embeddings
    model.transformer.wpe.weight.data = new_position_embeddings
    model.config.max_position_embeddings = new_max_position_embeddings


GPT2_model_name = 'gpt2'
gpt2_tokenizer = GPT2Tokenizer.from_pretrained(GPT2_model_name)
gpt2_config = GPT2Config.from_pretrained(GPT2_model_name)
gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_model_name, config=gpt2_config)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

esm_model_name = "facebook/esm2_t6_8M_UR50D"
esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
esm_model = AutoModel.from_pretrained(esm_model_name)

projection = nn.Linear(esm_model.config.hidden_size, gpt2_config.n_embd)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt2_model.to(device)
esm_model.to(device)
projection.to(device)
print(device)
learning_rate = 5e-5
new_max_position_embeddings = 1024
resize_position_embeddings(gpt2_model, new_max_position_embeddings)

train_dataset = ProteinDataset("train_dataset_protein_v2_0.txt",
                               gpt2_tokenizer, esm_tokenizer)
test_dataset = ProteinDataset("test_dataset_protein_v2_0.txt", gpt2_tokenizer,
                              esm_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

optimizer = AdamW(list(gpt2_model.parameters()) + list(esm_model.parameters()) + list(projection.parameters()),
                  lr=learning_rate)

rouge = ROUGEScore()

num_epochs = 12


for epoch in range(num_epochs):
    gpt2_model.train()
    esm_model.train()
    projection.train()

    rouge_train_accumulated = 0.0
    num_train_batches = 0

    for batch in train_loader:
        num_train_batches += 1

        text = batch["text_list"]
        labels = batch["labels"].to(device)

        concat_hidden_states = concat_seqs(text)
        projected_hidden_states = projection(concat_hidden_states)

        optimizer.zero_grad()

        input_embeddings = projected_hidden_states
        bos_embedding = gpt2_model.transformer.wte(torch.tensor([[gpt2_tokenizer.bos_token_id]], device=device))
        input_embeddings = torch.cat([bos_embedding, input_embeddings], dim=1)

        # Generate position_ids explicitly
        position_ids = torch.arange(0, input_embeddings.size(1), dtype=torch.long,
                                    device=input_embeddings.device).unsqueeze(0)

        gpt2_outputs = gpt2_model(
            inputs_embeds=input_embeddings,
            past_key_values=None,
            labels=labels,
            position_ids=position_ids,  # Pass the position_ids explicitly
        )

        with torch.no_grad():
            train_predicted_labels = gpt2_tokenizer.decode(gpt2_outputs.logits[0].argmax(dim=-1).tolist(),
                                                           skip_special_tokens=True)
            train_true_labels = [batch["label"][0]]
            train_rouge_score = rouge(train_predicted_labels, train_true_labels)["rouge1_fmeasure"]
            rouge_train_accumulated += train_rouge_score

        loss = gpt2_outputs.loss
        loss.backward()
        optimizer.step()

    # Calculate and print the average training Rouge score for the epoch
    rouge_train_average = rouge_train_accumulated / num_train_batches
    print(f"Epoch {epoch + 1}/{num_epochs}, Training Rouge: {rouge_train_average:.4f}")

