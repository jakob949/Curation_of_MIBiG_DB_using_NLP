from torch.utils.data import DataLoader
import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel, AdamW
from torchmetrics.text.rouge import ROUGEScore

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, T5_tokenizer, esm_tokenizer, max_length=1750):
        self.file_path = file_path
        self.data = self.load_data()
        self.T5_tokenizer = T5_tokenizer
        self.esm_tokenizer = esm_tokenizer
        self.max_length = max_length

    def load_data(self):
        data = []
        with open(self.file_path, 'r') as f:
            for line in f:
                text = line.split('\t')[1]
                label = line.split('\t')[2].strip('\n')
                text_list = text.split('_')

                # Check if any element in text_list is longer than 2000 characters
                if all(len(element) <= 1750 for element in text_list):
                    data.append((text_list, label))
        print(len(data))
        return data


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        input_encoding = self.T5_tokenizer(text, return_tensors="pt", max_length=self.max_length, padding="max_length",
                                           truncation=True)
        target_encoding = self.T5_tokenizer(label, return_tensors="pt", max_length=200, padding="max_length",
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


# Set up the training parameters
num_epochs = 12
learning_rate = 5e-5

T5_model_name = 'google/flan-t5-base'
t5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)
t5_config = T5Config.from_pretrained(T5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_model_name, config=t5_config)

esm_model_name = "facebook/esm2_t12_35M_UR50D"
esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
esm_model = AutoModel.from_pretrained(esm_model_name)

projection = nn.Linear(esm_model.config.hidden_size, t5_config.d_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

t5_model.to(device)
esm_model.to(device)
projection.to(device)
print(device)
train_dataset = ProteinDataset("train_dataset_protein_v2_0.txt", t5_tokenizer, esm_tokenizer)
test_dataset = ProteinDataset("test_dataset_protein_v2_0.txt", t5_tokenizer, esm_tokenizer)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

optimizer = AdamW(list(t5_model.parameters()) + list(esm_model.parameters()) + list(projection.parameters()), lr=learning_rate)

rouge = ROUGEScore()

# Training loop
for epoch in range(num_epochs):
    t5_model.train()
    esm_model.train()
    projection.train()

    rouge_train_accumulated = 0.0
    num_train_batches = 0

    for batch in train_loader:
        # Should be fixed - This only works for batch size 1...
        #
        num_train_batches += 1

        text = batch["text_list"]
        labels = batch["labels"].to(device)

        concat_hidden_states = concat_seqs(text)

        projected_hidden_states = projection(concat_hidden_states)
        optimizer.zero_grad()

        decoder_input_ids = torch.cat((torch.full((labels.size(0), 1), 0, dtype=torch.long, device=device), labels[:, :-1]), dim=-1)

        t5_outputs = t5_model(
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=(projected_hidden_states, None),
            labels=labels,
        )

        with torch.no_grad():
            train_predicted_labels = t5_tokenizer.decode(t5_outputs.logits[0].argmax(dim=-1).tolist(), skip_special_tokens=True)
            train_true_labels = [batch["label"][0]]
            train_rouge_score = rouge(train_predicted_labels, train_true_labels)["rouge1_fmeasure"]
            rouge_train_accumulated += train_rouge_score
            #print(f"train_rouge_score: {train_rouge_score}")
            #print(f"train_true_labels: {train_true_labels},train_predicted_labels: {train_predicted_labels} ")

        loss = t5_outputs.loss
        loss.backward()
        optimizer.step()



    # Test loop
    t5_model.eval()
    esm_model.eval()
    projection.eval()

    rouge_test_accumulated = 0.0
    num_test_batches = 0

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
                labels=labels,
            )

            test_predicted_labels = t5_tokenizer.decode(test_outputs.logits[0].argmax(dim=-1).tolist(), skip_special_tokens=True)
            test_true_labels = [batch["label"][0]]
            test_rouge_score = rouge(test_predicted_labels, test_true_labels)["rouge1_fmeasure"]
            rouge_test_accumulated += test_rouge_score
            #print(f"test_true_labels: {test_true_labels}, test_predicted_labels: {test_predicted_labels}, test_rouge_score: {test_rouge_score}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Avg Train ROUGE-1 F1 Score: {rouge_train_accumulated / num_train_batches}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Avg Test ROUGE-1 F1 Score: {rouge_test_accumulated / num_test_batches}")