import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel, AdamW

# Set up the training parameters
num_epochs = 50
learning_rate = 5e-5

T5_model_name = 't5-small'
t5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)
t5_config = T5Config.from_pretrained(T5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_model_name, config=t5_config)

esm_model_name = "facebook/esm2_t6_8M_UR50D"
esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
esm_model = AutoModel.from_pretrained(esm_model_name)

projection = nn.Linear(esm_model.config.hidden_size, t5_config.d_model)

input_text1 = "ULLI"
input_text2 = "IILUI"

# Prepare the target text (ground truth) for both input sequences
target_text1 = "LLUL"
target_text2 = "LIUUI"

# Tokenize target text
target_tokens1 = t5_tokenizer(target_text1, return_tensors='pt', padding=True, truncation=True)['input_ids']
target_tokens2 = t5_tokenizer(target_text2, return_tensors='pt', padding=True, truncation=True)['input_ids']

# Create the optimizer
optimizer = AdamW(list(t5_model.parameters()) + list(esm_model.parameters()) + list(projection.parameters()), lr=learning_rate)

# Move models to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
t5_model.to(device)
esm_model.to(device)
projection.to(device)

# Define the loss function
loss_fn = nn.CrossEntropyLoss(ignore_index=t5_tokenizer.pad_token_id)

# Function to process input sequences with the ESM2 model and return hidden states
def get_esm_hidden_states(input_text):
    esm_input_tokens = esm_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    esm_input_ids = esm_input_tokens['input_ids']
    esm_attention_mask = esm_input_tokens['attention_mask']
    esm_outputs = esm_model(input_ids=esm_input_ids, attention_mask=esm_attention_mask)
    return esm_outputs[0]

# Training loop
for epoch in range(num_epochs):
    esm_hidden_states1 = get_esm_hidden_states(input_text1)
    esm_hidden_states2 = get_esm_hidden_states(input_text2)
    concat_hidden_states = torch.cat((esm_hidden_states1, esm_hidden_states2), dim=1)
    projected_hidden_states = projection(concat_hidden_states)

    optimizer.zero_grad()

    decoder_input_ids = torch.cat((target_tokens1[:, :-1], target_tokens2[:, :-1]), dim=1).to(device)
    labels = torch.cat((target_tokens1[:, 1:], target_tokens2[:, 1:]), dim=1).to(device)

    t5_outputs = t5_model(
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=decoder_input_ids,
        encoder_outputs=(projected_hidden_states, None),
        labels=labels,
    )
    decode_text = t5_tokenizer.decode(t5_outputs.logits[0].argmax(dim=-1).tolist(), skip_special_tokens=True)
    loss = t5_outputs.loss
    loss.backward()
    optimizer.step()
    print(f"labels: {labels}, decoder_input_ids: {decoder_input_ids}")
    print(f"decode_text: {decode_text}")
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: { loss.item() }")





# Good luck! With combing the code under with the code above, I think you'll be able to get it working.

# import torch
# from torch import nn
# from torch.optim import AdamW
# from torch.utils.data import DataLoader
# from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel
#
# class ProteinDataset(torch.utils.data.Dataset):
#     def __init__(self, file_path, T5_tokenizer, esm_tokenizer, max_length=512):
#         self.file_path = file_path
#         self.data = self.load_data()
#         self.T5_tokenizer = T5_tokenizer
#         self.esm_tokenizer = esm_tokenizer
#         self.max_length = max_length
#
#     def load_data(self):
#         data = []
#         with open(self.file_path, 'r') as f:
#             for line in f:
#                 text = line.split('\t')[1]
#                 label = line.split('\t')[2].strip('\n')
#                 text_list = text.split('_')
#                 data.append((text_list, label))
#         return data
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         text, label = self.data[idx]
#         input_encoding = self.T5_tokenizer(text, return_tensors="pt", max_length=self.max_length, padding="max_length",
#                                            truncation=True)
#         target_encoding = self.T5_tokenizer(label, return_tensors="pt", max_length=200, padding="max_length",
#                                             truncation=True)
#
#         return {
#             "input_ids": input_encoding["input_ids"].squeeze(),
#             "attention_mask": input_encoding["attention_mask"].squeeze(),
#             "labels": target_encoding["input_ids"].squeeze(),
#             "text_list": text,
#             "label": label
#         }
#
#
# def get_esm_hidden_states(input_texts, esm_tokenizer, esm_model):
#     hidden_states = []
#     for input_text in input_texts:
#         esm_input_tokens = esm_tokenizer(input_text, return_tensors='pt', padding=True, truncation=False)
#         esm_input_ids = esm_input_tokens['input_ids']
#         esm_attention_mask = esm_input_tokens['attention_mask']
#         esm_outputs = esm_model(input_ids=esm_input_ids, attention_mask=esm_attention_mask)
#         hidden_states.append(esm_outputs[0])
#     hidden_states = torch.cat(hidden_states, dim=1)
#     return hidden_states
#
#
# T5_model_name = 't5-small'
# t5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)
# t5_config = T5Config.from_pretrained(T5_model_name)
# t5_model = T5ForConditionalGeneration.from_pretrained(T5_model_name, config=t5_config)
#
# esm_model_name = "facebook/esm2_t6_8M_UR50D"
# esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
# esm_model = AutoModel.from_pretrained(esm_model_name)
#
# dataset = ProteinDataset("dataset_protein.txt", t5_tokenizer, esm_tokenizer)
# batch_size = 8  # Changed from 1 to 8
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# epochs = 1
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# lr = 1e-4
# t5_model.to(device)
# optimizer = AdamW(t5_model.parameters(), lr=lr)
#
# projection = nn.Linear(esm_model.config.hidden_size, t5_config.d_model).to(device)  # Moved outside the loop
#
# for epoch in range(epochs):
#     t5_model.train()  # Set the model to train mode
#     for batch in dataloader:
#         optimizer.zero_grad()
#         input_text = batch['text_list']
#         label = batch['label']
#         label_encoded = batch['labels'].to(device)
#         EMS_embedding = get_esm_hidden_states(input_text, esm_tokenizer, esm_model)
#
#         projected_hidden_states = projection(EMS_embedding)
#         print(f"projected_hidden_states.shape: {projected_hidden_states.shape}")
#
#         decoder_start_token = t5_tokenizer.pad_token_id
#         decoder_input_ids = torch.tensor([[decoder_start_token]] * batch_size, dtype=torch.long).to(device)
#
#         t5_outputs = t5_model(
#             input_ids=None,
#             attention_mask=None,
#             decoder_input_ids=decoder_input_ids,
#             encoder_outputs=projected_hidden_states,
#             #labels=label_encoded,
#         )
#
#         # loss = t5_outputs.loss
#         # loss.backward()
#         # optimizer.step()
#
# print("Training complete.")
