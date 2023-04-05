import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config, AutoModel, AutoTokenizer, T5Model
import argparse
import time
from transformers import T5ForConditionalGeneration, T5PreTrainedModel
from torch.cuda.amp import GradScaler, autocast

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logfile', type=str, help='name of the log file')
parser.add_argument('-tr', '--trainfile', type=str, help='name of the training file')
parser.add_argument('-te', '--testfile', type=str, help='name of the test file')
args = parser.parse_args()

class CustomT5Model(nn.Module):
    def __init__(self, esm_model, t5_model):
        super(CustomT5Model, self).__init__()
        self.esm_model = esm_model
        self.t5_model = t5_model
        self.embedding_projection = nn.Linear(1280, 768)  # Create a linear layer to project ESM output to the desired dimension

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None, encoder_outputs=None, past_key_values=None, inputs_embeds=None, decoder_inputs_embeds=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None):
        # Pass the input_ids through the ESM model
        esm_outputs = self.esm_model(input_ids)

        # Extract the embeddings from the ESM outputs
        esm_embeds = esm_outputs["representations"]

        # Project the embeddings to the desired dimension
        projected_embeds = self.embedding_projection(esm_embeds)

        # Pass the projected embeddings and other arguments through the T5 model
        t5_outputs = self.t5_model(
            input_ids=None,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=(projected_embeds, ),
            past_key_values=past_key_values,
            inputs_embeds=projected_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        return t5_outputs



class Dataset(Dataset):
    def __init__(self, filename, tokenizer, esm_tokenizer, esm_model, max_length=4000):
        self.tokenizer = tokenizer
        self.esm_tokenizer = esm_tokenizer
        self.esm_model = esm_model
        self.data = []
        with open(filename, "r") as f:
            for line in f:
                text, label = line.strip().split("\t")
                self.data.append((text, label))

        self.max_length = max_length

    def _process_esm_output(self, text, max_chunk_length=4096):
        # Tokenize the input text and get the number of tokens
        esm_input_encoding = self.esm_tokenizer(text, return_tensors="pt", padding="max_length", truncation=True)
        num_tokens = esm_input_encoding["input_ids"].size(1)

        # Calculate the number of chunks needed
        num_chunks = (num_tokens + max_chunk_length - 1) // max_chunk_length

        # Process each chunk and aggregate the results
        esm_output_reprs = []
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * max_chunk_length
            end_idx = min((chunk_idx + 1) * max_chunk_length, num_tokens)

            # Get the input encoding for the current chunk
            chunk_input_encoding = {k: v[:, start_idx:end_idx] for k, v in esm_input_encoding.items()}

            # Process the chunk using the ESM model
            esm_output = self.esm_model(**chunk_input_encoding)

            # Compute the mean-pooling for the current chunk
            chunk_esm_output_repr = torch.mean(esm_output[0], dim=1)
            esm_output_reprs.append(chunk_esm_output_repr)

        # Concatenate and average the mean-pooled representations of all chunks
        esm_output_repr = torch.mean(torch.cat(esm_output_reprs, dim=0), dim=0, keepdim=True)
        return esm_output_repr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text, label = self.data[idx]
        esm_output_repr = self._process_esm_output(text)
        input_encoding = self.tokenizer(f"protein2Smile: {esm_output_repr}", return_tensors="pt", max_length=200, padding="max_length", truncation=True)

        target_encoding = self.tokenizer(label, return_tensors="pt", max_length=200, padding="max_length",
                                         truncation=True)

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }
start_time = time.time()


# Load the ESM model
esm_model, esm_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

# Load the T5 model
t5_config = T5Config.from_pretrained("t5-small")
t5_model = T5ForConditionalGeneration(t5_config)

# Create the custom model with the ESM and T5 instances
custom_model = CustomT5Model(esm_model, t5_model)

# Tokenizer and config for T5 model
model_name = "google/flan-t5-base"
tokenizer = T5TokenizerFast.from_pretrained(model_name)
config = T5Config.from_pretrained(model_name)
config.n_positions = 26000  # max length needed for protein sequences > 25,000

train_dataset = Dataset(args.trainfile, tokenizer, esm_tokenizer, esm_model)
test_dataset = Dataset(args.testfile, tokenizer, esm_tokenizer, esm_model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
custom_model.to(device)

batch_size = 1
epochs = 7
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

with open(args.logfile, 'w') as f:
    f.write(f"Model name: {model_name}, Train file: {args.trainfile}, Test file: {args.testfile}, Batch size: {batch_size}, Epochs: {epochs}, Device: {device}\n\n")

for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        esm_output_repr = batch["input_ids"].to(device).unsqueeze(1)  # Add sequence length dimension
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_outputs=esm_output_repr,
            decoder_input_ids=labels,
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()



    model.eval()
    correct_predictions = 0
    total_predictions = 0
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            outputs = model.generate(input_ids, attention_mask=attention_mask)
            predicted_labels = [tokenizer.decode(pred, skip_special_tokens=True) for pred in outputs]
            print('outputs: ', outputs)
            print('Predicted labels: ', predicted_labels)
            true_labels = [tokenizer.decode(label, skip_special_tokens=True) for label in labels]
            print('True labels: ', true_labels)

        for pred, true in zip(predicted_labels, true_labels):
            total_predictions += 1
            if pred == true:
                print('\npred: ',pred,'\ntrue: ', true)
                correct_predictions += 1

    with open(args.logfile, 'a') as f:
        print(f"Epoch {epoch + 1}/{epochs}", file=f)
        print(f"Accuracy: {round(correct_predictions / total_predictions, 3)}", file=f)
model.save_pretrained("fine_tuned_flan-t5-base")
end_time = time.time()
with open(args.logfile, 'a') as f:
    print(f"Total time: {round((end_time - start_time)/60, 2)} minutes", file=f)
