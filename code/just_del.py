import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Config, AutoModel, AutoTokenizer, T5Model
import argparse
import time
from transformers import T5EncoderModel, T5Decoder, T5PreTrainedModel

from torch.cuda.amp import GradScaler, autocast

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"


parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logfile', type=str, help='name of the log file')
parser.add_argument('-tr', '--trainfile', type=str, help='name of the training file')
parser.add_argument('-te', '--testfile', type=str, help='name of the test file')
args = parser.parse_args()

class CustomT5Model(T5PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.decoder = T5Decoder(config)
        self.lm_head = torch.nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        head_mask=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # ESM2 output is directly used as an input to the decoder
        encoder_outputs = (encoder_outputs,)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            inputs_embeds=decoder_inputs_embeds,
        )

        sequence_output = decoder_outputs[0]
        lm_logits = self.lm_head(sequence_output)

        return (lm_logits,) + decoder_outputs[1:]


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
        input_encoding = self.tokenizer(f"protein2Smile: {esm_output_repr}", return_tensors="pt", max_length=self.max_length,
                                        padding="max_length", truncation=True)
        target_encoding = self.tokenizer(label, return_tensors="pt", max_length=200, padding="max_length",
                                         truncation=True)

        return {
            "input_ids": input_encoding["input_ids"].squeeze(),
            "attention_mask": input_encoding["attention_mask"].squeeze(),
            "labels": target_encoding["input_ids"].squeeze(),
        }
start_time = time.time()

esm_model_name = "facebook/esm2_t6_8M_UR50D" # smallest model
esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
esm_model = AutoModel.from_pretrained(esm_model_name)


model_name = "google/flan-t5-base"
tokenizer = T5TokenizerFast.from_pretrained(model_name)
config = T5Config.from_pretrained(model_name)
config.n_positions = 26000 # max length needed for protein sequences > 25,000
model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)


train_dataset = Dataset(args.trainfile, tokenizer, esm_tokenizer, esm_model)
test_dataset = Dataset(args.testfile, tokenizer, esm_tokenizer, esm_model)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

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
        esm_output_repr = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        lm_logits = model(
            input_ids=None,
            attention_mask=attention_mask,
            encoder_outputs=esm_output_repr,
            decoder_input_ids=labels)[0]

        loss = torch.nn.functional.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1), ignore_index=-100)
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
