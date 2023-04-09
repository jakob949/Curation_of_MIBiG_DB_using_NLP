import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer, AutoTokenizer, AutoModel

T5_model_name = 't5-small'
t5_tokenizer = T5Tokenizer.from_pretrained(T5_model_name)
t5_config = T5Config.from_pretrained(T5_model_name)
t5_model = T5ForConditionalGeneration.from_pretrained(T5_model_name, config=t5_config)

esm_model_name = "facebook/esm2_t6_8M_UR50D"
esm_tokenizer = AutoTokenizer.from_pretrained(esm_model_name)
esm_model = AutoModel.from_pretrained(esm_model_name)

# Resamples a protein sequence
input_text = "ULLI"

esm_input_tokens = esm_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
esm_input_ids = esm_input_tokens['input_ids']
esm_attention_mask = esm_input_tokens['attention_mask']

# saves the hidden states of the ESM2 model
esm_outputs = esm_model(input_ids=esm_input_ids, attention_mask=esm_attention_mask)
esm_hidden_states = esm_outputs[0]

# projects the hidden states to the same dimension as the T5 model
projection = nn.Linear(esm_hidden_states.shape[-1], t5_config.d_model)
projected_hidden_states = projection(esm_hidden_states)

print(f"ESM hidden states shape: {esm_hidden_states.shape}\nhidden states shape: {projected_hidden_states.shape}")

# Prepare the T5 decoder inputs
decoder_start_token = t5_tokenizer.pad_token_id
decoder_input_ids = torch.tensor([[decoder_start_token]], dtype=torch.long)

# Call the T5 model with the projected_hidden_states as encoder_outputs
t5_outputs = t5_model(
    input_ids=None,
    attention_mask=None,
    decoder_input_ids=decoder_input_ids,
    encoder_outputs=(projected_hidden_states, esm_attention_mask),
)

# Get the predicted logits
predicted_logits = t5_outputs.logits
print(f"Predicted logits shape: {predicted_logits.shape}")

# Convert logits to token ids
predicted_token_ids = torch.argmax(predicted_logits, dim=-1)

# Decode the token ids to get the generated text
generated_text = t5_tokenizer.decode(predicted_token_ids[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")

# should the ESM2 Tokens be used for the T5 model?
# here we use the T5 tokenizer to tokenize the original input text
t5_input_tokens = t5_tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
t5_input_ids = t5_input_tokens['input_ids']
print(f"T5 input ids: {t5_input_ids}")

# Maybe restrict the T5 tokenizer vocabulary to the same as ESM2 vocabulary? To avoid biological arbitrary tokens
decoded_text = t5_tokenizer.decode(t5_input_ids[0], skip_special_tokens=True)
print(f"Decoded text: {decoded_text}")



t5_outputs = t5_model(input_ids=None, attention_mask=None, decoder_input_ids=t5_input_ids,
                      decoder_attention_mask=None, encoder_outputs=(projected_hidden_states,))
t5_logits = t5_outputs.logits

print(f"T5 logits shape: {t5_logits.shape}")

t5_output_ids = torch.argmax(t5_logits, dim=-1)
t5_decoded_output = t5_tokenizer.decode(t5_output_ids[0], skip_special_tokens=True)

print(f"Input: {input_text}")
print(f"Output: {t5_decoded_output}")

