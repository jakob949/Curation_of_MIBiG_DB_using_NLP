import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW

class BinaryClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer
        self.lines = open(file_path, 'r', encoding='utf-8').readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        text, label = line.split('\t')
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=512)
        return {**encoding, 'labels': torch.tensor(int(label))}

def train(model, dataloader, optimizer, device):
    model.train()
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

def main():
    # Config
    train_file = 'spacy_train.txt'
    test_file = 'spacy_test.txt'
    epochs = 3
    batch_size = 8
    learning_rate = 5e-5

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token
    model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    # Load data
    train_dataset = BinaryClassificationDataset(train_file, tokenizer)
    test_dataset = BinaryClassificationDataset(test_file, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Train
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train(model, train_dataloader, optimizer, device)

    # Save model
    model.save_pretrained("fine_tuned_gpt2")

if __name__ == '__main__':
    main()
