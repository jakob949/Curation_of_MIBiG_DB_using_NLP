import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer, T5Config, AdamW, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-base")

# Define the dataset class
class ClassificationDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_ids, attention_mask, label = self.data[idx]
        return input_ids, attention_mask, label

def read_file(filepath, tokenizer):
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()
    data = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        text, label = line.split("\t")
        encoding = tokenizer(text, return_tensors="pt", truncation=True, max_length='max_length')
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        label = int(label)
        data.append((input_ids, attention_mask, label))
    return data

def collate_fn(batch):
    input_ids, attention_masks, labels = zip(*batch)
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return input_ids, attention_masks, labels

# Define the evaluation function
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    total_correct = 0
    with torch.no_grad():
        for input_ids, attention_mask, labels in dataloader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            correct = (preds == labels).sum().item()
            total_correct += correct

    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)

# Read the training data
train_data = read_file("spacy_train.txt", tokenizer)

# Read the test data
test_data = read_file("spacy_test.txt", tokenizer)

# Create the datasets and dataloaders
train_dataset = ClassificationDataset(train_data)
test_dataset = ClassificationDataset(test_data)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=2, collate_fn=collate_fn)

# Initialize the model and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
optimizer = AdamW(model.parameters(), lr=5e-5)

# Fine-tune the model
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for input_ids, attention_mask, labels in train_dataloader:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels= labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    # Evaluate the model on the test set
    val_loss, val_acc = evaluate(model, test_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss.item()}, Test Loss: {val_loss}, Test Acc: {val_acc}")

# Save the fine-tuned model
model.save_pretrained("t5_finetuned_classification")

# Load the saved model for further use
loaded_model = T5ForConditionalGeneration.from_pretrained("t5_finetuned_classification").to(device)

# Test the loaded model with a sample input
sample_input = "This is a sample input text for classification."
encoding = tokenizer(sample_input, return_tensors="pt", padding=True, truncation=True, max_length=2500)
input_ids = encoding["input_ids"].to(device)
attention_mask = encoding["attention_mask"].to(device)

with torch.no_grad():
    logits = loaded_model(input_ids=input_ids, attention_mask=attention_mask).logits
    prediction = torch.argmax(logits, dim=-1).item()

print(f"Sample input classification: {prediction}")

