import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoModel, AutoTokenizer)
from torch.optim import AdamW

class ClassificationModel(nn.Module):
    def __init__(self, base_model, num_labels):
        super(ClassificationModel, self).__init__()
        self.base_model = base_model
        self.num_labels = num_labels
        self.classifier = nn.Linear(base_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, 0, :])
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits

class TextClassificationDataset(Dataset):
    def __init__(self, file_paths):
        self.data = []
        for file_path in file_paths:
            with open(file_path, "r") as f:
                for line in f:
                    text, label = line.strip().split("\t")
                    self.data.append((text, int(label)))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the pre-trained model
base_model = AutoModel.from_pretrained("microsoft/biogpt").to(device)
tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
num_labels = 2
model = ClassificationModel(base_model, num_labels).to(device)

# Define the dataloader
file_paths = ["spacy.txt"]
dataset = TextClassificationDataset(file_paths)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

## Fine-tune the model ##
save_model = False
model.train()
num_of_epochs = 1
optimizer = AdamW(model.parameters(), lr=1e-5)
for epoch in range(num_of_epochs):
    print(f"Epoch {epoch + 1}/{num_of_epochs}")
    for i, batch in enumerate(dataloader):
        print(f"Batch {i + 1}/{len(dataloader)}")
        texts, labels = batch
        inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        labels = labels.to(device)
        loss, logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
