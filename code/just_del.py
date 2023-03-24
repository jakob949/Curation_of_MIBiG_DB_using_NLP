import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW

parser = argparse.ArgumentParser(description="Text Classification")
parser.add_argument("--trainfile", type=str, required=True, help="Path to the training file.")
parser.add_argument("--testfile", type=str, required=True, help="Path to the test file.")
parser.add_argument("--logfile", type=str, default="log.txt", help="Path to the log file.")
args = parser.parse_args()

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

# Load the pre-trained model
base_model = AutoModel.from_pretrained("microsoft/biogpt")
tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
num_labels = 2
model = ClassificationModel(base_model, num_labels)

# Define the dataloader
train_file_path = [args.trainfile]
dataset = TextClassificationDataset(train_file_path)
dataloader = DataLoader(dataset, batch_size=3, shuffle=True)

# Fine-tune the model
model.train()
num_of_epochs = 1
optimizer = AdamW(model.parameters(), lr=1e-5)

with open(args.logfile, "w") as f:
    for epoch in range(num_of_epochs):
        print(f"Epoch {epoch + 1}/{num_of_epochs}", file=f)
        for i, batch in enumerate(dataloader):
            print(f"Batch {i + 1}/{len(dataloader)}", file=f)
            texts, labels = batch
            inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            loss, logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], labels=labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    # Define the test dataloader
    test_file_path = [args.testfile]
    test_dataset = TextClassificationDataset(test_file_path)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # Evaluate the model on the test dataset
    model.eval()
    total_correct_preds = 0
    total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            print(f"Batch {i + 1}/{len(test_dataloader)}", file=f)
            abstract_text, labels = batch
            inputs = tokenizer(abstract_text, padding=True, truncation=True, return_tensors="pt")
            outputs = model(inputs["input_ids"], inputs["attention_mask"])
            predictions = torch.argmax(outputs, dim=1)
            print(f"Prediction class: {predictions.item()}\tCorrect label: {labels.item()}\tprobs {torch.nn.functional.softmax(outputs, dim=1).tolist()[0]}", file=f)
            total_correct_preds += torch.sum(predictions == labels).item()
            total_samples += 1

    accuracy = total_correct_preds / total_samples
    print(f"Accuracy: {accuracy * 100:.2f}%", file=f)

