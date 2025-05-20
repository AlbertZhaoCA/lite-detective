import torch
import csv
from transformers import BertTokenizer
from models.classifier import ToxicTextClassifier
from sklearn.metrics import classification_report, accuracy_score
from torch.utils.data import DataLoader, Dataset

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class ToxicTestDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'label': torch.tensor(label),
            'text': text
        }

model = ToxicTextClassifier()
model.load_state_dict(torch.load("/home/cw/llm-security/lite-detective/v2/output/lited_best.pth", map_location=device))
model.to(device)
model.eval()
result = []
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

with open("test.tsv", "r", encoding="utf-8") as f:
    correct = 0
    reader = csv.DictReader(f)
    texts = []
    labels = []
    for row in reader:
        print(row)
        texts.append(row["TEXT"])
        labels.append(int(row["label"]))
        if int(row["label"]) == 1:
            correct += 1
    print(f"Number of correct labels: {correct}")

print(f"Number of test samples: {len(texts)}")
print(f"Number of test labels: {len(labels)}")

batch_size = 2400
test_dataset = ToxicTestDataset(texts, labels, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

all_predictions = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)
        texts = batch['text']

        logits = model(input_ids, attention_mask, token_type_ids)
        predictions = torch.argmax(logits, dim=1)
        # for label, prediction,text in zip(labels, predictions,texts):
        #     if label != prediction:
        #         print(f"Mismatch - Label: {label.item()}, Prediction: {prediction.item()}, Text: {text}")
        #         result.append((label.item(), prediction.item()))

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, target_names=['non-toxic', 'toxic'])

        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)