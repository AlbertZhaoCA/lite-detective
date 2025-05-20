import torch
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from models.classifier import ToxicTextClassifier
from libs.data_processing.dataset import ToxicDataset,COLDDataset
import os
import transformers
import csv
transformers.logging.set_verbosity_error()

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
        }
    
with open("test.tsv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    texts = []
    labels = []
    for row in reader:
        texts.append(row["TEXT"])
        labels.append(int(row["label"]))

def main():
    model = ToxicTextClassifier()
    
    dataset1 = ToxicDataset('/home/cw/llm-security/lite-detective/v2/data/final.jsonl', model.tokenizer)
    batch_size = 40
    test_dataset = ToxicTestDataset(texts, labels, model.tokenizer)
    
    full_dataset = dataset1
    print(f"Total training samples: {len(full_dataset)}")

    train_size = int(0.9 * len(full_dataset))
    print(len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2000, shuffle=True)

    os.makedirs('output', exist_ok=True)

    model.train_model(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=10,
        device='cuda:2' if torch.cuda.is_available() else 'cpu',
        validate_every=100,
        early_stop_patience=5,
    )

if __name__ == '__main__':
    main()