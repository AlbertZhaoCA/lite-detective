import json
import csv
from collections import defaultdict
import torch
from torch.utils.data import Dataset

class ToxicDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.samples = []
        
        label_counts = defaultdict(int)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                text = obj['text']
                label_with_context = obj['label_with_context']
                context = obj.get('context', [])
                context_str = ' '.join(context) if context else ''

                self.samples.append((text, context_str.strip(), label_with_context))
                label_counts[label_with_context] += 1

                label_without_context = obj['label_without_context']
                
                self.samples.append((text, '', label_without_context))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, context, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            context,
            padding='max_length',
            truncation='longest_first',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc.input_ids.squeeze(0),
            'attention_mask': enc.attention_mask.squeeze(0),
            'token_type_ids': enc.token_type_ids.squeeze(0),
            'label': torch.tensor(label),
        }

    def get_label(self, idx):
        """Returns the label of the sample at the given index."""
        return self.samples[idx][2]
    
    def get_label_distribution(self):
        """@deprecated
        Returns the distribution of labels in the dataset."""
        label_counts = defaultdict(int)
        for _, _, label in self.samples:
            label_counts[label] += 1
        return dict(label_counts)
    
class COLDDataset(Dataset):
    def __init__(self, filepath, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.samples = []
        self.max_length = max_length

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
           
            for row in reader:
                text = row["TEXT"]
                label = int(row["label"])
                self.samples.append((text, '', label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, context, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            context,
            padding='max_length',
            truncation='longest_first',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc.input_ids.squeeze(0),
            'attention_mask': enc.attention_mask.squeeze(0),
            'token_type_ids': enc.token_type_ids.squeeze(0),
            'label': torch.tensor(label),
        }