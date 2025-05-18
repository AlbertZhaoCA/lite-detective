from matplotlib import ticker, transforms
from matplotlib.colors import LinearSegmentedColormap
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW, lr_scheduler
from .text_cnn import DynamicTextCNN
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
from PIL import Image
from torchvision.transforms import ToTensor

class ToxicTextClassifier(nn.Module):
    def __init__(self,
                 bert_name='hfl/chinese-roberta-wwm-ext',
                 num_filters=1536,
                 filter_sizes=(1,2,3,4),
                 K=4,
                 fc_dim=128,
                 num_classes=2,
                 dropout=0.1,
                 name='lited_best'):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained(bert_name,from_tf=True)
        self.bert = BertModel.from_pretrained(bert_name)
        self.name = name
        self.unfrozen_layers = 0

        hidden_size = self.bert.config.hidden_size * 2
        os.makedirs(f'data/{name}', exist_ok=True)

        self.text_cnn = DynamicTextCNN(hidden_size, num_filters, filter_sizes, K, dropout)
        input_dim = len(filter_sizes) * num_filters
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, fc_dim),
            nn.ReLU(),
            nn.LayerNorm(fc_dim),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, fc_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim // 2, num_classes)
        )

        self.criterion = nn.CrossEntropyLoss()
        self._rebuild_optimizer()
    
        self.warmup_scheduler = None

    def _get_warmup_scheduler(self, warmup_steps=1000):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        return lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _rebuild_optimizer(self):
        param_groups = [
            {'params': self.text_cnn.parameters(),   'lr': 1e-4},
            {'params': self.classifier.parameters(), 'lr': 1e-4},
        ]
        
        if self.unfrozen_layers > 0:
            layers = self.bert.encoder.layer[-self.unfrozen_layers:]
            bert_params = []
            for layer in layers:
                for p in layer.parameters():
                    p.requires_grad = True
                    bert_params.append(p)
            param_groups.append({'params': bert_params, 'lr': 2e-5})

        self.optimizer = AdamW(param_groups, weight_decay=0.01)
        
        self.scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
        )


    def forward(self, input_ids, attention_mask, token_type_ids=None):
        bert_out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
        )   
        hidden = torch.cat(bert_out.hidden_states[-2:], dim=-1)
        feat = self.text_cnn(hidden)
        return self.classifier(feat)

    def validate(self, val_loader, device):
        self.eval()
        val_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validating')
            for batch in pbar:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                types = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)

                logits = self(ids, mask, types)
                loss = self.criterion(logits, labels)
                val_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        epoch_acc = correct / total if total > 0 else 0
        metrics = {
            'loss': val_loss / len(val_loader),
            'acc': epoch_acc,
            'report': classification_report(all_labels, all_preds, target_names=['non-toxic','toxic']),
            'confusion_matrix': confusion_matrix(all_labels, all_preds)
        }
        torch.cuda.empty_cache()
        return metrics
    
    def train_model(self, train_loader, val_loader,
                    num_epochs=3, device='cpu',
                    save_path=None,
                    logdir=None,
                    validate_every=100,
                    warmup_steps=1000,
                    early_stop_patience=3):
        self.to(device)

        for param in self.bert.parameters():
            param.requires_grad = False
        
        best_val_loss = float('inf')
        global_step = 0
        epochs_no_improve = 0
        best_model_state = None

        if save_path is None:
            save_path = f'output/{self.name}.pth'

        if logdir is None:
            logdir = f'runs/{self.name}'
            writer = SummaryWriter(logdir)

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")

            total_loss = 0
            correct = 0
            total = 0

            self.warmup_scheduler = self._get_warmup_scheduler(warmup_steps)

            if epoch == 2:
                print("Unfreezing 4 layers of BERT")
                self.unfrozen_layers = 2
                self._rebuild_optimizer()

            pbar = tqdm(train_loader, desc='Training')
            for batch in pbar:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                types = batch['token_type_ids'].to(device)
                labels = batch['label'].to(device)

                logits = self(ids, mask, types)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                self.optimizer.step()

                if global_step < warmup_steps:
                    self.warmup_scheduler.step()
                
                for i, group in enumerate(self.optimizer.param_groups):
                    writer.add_scalar(f'LR/group_{i}', group['lr'], global_step)

                for name, param in self.named_parameters():
                     if "convs" in name:
                        grad_norm = param.grad.norm().item()
                        writer.add_scalar(f'Gradients/{name}', grad_norm, global_step)
             
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                acc = correct / total

                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Acc/train', acc, global_step)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.4f}'})
                global_step += 1

                if global_step % validate_every == 0:
                    torch.cuda.empty_cache()
                    self.eval()
                    with torch.no_grad():
                        metrics = self.validate(val_loader, device)
                    val_loss, val_acc = metrics['loss'], metrics['acc']

                    self.scheduler.step(val_loss)
                    
                    print(f"\n[Step {global_step}] Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
                    print(metrics['report'])
                    report_text = metrics['report']
                    conf_mat = metrics['confusion_matrix']
                    print(report_text)
                    writer.add_text('Classification Report', report_text, global_step)
                    writer.add_scalar('Loss/vali', val_loss, global_step)
                    writer.add_scalar('Acc/vali', val_acc, global_step)
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = self.state_dict()
                        epochs_no_improve = 0
                        torch.save(best_model_state, save_path)
                        print(f"Saved best model (step {global_step}) with loss {best_val_loss:.4f}")
                    else:
                        epochs_no_improve += 1
                        print(f"No improvement for {epochs_no_improve} checks")
                        
                        if epochs_no_improve >= early_stop_patience:
                            print(f"Early stopping triggered at step {global_step}!")
                            self.load_state_dict(best_model_state)
                            writer.close()
                            return

                    flame_colors = ['#ffffcc', '#ffeda0', '#feb24c', '#fd8d3c', '#f03b20', '#bd0026']
                    flame_cmap = LinearSegmentedColormap.from_list("flame", flame_colors, N=256)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.set_theme(font_scale=1.4)
                    sns.heatmap(
                        conf_mat,
                        annot=True,
                        fmt='d',
                        cmap=flame_cmap,
                        linewidths=0.5,
                        linecolor='gray',
                        square=True,
                        cbar=True,
                        xticklabels=['non-toxic', 'toxic'],
                        yticklabels=['non-toxic', 'toxic'],
                        annot_kws={"size": 16, "weight": "bold"}
                    )

                    ax.set_xlabel('Predicted', fontsize=14, labelpad=10)
                    ax.set_ylabel('True', fontsize=14, labelpad=10)
                    ax.set_title('Confusion Matrix', fontsize=16, pad=12)

                    ax.xaxis.set_tick_params(labelsize=12)
                    ax.yaxis.set_tick_params(labelsize=12)
                    ax.xaxis.set_major_locator(ticker.FixedLocator([0.5, 1.5]))
                    ax.yaxis.set_major_locator(ticker.FixedLocator([0.5, 1.5]))

                    buf = io.BytesIO()
                    plt.tight_layout()
                    plt.savefig(buf, format='png', dpi=150)
                    plt.savefig(f'data/{self.name}/conf_matrix_step{global_step}.pdf', format='pdf', bbox_inches='tight')
                    buf.seek(0)
                    image = Image.open(buf)
                    image_tensor = ToTensor()(image)
                    writer.add_image('Confusion Matrix', image_tensor, global_step)

                    buf.close()
                    plt.close(fig)

                    self.train()

        writer.close()
        
    def predict(self, texts, device='cpu'):
        """Used for inference. Predicts the class of the input text.
        Args:
            texts (str or list of str): The input text(s) to classify, pass str.
                - If a list is passed, the model will classify each text in the list as batch.
                - If a single string is passed, the model will classify the text as a single instance.
                - If a list of list is passed, the model will treate the first element as detected text and the second element as the context text.
            device (str): The device to run the model on ('cpu', 'cuda', or 'mps'). If None, it will use the available device.
            max_length (int): The maximum length of the input text.
        Returns:
            list: A list of dictionaries containing the prediction and probabilities for each input text.
                Each dictionary contains:
                    - 'text': The input text.
                    - 'prediction': The predicted class (0 or 1).
                    - 'probabilities': The probabilities for each class.
    """
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'

        self.eval()
        self.to(device)

        if isinstance(texts, str):
            texts = [texts]
            encoded_inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        elif isinstance(texts, list) and all(isinstance(item, list) for item in texts):
            encoded_inputs = self.tokenizer(
            [item[0] for item in texts],
            [item[1] for item in texts],
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        elif isinstance(texts, list) and all(isinstance(item, str) for item in texts):
            encoded_inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(device)
        else:
            raise ValueError("Invalid input type. Expected str or list of str.")

        input_ids = encoded_inputs['input_ids']
        attention_mask = encoded_inputs['attention_mask']
        token_type_ids = encoded_inputs.get('token_type_ids', None)

        with torch.no_grad():
            logits = self(input_ids, attention_mask, token_type_ids)
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

        results = []
        for i, text in enumerate(texts):
            results.append({
                'text': text,
                'prediction': preds[i].item(),
                'probabilities': probs[i].cpu().tolist()
            })
        return results





