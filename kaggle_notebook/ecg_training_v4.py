"""
ECG Arrhythmia Classification - v4 Optimized Pipeline
Kaggle Notebook for GPU Training with AMP, LR Scheduling, Augmentation, Focal Loss
"""

# %% [markdown]
# # ECG Arrhythmia Classification - v4 Optimized
# ## Deep Learning Models: ResNet, LSTM, Multi-Module, PhysioWave
# 
# ### v4 Optimizations:
# - Automatic Mixed Precision (AMP) for faster GPU training
# - Learning Rate Scheduling (ReduceLROnPlateau)
# - Signal Augmentation (noise injection, temporal shift)
# - Focal Loss for class imbalance handling

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 2. Configuration

# %%
class Config:
    TRAIN_CSV = "/kaggle/input/heartbeat/mitbih_train.csv"
    TEST_CSV = "/kaggle/input/heartbeat/mitbih_test.csv"
    BATCH_SIZE = 128
    NUM_EPOCHS = 30
    LEARNING_RATE = 1e-3
    NUM_CLASSES = 5
    LOG_DIR = "runs"
    USE_AMP = True
    USE_FOCAL_LOSS = True
    USE_AUGMENTATION = True
    
config = Config()

# %% [markdown]
# ## 3. Data Loading with Augmentation

# %%
class ECGDataset(Dataset):
    def __init__(self, data_path, transform=None, is_training=False):
        df = pd.read_csv(data_path, header=None)
        self.labels = df.iloc[:, -1].values.astype(np.int64)
        self.signals = df.iloc[:, :-1].values.astype(np.float32)
        self.signals = np.expand_dims(self.signals, axis=1)  # (N, 1, 187)
        self.transform = transform
        self.is_training = is_training
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        
        if self.is_training and config.USE_AUGMENTATION:
            signal = self._augment(signal)
        
        if self.transform:
            signal = self.transform(signal)
            
        return torch.tensor(signal), torch.tensor(label)
    
    def _augment(self, signal):
        """Apply 1D signal augmentation"""
        signal = signal.copy()
        # Random noise injection
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.01, signal.shape)
            signal = signal + noise.astype(np.float32)
        # Slight temporal shift (wrap around)
        if np.random.rand() > 0.5:
            shift = np.random.randint(-5, 5)
            signal = np.roll(signal, shift, axis=1)
        return signal

def get_dataloaders(batch_size=128):
    train_dataset = ECGDataset(config.TRAIN_CSV, is_training=True)
    test_dataset = ECGDataset(config.TEST_CSV, is_training=False)
    
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader, train_dataset.signals.shape[2]

# %% [markdown]
# ## 4. Model Architectures

# %%
class ResBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ECG_ResNet(nn.Module):
    def __init__(self, num_classes=5, block_config=[3, 4, 6, 3]):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, block_config[0], stride=1)
        self.layer2 = self._make_layer(128, block_config[1], stride=2)
        self.layer3 = self._make_layer(256, block_config[2], stride=2)
        self.layer4 = self._make_layer(512, block_config[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(ResBlock1D(self.in_channels, out_channels, s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).squeeze(-1)
        return self.fc(x)

class ECG_LSTM(nn.Module):
    def __init__(self, num_classes=5, input_size=1, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])

class ECG_MultiModule(nn.Module):
    def __init__(self, num_classes=5, seq_len=187):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True, bidirectional=True)
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = nn.AdaptiveAvgPool1d(1)(x).squeeze(-1)
        return self.fc(x)

class ECG_PhysioWave(nn.Module):
    def __init__(self, num_classes=5, seq_len=187):
        super().__init__()
        self.scale1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.scale2 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.scale3 = nn.Conv1d(1, 32, kernel_size=11, padding=5)
        self.fusion = nn.Conv1d(96, 64, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        s1 = torch.relu(self.scale1(x))
        s2 = torch.relu(self.scale2(x))
        s3 = torch.relu(self.scale3(x))
        x = torch.cat([s1, s2, s3], dim=1)
        x = self.fusion(x).permute(0, 2, 1)
        x = self.transformer(x)
        x = x.permute(0, 2, 1)
        x = nn.AdaptiveAvgPool1d(1)(x).squeeze(-1)
        return self.fc(x)

# %% [markdown]
# ## 5. Focal Loss Implementation

# %%
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, inputs, targets):
        ce_loss = self.cross_entropy(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# %% [markdown]
# ## 6. Optimized Trainer with AMP and LR Scheduling

# %%
class OptimizedTrainer:
    def __init__(self, model, model_name, train_loader, val_loader, test_loader):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.criterion = FocalLoss() if config.USE_FOCAL_LOSS else nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', patience=5, factor=0.5)
        
        self.scaler = torch.cuda.amp.GradScaler() if config.USE_AMP and torch.cuda.is_available() else None
        
        run_name = f"{model_name}_v4_{int(time.time())}"
        self.writer = SummaryWriter(log_dir=os.path.join(config.LOG_DIR, run_name))
        self.best_val_acc = 0.0
        self.save_path = f"best_{model_name}_v4.pth"
        
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for signals, labels in pbar:
            signals, labels = signals.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(signals)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item() * signals.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': loss.item()})
        
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        self.writer.add_scalar('Loss/Train', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
        self.writer.add_scalar('F1/Train', f1, epoch)
        
        return epoch_loss, epoch_acc, f1
    
    def validate_epoch(self, epoch):
        self.model.eval()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for signals, labels in pbar:
                signals, labels = signals.to(self.device), labels.to(self.device)
                outputs = self.model(signals)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * signals.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        self.writer.add_scalar('Loss/Validation', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/Validation', epoch_acc, epoch)
        self.writer.add_scalar('F1/Validation', f1, epoch)
        
        self.scheduler.step(epoch_loss)
        
        if epoch_acc > self.best_val_acc:
            self.best_val_acc = epoch_acc
            torch.save(self.model.state_dict(), self.save_path)
            print(f"  -> Saved best model with Val Acc: {epoch_acc:.4f}")
        
        return epoch_loss, epoch_acc, f1
    
    def test(self):
        self.model.load_state_dict(torch.load(self.save_path))
        self.model.eval()
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for signals, labels in tqdm(self.test_loader, desc="Testing"):
                signals, labels = signals.to(self.device), labels.to(self.device)
                outputs = self.model(signals)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)
        
        print(f"\n{'='*50}")
        print(f"Test Results for {self.model_name}")
        print(f"Accuracy: {acc:.4f}")
        print(f"F1-Score (weighted): {f1:.4f}")
        print(f"\nConfusion Matrix:\n{cm}")
        print(f"\nClassification Report:")
        print(classification_report(all_labels, all_preds))
        print(f"{'='*50}\n")
        
        return acc, f1
    
    def train(self, num_epochs=30):
        print(f"\n{'='*60}")
        print(f"Training {self.model_name} with v4 Optimizations")
        print(f"Device: {self.device}, AMP: {self.scaler is not None}")
        print(f"{'='*60}\n")
        
        for epoch in range(1, num_epochs + 1):
            t_loss, t_acc, t_f1 = self.train_epoch(epoch)
            v_loss, v_acc, v_f1 = self.validate_epoch(epoch)
            print(f"Epoch {epoch}/{num_epochs} - Train: Loss={t_loss:.4f}, Acc={t_acc:.4f}, F1={t_f1:.4f} | Val: Loss={v_loss:.4f}, Acc={v_acc:.4f}, F1={v_f1:.4f}")
        
        self.writer.close()
        return self.test()

# %% [markdown]
# ## 7. Main Execution

# %%
if __name__ == "__main__":
    print("Loading data...")
    train_loader, val_loader, test_loader, seq_len = get_dataloaders(config.BATCH_SIZE)
    print(f"Sequence length: {seq_len}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    models = {
        "ECG_ResNet": ECG_ResNet(num_classes=config.NUM_CLASSES),
        "ECG_LSTM": ECG_LSTM(num_classes=config.NUM_CLASSES),
        "ECG_MultiModule": ECG_MultiModule(num_classes=config.NUM_CLASSES, seq_len=seq_len),
        "ECG_PhysioWave": ECG_PhysioWave(num_classes=config.NUM_CLASSES, seq_len=seq_len)
    }
    
    results = {}
    for name, model in models.items():
        trainer = OptimizedTrainer(model, name, train_loader, val_loader, test_loader)
        acc, f1 = trainer.train(config.NUM_EPOCHS)
        results[name] = {"accuracy": acc, "f1_score": f1}
    
    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)
    for name, metrics in results.items():
        print(f"{name}: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1_score']:.4f}")
    print("="*60)
