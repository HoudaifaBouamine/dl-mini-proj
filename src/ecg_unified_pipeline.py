"""
Unified ECG Deep Learning Pipeline
Optimized for Kaggle Kernels.

This script contains reusable components to train and compare multiple 
Deep Learning models (CNN, RNN, Transformer) on ECG data.

To use in Kaggle:
1. Create a new notebook/script.
2. Add the 'Heartbeat' dataset to the session.
3. Run the pipeline to train and save models.
"""

# %% [markdown]
# ## 1. Setup & Imports
# Install necessary libraries if not present:
# `!pip install torch torchvision pandas numpy scikit-learn tqdm tensorboard`

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
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# %% [markdown]
# ## 2. Data Loading & Preprocessing
# We assume the data is in CSV format (like the Kaggle MIT-BIH Heartbeat dataset),
# where each row is a sample, the last column is the label, and previous columns are the 1D signal.

# %%
class ECGDataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        data_path: Path to the CSV file.
        """
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path, header=None)
        
        # Assume last column is the label
        self.labels = df.iloc[:, -1].values.astype(np.int64)
        
        # Previous columns are the signal features
        self.signals = df.iloc[:, :-1].values.astype(np.float32)
        
        # Add a channel dimension for 1D convolutions: shape becomes (N, 1, Sequence_Length)
        self.signals = np.expand_dims(self.signals, axis=1)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        
        if self.transform:
            signal = self.transform(signal)
            
        return torch.tensor(signal), torch.tensor(label)

def get_dataloaders(train_csv, test_csv, batch_size=64):
    """
    Utility function to create train/val/test dataloaders.
    If you only have one large CSV, you can split it using sklearn.
    """
    train_dataset = ECGDataset(train_csv)
    test_dataset = ECGDataset(test_csv)
    
    # Split train into train and val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, train_dataset.signals.shape[2]

# %% [markdown]
# ## 3. Model Architectures
# We define 3 distinct architectures: CNN (ResNet-like), RNN (LSTM), and Transformer.
# They all share a standard input/output interface to easily swap them.

# %%
# --- Model 1: 1D ResNet (ResNet-34 Inspiration) ---
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


# --- Model 2: LSTM ---
class ECG_LSTM(nn.Module):
    def __init__(self, num_classes=5, input_size=1, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, 
                            num_layers=num_layers, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        x = x.permute(0, 2, 1) # (batch, seq, 1)
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :]) # Use last hidden state


# --- Model 3: Multi-Module (CNN + LSTM + Transformer) ---
class ECG_MultiModule(nn.Module):
    def __init__(self, num_classes=5, seq_len=187):
        super().__init__()
        # CNN Module
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # LSTM Module
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True, bidirectional=True)
        # Transformer Module
        encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x) # (batch, 32, seq/2)
        x = x.permute(0, 2, 1) # (batch, seq/2, 32)
        x, _ = self.lstm(x) # (batch, seq/2, 128)
        x = self.transformer(x) # (batch, seq/2, 128)
        x = x.permute(0, 2, 1) # (batch, 128, seq/2)
        x = self.avgpool(x).squeeze(-1)
        return self.fc(x)


# --- Model 4: PhysioWave-inspired (Multi-Scale Wavelet Transformer) ---
class ECG_PhysioWave_Basic(nn.Module):
    def __init__(self, num_classes=5, seq_len=187):
        super().__init__()
        # Multi-scale feature extraction (simulating wavelet decomposition)
        self.scale1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.scale2 = nn.Conv1d(1, 32, kernel_size=7, padding=3)
        self.scale3 = nn.Conv1d(1, 32, kernel_size=11, padding=5)
        
        self.fusion = nn.Conv1d(96, 64, kernel_size=1)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        s1 = torch.relu(self.scale1(x))
        s2 = torch.relu(self.scale2(x))
        s3 = torch.relu(self.scale3(x))
        
        x = torch.cat([s1, s2, s3], dim=1) # (batch, 96, seq)
        x = self.fusion(x).permute(0, 2, 1) # (batch, seq, 64)
        x = self.transformer(x)
        x = x.permute(0, 2, 1) # (batch, 64, seq)
        x = self.avgpool(x).squeeze(-1)
        return self.fc(x)



# %% [markdown]
# ## 4. Unified Training Pipeline with Logging
# This class handles training, validation, saving best checkpoints, and TensorBoard logging.

# %%
class UnifiedTrainer:
    def __init__(self, model, model_name, train_loader, val_loader, 
                 learning_rate=1e-3, log_dir="runs"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # TensorBoard Writer
        run_name = f"{model_name}_{int(time.time())}"
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
        self.best_val_loss = float('inf')
        self.save_path = f"best_{model_name}.pth"
        
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds, all_labels = [], []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for signals, labels in pbar:
            signals, labels = signals.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
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
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/Train', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
        
        return epoch_loss, epoch_acc

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
        
        # Log to TensorBoard
        self.writer.add_scalar('Loss/Validation', epoch_loss, epoch)
        self.writer.add_scalar('Accuracy/Validation', epoch_acc, epoch)
        
        # Save best model
        if epoch_loss < self.best_val_loss:
            self.best_val_loss = epoch_loss
            torch.save(self.model.state_dict(), self.save_path)
            
        return epoch_loss, epoch_acc

    def train(self, num_epochs=10):
        print(f"--- Starting Training for {self.model_name} ---")
        for epoch in range(1, num_epochs + 1):
            t_loss, t_acc = self.train_epoch(epoch)
            v_loss, v_acc = self.validate_epoch(epoch)
            print(f"Epoch {epoch}/{num_epochs} - Train Loss: {t_loss:.4f}, Train Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f}, Val Acc: {v_acc:.4f}")
            
        self.writer.close()
        print(f"Training complete. Best validation loss: {self.best_val_loss:.4f}. Model saved to {self.save_path}\n")

# %% [markdown]
# ## 5. Execution Example (How to run all models)
# To run this in Colab, uncomment the code below and provide the correct paths to your CSV files.

# %%
def run_experiment():
    # 1. Provide paths to Kaggle Heartbeat dataset CSVs (MIT-BIH)
    train_csv_path = "/kaggle/input/heartbeat/mitbih_train.csv"
    test_csv_path = "/kaggle/input/heartbeat/mitbih_test.csv"

    
    # Check if files exist before running
    if not os.path.exists(train_csv_path) or not os.path.exists(test_csv_path):
        print("Data files not found. Please upload them to Colab and update the paths.")
        return
        
    # 2. Get Dataloaders
    train_loader, val_loader, test_loader, seq_len = get_dataloaders(train_csv_path, test_csv_path, batch_size=128)
    print(f"Sequence Length detected: {seq_len}")
    
    # 3. Instantiate Models
    num_classes = 5 # MIT-BIH has 5 heartbeat classes
    
    models_to_train = {
        "ECG_ResNet": ECG_ResNet(num_classes=num_classes),
        "ECG_LSTM": ECG_LSTM(num_classes=num_classes),
        "ECG_MultiModule": ECG_MultiModule(num_classes=num_classes, seq_len=seq_len),
        "ECG_PhysioWave": ECG_PhysioWave_Basic(num_classes=num_classes, seq_len=seq_len)
    }

    
    # 4. Train sequentially and compare
    num_epochs = 10
    
    for name, model in models_to_train.items():
        trainer = UnifiedTrainer(model=model, model_name=name, 
                                 train_loader=train_loader, val_loader=val_loader)
        trainer.train(num_epochs=num_epochs)

    # Note: To view TensorBoard logs in Colab, run the following in a new cell:
    # %load_ext tensorboard
    # %tensorboard --logdir runs

if __name__ == "__main__":
    run_experiment()

