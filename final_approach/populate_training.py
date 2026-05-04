import json
import os

def populate_signal_to_class():
    path = '/home/me/mini-proj/final_approach/1_train_signal_to_class.ipynb'
    with open(path, 'r') as f: nb = json.load(f)
    
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "class SimpleCNN1D(nn.Module):\n",
            "    def __init__(self, num_classes=5):\n",
            "        super().__init__()\n",
            "        self.features = nn.Sequential(\n",
            "            nn.Conv1d(1, 32, kernel_size=5, padding=2),\n",
            "            nn.ReLU(),\n",
            "            nn.MaxPool1d(2),\n",
            "            nn.Conv1d(32, 64, kernel_size=5, padding=2),\n",
            "            nn.ReLU(),\n",
            "            nn.MaxPool1d(2),\n",
            "            nn.Dropout(0.2)\n",
            "        )\n",
            "        self.classifier = nn.Sequential(\n",
            "            nn.Flatten(),\n",
            "            nn.Linear(64 * 46, 128),\n",
            "            nn.ReLU(),\n",
            "            nn.Linear(128, num_classes)\n",
            "        )\n",
            "    def forward(self, x): return self.classifier(self.features(x))\n",
            "\n",
            "model = SimpleCNN1D().to(DEVICE)\n",
            "criterion = nn.CrossEntropyLoss()\n",
            "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
            "\n",
            "# Data Prep\n",
            "X = torch.tensor(df_sample.iloc[:, :187].values, dtype=torch.float32).unsqueeze(1).to(DEVICE)\n",
            "y = torch.tensor(df_sample.iloc[:, 187].values, dtype=torch.long).to(DEVICE)\n",
            "loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)\n",
            "\n",
            "print('🚀 Production Training: Signal-to-Class')\n",
            "for epoch in range(2):\n",
            "    model.train()\n",
            "    total_loss, correct = 0, 0\n",
            "    for signals, labels in loader:\n",
            "        optimizer.zero_grad()\n",
            "        outputs = model(signals)\n",
            "        loss = criterion(outputs, labels)\n",
            "        loss.backward()\n",
            "        optimizer.step()\n",
            "        total_loss += loss.item()\n",
            "        correct += (outputs.argmax(1) == labels).sum().item()\n",
            "    \n",
            "    avg_loss = total_loss / len(loader)\n",
            "    acc = correct / len(df_sample)\n",
            "    print(f'Epoch {epoch+1}: Loss={avg_loss:.4f}, Accuracy={acc:.4f}')\n",
            "\n",
            "if not os.path.exists('models'): os.makedirs('models')\n",
            "torch.save(model.state_dict(), 'models/signal_to_class.pth')\n",
            "print('✅ Model Saved to models/signal_to_class.pth')"
        ]
    })
    with open(path, 'w') as f: json.dump(nb, f, indent=1)

def populate_image_to_signal():
    path = '/home/me/mini-proj/final_approach/2_train_image_to_signal.ipynb'
    with open(path, 'r') as f: nb = json.load(f)
    
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import cv2\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "def fast_signal_to_image(signal, img_size=(128, 128)):\n",
            "    # Faster generation using direct numpy plotting for production pipelines\n",
            "    img = np.zeros((img_size[0], img_size[1]), dtype=np.float32)\n",
            "    x_coords = np.linspace(0, img_size[1]-1, len(signal)).astype(int)\n",
            "    y_coords = ((1 - signal) * (img_size[0]-1)).astype(int)\n",
            "    for i in range(len(x_coords)-1):\n",
            "        cv2.line(img, (x_coords[i], y_coords[i]), (x_coords[i+1], y_coords[i+1]), 1.0, 1)\n",
            "    return img\n",
            "\n",
            "class DigitizerCNN(nn.Module):\n",
            "    def __init__(self):\n",
            "        super().__init__()\n",
            "        self.features = nn.Sequential(\n",
            "            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), \n",
            "            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n",
            "            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)\n",
            "        )\n",
            "        self.regressor = nn.Sequential(\n",
            "            nn.Flatten(),\n",
            "            nn.Linear(64 * 16 * 16, 512),\n",
            "            nn.ReLU(),\n",
            "            nn.Linear(512, 187)\n",
            "        )\n",
            "    def forward(self, x): return self.regressor(self.features(x))\n",
            "\n",
            "model = DigitizerCNN().to(DEVICE)\n",
            "criterion = nn.MSELoss()\n",
            "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
            "\n",
            "print('🎨 Generating training images on-the-fly...')\n",
            "X_imgs = torch.tensor([fast_signal_to_image(s) for s in df_sample.iloc[:200, :187].values], dtype=torch.float32).unsqueeze(1).to(DEVICE)\n",
            "y_sigs = torch.tensor(df_sample.iloc[:200, :187].values, dtype=torch.float32).to(DEVICE)\n",
            "\n",
            "print('🚀 Production Training: Image-to-Signal Reconstruction')\n",
            "for epoch in range(2):\n",
            "    optimizer.zero_grad()\n",
            "    preds = model(X_imgs)\n",
            "    loss = criterion(preds, y_sigs)\n",
            "    loss.backward()\n",
            "    optimizer.step()\n",
            "    print(f'Epoch {epoch+1}: Reconstruction Loss={loss.item():.6f}')\n",
            "\n",
            "if not os.path.exists('models'): os.makedirs('models')\n",
            "torch.save(model.state_dict(), 'models/image_to_signal.pth')\n",
            "print('✅ Model Saved.')"
        ]
    })
    with open(path, 'w') as f: json.dump(nb, f, indent=1)

def populate_image_to_class():
    path = '/home/me/mini-proj/final_approach/3_train_image_to_class.ipynb'
    with open(path, 'r') as f: nb = json.load(f)
    
    nb['cells'].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import cv2\n",
            "def fast_signal_to_image(signal, img_size=(128, 128)):\n",
            "    img = np.zeros((img_size[0], img_size[1]), dtype=np.float32)\n",
            "    x_coords = np.linspace(0, img_size[1]-1, len(signal)).astype(int)\n",
            "    y_coords = ((1 - signal) * (img_size[0]-1)).astype(int)\n",
            "    for i in range(len(x_coords)-1):\n",
            "        cv2.line(img, (x_coords[i], y_coords[i]), (x_coords[i+1], y_coords[i+1]), 1.0, 1)\n",
            "    return img\n",
            "\n",
            "class ImageClassifier2D(nn.Module):\n",
            "    def __init__(self, num_classes=5):\n",
            "        super().__init__()\n",
            "        self.features = nn.Sequential(\n",
            "            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n",
            "            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),\n",
            "            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)\n",
            "        )\n",
            "        self.classifier = nn.Sequential(\n",
            "            nn.Flatten(),\n",
            "            nn.Linear(64 * 16 * 16, 128),\n",
            "            nn.ReLU(),\n",
            "            nn.Linear(128, num_classes)\n",
            "        )\n",
            "    def forward(self, x): return self.classifier(self.features(x))\n",
            "\n",
            "model = ImageClassifier2D().to(DEVICE)\n",
            "criterion = nn.CrossEntropyLoss()\n",
            "optimizer = optim.Adam(model.parameters(), lr=0.001)\n",
            "\n",
            "print('🎨 Generating images for 10% sample...')\n",
            "X_imgs = torch.tensor([fast_signal_to_image(s) for s in df_sample.iloc[:200, :187].values], dtype=torch.float32).unsqueeze(1).to(DEVICE)\n",
            "y_labels = torch.tensor(df_sample.iloc[:200, 187].values, dtype=torch.long).to(DEVICE)\n",
            "\n",
            "print('🚀 Production Training: Image-to-Class Visual Diagnosis')\n",
            "for epoch in range(2):\n",
            "    optimizer.zero_grad()\n",
            "    outputs = model(X_imgs)\n",
            "    loss = criterion(outputs, y_labels)\n",
            "    loss.backward()\n",
            "    optimizer.step()\n",
            "    acc = (outputs.argmax(1) == y_labels).float().mean()\n",
            "    print(f'Epoch {epoch+1}: Loss={loss.item():.4f}, Accuracy={acc:.4f}')\n",
            "\n",
            "if not os.path.exists('models'): os.makedirs('models')\n",
            "torch.save(model.state_dict(), 'models/image_to_class.pth')\n",
            "print('✅ Model Saved.')"
        ]
    })
    with open(path, 'w') as f: json.dump(nb, f, indent=1)

if __name__ == '__main__':
    populate_signal_to_class()
    populate_image_to_signal()
    populate_image_to_class()
    print('✅ Training notebooks populated with production code.')
