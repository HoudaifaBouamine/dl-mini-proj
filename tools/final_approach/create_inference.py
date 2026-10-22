# Run from the repository root (paths below are relative to it).
import json

path = 'notebooks/final_approach/4_inference_and_deployment.ipynb'
nb = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🚀 PhysioAI: Unified Inference & Deployment\n",
    "\n",
    "This notebook demonstrates how the three trained production models are integrated into a single diagnostic pipeline."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import numpy as np\n",
    "import cv2\n",
    "import matplotlib.pyplot as plt\n",
    "import os\n",
    "\n",
    "DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "class_names = ['Normal', 'SVEB', 'VEB', 'Fusion', 'Unknown']\n",
    "\n",
    "# 1. Model Definitions (Must match training architectures)\n",
    "class SimpleCNN1D(nn.Module):\n",
    "    def __init__(self, num_classes=5):\n",
    "        super().__init__()\n",
    "        self.features = nn.Sequential(nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2), \n",
    "                                     nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2))\n",
    "        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 46, 128), nn.ReLU(), nn.Linear(128, num_classes))\n",
    "    def forward(self, x): return self.classifier(self.features(x))\n",
    "\n",
    "class DigitizerCNN(nn.Module):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), \n",
    "                                     nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), \n",
    "                                     nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))\n",
    "        self.regressor = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16 * 16, 512), nn.ReLU(), nn.Linear(512, 187))\n",
    "    def forward(self, x): return self.regressor(self.features(x))\n",
    "\n",
    "class ImageClassifier2D(nn.Module):\n",
    "    def __init__(self, num_classes=5):\n",
    "        super().__init__()\n",
    "        self.features = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), \n",
    "                                     nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), \n",
    "                                     nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))\n",
    "        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16 * 16, 128), nn.ReLU(), nn.Linear(128, num_classes))\n",
    "    def forward(self, x): return self.classifier(self.features(x))\n",
    "\n",
    "print('✅ Model architectures defined.')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def load_models():\n",
    "    paths = {\n",
    "        'signal_to_class': 'models/signal_to_class.pth',\n",
    "        'image_to_signal': 'models/image_to_signal.pth',\n",
    "        'image_to_class': 'models/image_to_class.pth'\n",
    "    }\n",
    "    models = {}\n",
    "    for name, p in paths.items():\n",
    "        if os.path.exists(p):\n",
    "            if name == 'signal_to_class': m = SimpleCNN1D()\n",
    "            elif name == 'image_to_signal': m = DigitizerCNN()\n",
    "            else: m = ImageClassifier2D()\n",
    "            \n",
    "            m.load_state_dict(torch.load(p, map_location=DEVICE))\n",
    "            m.to(DEVICE).eval()\n",
    "            models[name] = m\n",
    "            print(f'✅ Loaded {name}')\n",
    "        else:\n",
    "            print(f'⚠️ Warning: {p} not found. Run training notebooks first.')\n",
    "    return models\n",
    "\n",
    "loaded_models = load_models()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Unified Diagnostic Interface\n",
    "We provide a wrapper that can handle both raw signals and images."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "def diagnostic_system(input_data, mode='signal'):\n",
    "    \"\"\"\n",
    "    High-level API for deployment.\n",
    "    mode: 'signal' (1D array) or 'image' (2D array/path)\n",
    "    \"\"\"\n",
    "    if mode == 'signal':\n",
    "        # Path 1: Direct 1D Diagnosis\n",
    "        m = loaded_models.get('signal_to_class')\n",
    "        if not m: return \"Model not loaded\"\n",
    "        inp = torch.tensor(input_data, dtype=torch.float32).view(1, 1, -1).to(DEVICE)\n",
    "        with torch.no_grad():\n",
    "            out = m(inp)\n",
    "            prob = torch.softmax(out, 1)\n",
    "            pred = out.argmax(1).item()\n",
    "        return f\"Diagnosis: {class_names[pred]} ({prob.max().item()*100:.1f}% confidence)\"\n",
    "    \n",
    "    elif mode == 'image':\n",
    "        # Path 3: Visual Diagnosis\n",
    "        m = loaded_models.get('image_to_class')\n",
    "        if not m: return \"Model not loaded\"\n",
    "        # Preprocess image\n",
    "        if isinstance(input_data, str): img = cv2.imread(input_data, 0)\n",
    "        else: img = input_data\n",
    "        img_t = torch.tensor(img/255.0, dtype=torch.float32).view(1, 1, 128, 128).to(DEVICE)\n",
    "        \n",
    "        with torch.no_grad():\n",
    "            out = m(img_t)\n",
    "            pred = out.argmax(1).item()\n",
    "            \n",
    "            # Path 2: Reconstruction (Optional but useful for clinical verification)\n",
    "            m_rec = loaded_models.get('image_to_signal')\n",
    "            if m_rec:\n",
    "                rec_signal = m_rec(img_t).cpu().numpy().flatten()\n",
    "                plt.plot(rec_signal)\n",
    "                plt.title(f\"Reconstructed Signal - Predicted: {class_names[pred]}\")\n",
    "                plt.show()\n",
    "                \n",
    "        return f\"Visual Diagnosis: {class_names[pred]}\"\n",
    "\n",
    "print('✅ Diagnostic system ready for deployment simulation.')"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
print('✅ Inference notebook created.')
