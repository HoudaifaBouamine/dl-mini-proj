# Run from the repository root (paths below are relative to it).
import json

path = 'notebooks/final_approach/0_research_and_problematic.ipynb'
with open(path, 'r') as f: nb = json.load(f)

# Clear existing cells and rebuild
nb['cells'] = []

# Title and Problematic
nb['cells'].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# 🔬 PhysioAI: Advanced ECG Diagnostic Framework\n",
        "\n",
        "## 1. Research Problematic\n",
        "The digitization and automation of ECG diagnosis are critical for modern cardiology. Our research addresses three primary bottlenecks:\n",
        "- **Signal Processing**: Efficiently extracting features from noisy 1D electrical signals.\n",
        "- **Digitization**: Bridging the gap between legacy paper records and digital systems using computer vision.\n",
        "- **Clinical Accuracy**: Maintaining high sensitivity for rare arrhythmias in imbalanced datasets.\n",
        "\n",
        "## 2. Dataset Analysis (MIT-BIH Arrhythmia Database)\n",
        "We utilize the gold-standard MIT-BIH dataset, which contains five primary clinical categories:\n",
        "- **N**: Normal Beat\n",
        "- **S**: Supraventricular Ectopic Beat\n",
        "- **V**: Ventricular Ectopic Beat\n",
        "- **F**: Fusion Beat\n",
        "- **Q**: Unknown/Unclassifiable (includes Paced beats and noise)"
    ]
})

# Code for distribution and samples
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import os\n",
        "\n",
        "DATA_PATH = '../../data/mitbih_train.csv'\n",
        "class_names = ['Normal', 'SVEB', 'VEB', 'Fusion', 'Unknown']\n",
        "colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f', '#9b59b6']\n",
        "\n",
        "if os.path.exists(DATA_PATH):\n",
        "    df = pd.read_csv(DATA_PATH, header=None)\n",
        "    y = df.iloc[:, 187].values\n",
        "    X = df.iloc[:, :187].values\n",
        "    \n",
        "    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), facecolor='#f8f9fa')\n",
        "    \n",
        "    # 1. Distribution Plot\n",
        "    counts = pd.Series(y).value_counts().sort_index()\n",
        "    ax1.bar(class_names, counts.values, color=colors, alpha=0.8)\n",
        "    ax1.set_title('Class Distribution (Log Scale)', fontsize=14, fontweight='bold')\n",
        "    ax1.set_yscale('log')\n",
        "    ax1.set_ylabel('Frequency')\n",
        "    ax1.grid(axis='y', linestyle='--', alpha=0.6)\n",
        "    \n",
        "    # 2. Sample Signals\n",
        "    for i in range(5):\n",
        "        idx = np.where(y == i)[0][0]\n",
        "        ax2.plot(X[idx], label=class_names[i], color=colors[i], linewidth=2)\n",
        "    ax2.set_title('ECG Morphologies by Class', fontsize=14, fontweight='bold')\n",
        "    ax2.legend()\n",
        "    ax2.grid(alpha=0.3)\n",
        "    \n",
        "    plt.tight_layout()\n",
        "    plt.show()\n",
        "else:\n",
        "    print('⚠️ Data not found.')"
    ]
})

# The 3 Path Strategy
nb['cells'].append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Implementation Strategy: Three Diagnostic Paths\n",
        "\n",
        "### 📈 Path 1: Signal-to-Class (Direct Diagnosis)\n",
        "**Approach**: Uses a 1D Convolutional Neural Network (CNN) to process the raw voltage time-series.\n",
        "- **Data**: Raw 1D signals normalized to [0, 1].\n",
        "- **Outcome**: Categorical diagnosis.\n",
        "\n",
        "### 🖼️ Path 2: Image-to-Signal (Reconstruction)\n",
        "**Approach**: Uses a 2D CNN Encoder-Decoder (Regressor) to transform visual ECG plots back into 1D numerical data.\n",
        "- **Data**: Synthetic ECG images generated from real signals with clinical grid noise.\n",
        "- **Outcome**: Digitized 1D signal for telemetry integration.\n",
        "\n",
        "### 🏥 Path 3: Image-to-Class (Visual AI)\n",
        "**Approach**: Mimics the human cardiologist's visual process using a 2D CNN Classifier on ECG plots.\n",
        "- **Data**: 2D plots of heartbeats.\n",
        "- **Outcome**: Immediate diagnostic label from images."
    ]
})

with open(path, 'w') as f:
    json.dump(nb, f, indent=1)
print('✅ Research notebook updated.')
