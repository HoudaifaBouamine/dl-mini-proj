# Technical Report: ECG Arrhythmia Classification

## 1. Project Overview
The goal of this project is to implement and compare multiple state-of-the-art Deep Learning architectures for the classification of electrocardiogram (ECG) signals. We focus on the MIT-BIH Arrhythmia Database, a standard benchmark in the field.

## 2. Dataset Description
*   **Source:** MIT-BIH Arrhythmia Database (via Kaggle CSV preprocessed version).
*   **Format:** 1D signals of length 187.
*   **Classes:** 5 categories of heartbeats:
    - 0: Normal
    - 1: Supraventricular premature beat
    - 2: Premature ventricular contraction
    - 3: Fusion of ventricular and normal beat
    - 4: Unclassifiable beat

## 3. Implemented Architectures
We have implemented a unified pipeline that allows training and comparing the following models:

### A. 1D ResNet (CNN)
*   **Description:** A 34-layer 1D Residual Network inspired by the Nature Medicine research "Cardiologist-level arrhythmia detection".
*   **Key Advantage:** Skip connections allow for training very deep networks without vanishing gradient issues, effectively capturing local morphological features of the ECG waveform.

### B. Bidirectional LSTM (RNN)
*   **Description:** A recurrent neural network with 2 layers of bidirectional LSTM cells.
*   **Key Advantage:** Excellent at capturing long-term temporal dependencies and rhythmic patterns in the signal.

### C. Multi-Module (CNN + LSTM + Transformer)
*   **Description:** A hybrid architecture that uses a CNN for local feature extraction, followed by an LSTM for temporal context, and finally a Transformer Encoder for global dependency modeling.
*   **Key Advantage:** Combines the strengths of all three paradigms (Local, Sequential, and Global).

### D. PhysioWave-inspired (Multi-Scale Transformer)
*   **Description:** A multi-scale approach that extracts features using different kernel sizes (simulating wavelet decomposition at different scales) followed by a Transformer Encoder.
*   **Key Advantage:** Robust to noise and capable of analyzing features at multiple resolutions simultaneously.

## 4. Training & Evaluation Strategy
*   **Framework:** PyTorch.
*   **Optimizer:** Adam with a learning rate of 1e-3.
*   **Loss Function:** Cross-Entropy Loss.
*   **Logging:** Integrated with TensorBoard for real-time monitoring of loss and accuracy.
*   **Validation:** 20% of the training data is used for validation to save the best model weights.

## 5. Deployment Plan
The best-performing model will be exported and integrated into a web application (e.g., Streamlit or Gradio) for real-time prediction on user-uploaded ECG samples.
