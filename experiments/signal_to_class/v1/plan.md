# Plan for v1

## Objective
Establish the baseline unified ECG pipeline with 4 distinct deep learning architectures:
1. 1D ResNet (CNN)
2. Bi-LSTM (RNN)
3. Multi-Module (CNN + LSTM + Transformer)
4. PhysioWave (Multi-Scale Wavelet Transformer)

## Action Items
- Write the PyTorch code for dataset loading (MIT-BIH).
- Implement the model classes.
- Create a unified `UnifiedTrainer` class with TensorBoard logging.
- Convert the raw `.py` script to a `.ipynb` Jupyter Notebook for execution on Kaggle.
