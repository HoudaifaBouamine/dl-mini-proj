# Plan for v4 (Optimization & Fine-Tuning)

## Objective
Once `v3` establishes a working baseline and we have our initial accuracy metrics, `v4` will focus purely on **Model Optimization and Training Efficiency** to squeeze out maximum performance from the architectures (especially the Multi-Module and PhysioWave models).

## Action Items (To be implemented in `ecg_pipeline_v4.ipynb`)

1.  **Automatic Mixed Precision (AMP)**: Implement `torch.amp` in the `UnifiedTrainer` to speed up GPU training and reduce memory footprint, allowing for potentially larger batch sizes.
2.  **Learning Rate Scheduling**: Add `ReduceLROnPlateau` or `CosineAnnealingLR` to dynamically adjust the learning rate during training, helping the models converge to better local minima.
3.  **Advanced Augmentation**: Introduce minor 1D signal augmentations (e.g., random noise injection, slight temporal shifting) during the Dataset `__getitem__` to improve model robustness and generalization.
4.  **Focal Loss (Optional)**: If `v3` results show poor performance on the minority classes ('F' and 'S' beats), replace the standard `CrossEntropyLoss` with a Focal Loss implementation to heavily penalize misclassifications of rare arrhythmias.
