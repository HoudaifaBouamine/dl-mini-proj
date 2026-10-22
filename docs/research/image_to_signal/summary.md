# ECG Image to 1D Signal (Digitization)

## Overview
This approach uses Deep Learning or Computer Vision techniques to convert a 2D image of an ECG (e.g., a photo of a medical record) back into a 1D numerical time-series array (voltages over time). Once extracted, the 1D signal is passed to a standard signal classifier (like a 1D ResNet or your PhysioWave model).

## Typical Architectures Found Online
1. **U-Net / Semantic Segmentation:** Training a U-Net to segment the "signal line" from the background grid paper, text, and noise.
2. **Autoencoders:** Using convolutional autoencoders to denoise the image and isolate the continuous ECG wave.
3. **Heuristic/OpenCV:** Many projects use traditional computer vision (color masking, contour detection) instead of Deep Learning to extract the signal coordinates, followed by interpolation.

## Pros & Cons
*   **Pros:** Highly modular. Once the signal is extracted, you can reuse all the state-of-the-art 1D sequence models (LSTM, 1D-CNN, Wavelet Transformers) which are generally much faster and more accurate for time-series data than 2D image classifiers.
*   **Cons:** Very difficult to get perfect extraction. If the segmentation model messes up a QRS peak, the downstream classifier will fail.

## Compatibility with Project Requirements
**VERY HIGH COMPATIBILITY.**
*   **Constraints:** Uses Autoencoders/CNNs (allowed) and outputs a signal for prediction.
*   **Architectural Fit:** This is arguably the best approach for a "Next-Generation" application. You can build a 2-stage pipeline:
    1. **Stage 1 (Image -> Signal):** An Autoencoder/U-Net that extracts the signal from a phone photo.
    2. **Stage 2 (Signal -> Class):** Your planned PhysioWave 1D Transformer that classifies the arrhythmia.
*   **Recommendation:** Having a separate AI module specifically for digitization, combined with your 1D classification model, makes for a highly advanced, multi-modal project that will definitely impress the jury.
