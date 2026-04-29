# Image to Signal (Digitization) - v1

## Objective
Build a Convolutional Deep Learning model that takes a 2D image of an ECG signal (e.g., a photo or scan) and converts it back into a 1D sequence of float values. This acts as the "Stage 1" digitizer before passing the signal to our "Stage 2" arrhythmia classifier (PhysioWave/ResNet).

## Data Strategy
Since we don't have a massive dataset of paired ECG photos and 1D arrays, we will create a **Synthetic Data Generator**. 
We will load the 1D MIT-BIH signals, use `matplotlib` to plot them onto an image with random augmentations (grid lines, noise, varying line thickness), and train the model to reconstruct the original 1D array from that image.

## Architecture
- **Input:** 2D Image (e.g., 128x128 Grayscale or RGB).
- **Encoder:** A series of Convolutional layers (e.g., ResNet blocks) to compress the image into a latent feature vector.
- **Decoder:** Dense layers or a 1D Convolutional Decoder to expand the latent vector into a `(187,)` 1D signal.
- **Loss:** Mean Squared Error (MSE) to measure reconstruction accuracy.
