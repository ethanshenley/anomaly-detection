# Vision Transformer Autoencoder for Anomaly Detection

This demo implements a **Vision Transformer (ViT) autoencoder** using PyTorch on the MNIST dataset.
Each input image is split into non-overlapping patches, which are linearly embedded and enriched with
learnable positional encodings. The resulting sequence is processed by transformer encoder layers.
A linear decoder then reconstructs the patches, and the patches are reassembled to form the output image.
Anomaly detection is performed via reconstruction error: images that deviate from normal training data yield higher errors.

## Anomaly Detection Use Case

**When to Use a Vision Transformer for Anomaly Detection:**

- **Capturing Global Context:**  
  ViTs use self-attention to capture long-range dependencies and global context in images, which can be crucial
  for detecting subtle or spatially distributed anomalies.

- **Reconstruction-Based Anomaly Detection:**  
  The autoencoder framework reconstructs images from latent patch representations. High reconstruction errors
  indicate that an input does not conform to the learned normal patterns.

- **Applications:**  
  Suitable for industrial inspection, medical imaging, and other domains where precise reconstruction of normal
  appearance is key to flagging defects or anomalies.

## Setup

1. **Clone the repository** and navigate to the `transformers/vision_transformer/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Download the MNIST dataset.
    * Train the Vision Transformer autoencoder on the training set.
    * Reconstruct a test image and compute reconstruction error (used as an anomaly score).
    * Display and save the original and reconstructed images to the outputs/ directory.