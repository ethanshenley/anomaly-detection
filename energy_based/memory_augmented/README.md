# Memory Augmented Autoencoder for Anomaly Detection

This demo implements a **memory augmented autoencoder** using PyTorch on the MNIST dataset. The model
consists of an encoder, a learnable memory module, and a decoder. The encoder maps input images to a latent
representation; the memory module (a learnable bank of memory vectors) refines this representation via a weighted
combination based on similarity; and the decoder reconstructs the image from the refined latent vector.
The network is trained using a reconstruction loss (MSE) along with a latent consistency loss that encourages the
original and refined latent vectors to be similar. At test time, higher reconstruction errors indicate anomalies.

## Anomaly Detection Use Case

**When to Use a Memory Augmented Autoencoder:**

- **Enhanced Feature Representation:**  
  The memory module enables the network to capture and recall prototypical normal patterns. Anomalies that do not match these patterns result in higher reconstruction error.

- **Robust Reconstruction:**  
  By learning a refined latent representation, the model is better able to reconstruct normal data accurately while failing to do so for anomalous data.

- **Unsupervised Detection:**  
  This approach is useful when only normal data is available for training, and anomalies are detected via reconstruction error.

## Setup

1. **Clone the repository** and navigate to the `energy_based/memory_augmented/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Download the MNIST dataset.
    * Train the memory augmented autoencoder on normal images.
    * Reconstruct a test image.
    * Display and save the original and reconstructed images. A high reconstruction error may signal an anomaly.