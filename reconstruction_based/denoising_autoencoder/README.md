# Denoising Autoencoder for Anomaly Detection

This demo implements a **denoising autoencoder** using PyTorch on the MNIST dataset. The autoencoder is trained to remove noise from input images, reconstructing the clean version.

## Anomaly Detection Use Case

**When to Use a Denoising Autoencoder:**

- **Noise Filtering:**  
  In real-world applications, sensor data or images are often corrupted by noise. A denoising autoencoder learns to recover the underlying clean signal, making it an effective pre-processing step before applying further anomaly detection algorithms.

- **Anomaly Highlighting:**  
  By learning to remove typical noise patterns, the autoencoder can highlight unusual or anomalous features that deviate from the learned distribution. For example, in medical imaging, it can help distinguish imaging artifacts from genuine anomalies.

## Setup

1. **Clone the repository** and navigate to the `reconstruction_based/denoising_autoencoder/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

* Download the MNIST dataset.
* Add noise to the images during training.
* Train the denoising autoencoder for the specified number of epochs.
* Visualize the original, noisy, and denoised images.
* Save the reconstruction plot to the outputs/ directory.