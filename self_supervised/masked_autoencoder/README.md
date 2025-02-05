# Masked Autoencoder for Anomaly Detection

This demo implements a **masked autoencoder** using PyTorch on the MNIST dataset. The model is trained to reconstruct the original image from a masked version, where a portion of the image is randomly removed.

## Anomaly Detection Use Case

**When to Use a Masked Autoencoder:**

- **Reconstruction of Incomplete Data:**  
  Masked autoencoders learn to fill in missing parts of an image, making them particularly useful for scenarios where data may be partially corrupted or incomplete.

- **Highlighting Deviations:**  
  During inference, if an input image is anomalous, the model's reconstruction—especially in the masked regions—may deviate significantly from the original. This high reconstruction error can be used to flag anomalies.

- **Pre-training for Downstream Tasks:**  
  The representations learned by masked autoencoders can be used for tasks such as classification or segmentation where detecting anomalous regions is critical.

## Setup

1. **Clone the repository** and navigate to the `self_supervised/masked_autoencoder/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

* Download the MNIST dataset.
* Randomly mask a portion of each input image.
* Train the masked autoencoder to reconstruct the original images.
* Visualize the original, masked, and reconstructed images.
* Save the reconstruction plot to the outputs/ directory.

Enjoy exploring and extending this masked autoencoder demo!