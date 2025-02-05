# Variational Autoencoder for Anomaly Detection

This demo implements a **variational autoencoder (VAE)** using PyTorch on the MNIST dataset. The VAE learns a probabilistic latent space that models the distribution of normal data. In anomaly detection, high reconstruction errors or low likelihood under the learned distribution can signal anomalous inputs.

## Anomaly Detection Use Case

**When to Use a Variational Autoencoder:**

- **Probabilistic Modeling:**  
  VAEs provide a probabilistic framework that models uncertainty in the data. This is especially useful for detecting subtle anomalies by assessing how likely a new observation is under the learned distribution.

- **Latent Space Analysis:**  
  The latent space of a VAE can be analyzed to find clusters of normal data. Inputs that do not fall into these clusters may be considered anomalous, which is beneficial in applications like fraud detection or quality control.

- **Generative Capabilities:**  
  VAEs can generate new samples similar to the training data, aiding in understanding the data distribution and synthesizing examples for rare events.

## Setup

1. **Clone the repository** and navigate to the `reconstruction_based/variational_autoencoder/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

* Download the MNIST dataset.
* Train the variational autoencoder for the specified number of epochs.
* Visualize the original and reconstructed images.
* Save the reconstruction plot to the outputs/ directory.

Enjoy exploring and extending this variational autoencoder demo!