# Autoencoder for Anomaly Detection

This demo implements a vanilla autoencoder for anomaly detection using PyTorch on the MNIST dataset. The autoencoder learns to compress and then reconstruct input images. In a real anomaly detection setting, a high reconstruction error can indicate an anomaly.

## Setup

1. **Clone the repository** (if you haven't already) and navigate to the `reconstruction_based/autoencoder/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
