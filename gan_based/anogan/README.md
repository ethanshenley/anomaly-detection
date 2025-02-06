# AnoGAN for Anomaly Detection

This demo implements **AnoGAN** using PyTorch on the MNIST dataset. A GAN is first trained on normal data,
and then for a given test image, latent space optimization (inversion) is performed to find the latent vector
that best reconstructs the image via the generator. The reconstruction error serves as an anomaly score:
a high error suggests that the image deviates from the normal data distribution.

## Anomaly Detection Use Case

**When to Use AnoGAN:**

- **Unsupervised Anomaly Detection:**  
  AnoGAN is useful when only normal data is available for training. It learns the distribution of normal images,
  and anomalies are identified based on poor reconstruction by the generator.

- **Complex Data Distributions:**  
  GANs can capture complex data distributions, making them effective in detecting subtle deviations that may indicate anomalies.

- **Medical Imaging, Industrial Inspection, etc.:**  
  Applications where normal appearance is well-defined and anomalies are deviations from that appearance.

## Setup

1. **Clone the repository** and navigate to the `gan_based/anogan/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Train a GAN on the MNIST dataset.
    * Perform latent space optimization on a test image to reconstruct it.
    * Compute the reconstruction error as an anomaly score.
    * Display and save the original and reconstructed images.