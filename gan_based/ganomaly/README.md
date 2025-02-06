# Ganomaly for Anomaly Detection

This demo implements **Ganomaly** using PyTorch on the MNIST dataset. Ganomaly features an encoder-decoder-encoder architecture,
where the generator reconstructs the input image and a secondary encoder extracts latent features from the reconstruction.
The anomaly score is computed as the difference between the latent representations of the input and the reconstruction.
Higher scores indicate that the input deviates from the normal data distribution and may be anomalous.

## Anomaly Detection Use Case

**When to Use Ganomaly:**

- **Reconstruction-based Anomaly Detection:**  
  Ganomaly learns to reconstruct normal data, and anomalies are detected based on high reconstruction errors or discrepancies in latent features.

- **Effective for Complex Data:**  
  The dual encoder structure helps in capturing subtle differences between the input and its reconstruction, making it useful in domains such as industrial inspection or medical imaging.

- **Unsupervised Training:**  
  Ganomaly is trained solely on normal data, so it is suitable when labeled anomalies are scarce.

## Setup

1. **Clone the repository** and navigate to the `gan_based/ganomaly/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Train the Ganomaly model on the MNIST dataset.
    * Reconstruct a test image using the encoder-decoder-encoder architecture.
    * Compute an anomaly score based on the difference between latent representations.
    * Display and save the original and reconstructed images along with the anomaly score.