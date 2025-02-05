# Contrastive Learning for Anomaly Detection

This demo implements a **contrastive learning** framework using PyTorch on the MNIST dataset.
By generating two augmented views of each image, the model learns a representation where similar (augmented) samples are
closely clustered in the latent space, while dissimilar samples are pushed apart.

## Anomaly Detection Use Case

**When to Use Contrastive Learning:**

- **Robust Representation Learning:**  
  Contrastive learning is effective in learning robust representations by maximizing agreement between different augmentations
  of the same input. This helps the model capture the intrinsic structure of the data without relying on labels.

- **Clustering of Normal Data:**  
  Once trained, normal data tends to form tight clusters in the learned latent space. Anomalous inputs, which deviate from
  normal patterns, will often fall outside these clusters, allowing for effective anomaly detection.

- **Unsupervised Approach:**  
  This technique does not require labeled anomalies, making it ideal for scenarios where anomalous samples are rare or
  hard to obtain.

## Setup

1. **Clone the repository** and navigate to the `self_supervised/contrastive_learning/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

   * Download the MNIST dataset.
   * Generate two augmented views for each image.
   * Train the contrastive model for the specified number of epochs.
   * Visualize the learned embeddings using t-SNE.
   * Save the t-SNE plot to the outputs/ directory.

   Enjoy exploring and extending this contrastive learning demo!