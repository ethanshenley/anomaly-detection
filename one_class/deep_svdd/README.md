# Deep SVDD for Anomaly Detection

This demo implements **Deep SVDD** using PyTorch on the MNIST dataset. The goal is to learn a compact representation of normal data by minimizing the squared distance of the network's output from a pre-computed center. Anomalous samples—which do not conform to the learned normal distribution—typically exhibit higher distances from this center.

## Anomaly Detection Use Case

**When to Use Deep SVDD:**

- **One-Class Classification:**  
  Deep SVDD is particularly useful when you have data from only one class (normal data) and want to identify anomalies that deviate from this normal distribution.

- **Compact Representations:**  
  By forcing the network to map normal data close to a center in the latent space, anomalies naturally emerge as outliers with higher distances.

- **Unsupervised Learning:**  
  The method does not require labeled anomalies for training, making it suitable for many real-world applications where anomalous data is rare.

## Setup

1. **Clone the repository** and navigate to the `one_class/deep_svdd/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Download the MNIST dataset.
    * Compute a center in the latent space using the training data.
    * Train the Deep SVDD model by minimizing the squared distance from this center.
    * Evaluate the model on the test data and plot a histogram of distances, which can help identify anomalies.

