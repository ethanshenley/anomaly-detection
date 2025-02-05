# One-Class Neural Network for Anomaly Detection

This demo implements a simple **one-class neural network** using PyTorch on the MNIST dataset.
The model is trained solely on normal data (digit 0 by default) so that it learns to output a value close to 0.
During testing, the absolute value of the network's output is used as an anomaly score.
Samples with higher anomaly scores are interpreted as anomalous.

## Anomaly Detection Use Case

**When to Use a One-Class Neural Network:**

- **Training on Normal Data Only:**  
  In many real-world scenarios, labeled anomalies are scarce. A one-class neural network is trained exclusively on normal data, making it ideal for such cases.

- **Anomaly Scoring:**  
  By enforcing a baseline output (e.g., 0) for normal data, deviations from this baseline during inference serve as an effective anomaly score.

- **Baseline Approach:**  
  This simple method can be used as a baseline or integrated into more complex anomaly detection systems.

## Setup

1. **Clone the repository** and navigate to the `one_class/one_class_nn/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Download the MNIST dataset.
    * Filter the training data to include only the normal class (digit 0 by default).
    * Train the one-class neural network on the filtered data.
    * Evaluate the model on the full test set and compute anomaly scores.
    * Visualize the distribution of anomaly scores and report the average scores for normal and anomalous classes.