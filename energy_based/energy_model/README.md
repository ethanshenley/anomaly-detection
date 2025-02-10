# Energy-Based Model for Anomaly Detection

This demo implements an **energy-based model** using PyTorch on the MNIST dataset. The model is a simple MLP that assigns a scalar energy value to each input image. During training, negative samples (random noise) are generated, and a margin-ranking loss is used to enforce that normal samples receive lower energy than negatives. At test time, the energy value itself serves as an anomaly score—higher energy indicates that an input may be anomalous.

## Anomaly Detection Use Case

**When to Use an Energy-Based Model:**

- **Unsupervised Learning:**  
  The model is trained solely on normal data and uses an energy score to detect deviations. This is ideal when only normal samples are available.

- **Flexible Objective:**  
  By adjusting the margin, you can control the separation between normal and anomalous samples.

- **Broad Applicability:**  
  Energy-based models have been applied in various domains (e.g., image, audio, sensor data) where anomalies need to be flagged based on reconstruction or deviation from learned patterns.

## Setup

1. **Clone the repository** and navigate to the `energy_based/energy_model/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Download the MNIST dataset.
    * Train the energy-based model to assign low energy to normal images and higher energy to random negative samples.
    * Evaluate the model on the test set.
    * Plot and save a histogram of energy scores, which can be used as anomaly scores.