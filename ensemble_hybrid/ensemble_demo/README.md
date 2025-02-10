# Ensemble Hybrid Anomaly Detection Demo

This demo implements an ensemble of two anomaly detection models on synthetic time series data:
1. A Reconstruction Autoencoder that learns to reconstruct the input sequence.
2. An LSTM Forecasting Model that predicts the next time step.

During testing, each sample’s anomaly score is computed as:
- The reconstruction error from the autoencoder (average MSE over the input sequence).
- The forecasting error from the LSTM (absolute error in the next value prediction).

These two scores are then combined via a weighted sum (with weight α) to produce a final ensemble anomaly score.
High ensemble scores indicate potential anomalies.

## Anomaly Detection Use Case

**When to Use an Ensemble Hybrid Approach:**

- **Robustness:**  
  By combining the strengths of multiple models, the ensemble approach can yield more robust and reliable anomaly detection.
  
- **Complementary Methods:**  
  Reconstruction-based methods capture global patterns, while forecasting models capture temporal dependencies. Their fusion helps flag anomalies that may be missed by any single model.

- **Unsupervised Setting:**  
  Both models are trained on normal data, and anomalies are detected based on deviations from normal reconstruction and prediction behaviors.

## Setup

1. **Clone the repository** and navigate to the `ensemble_hybrid/ensemble_demo/` directory.
2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # (On Windows: venv\Scripts\activate)

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Generate synthetic time series data.
    * Train the autoencoder and LSTM forecasting models separately.
    * Compute individual anomaly scores for reconstruction and forecasting.
    * Combine the scores to produce a final ensemble anomaly score.
    * Plot and save the anomaly scores.