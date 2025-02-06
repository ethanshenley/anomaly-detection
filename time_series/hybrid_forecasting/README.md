# Hybrid Forecasting for Anomaly Detection

This demo implements a hybrid forecasting model that combines features from both an LSTM branch and a TCN branch using PyTorch. The model is trained on synthetic time series data generated from a sine wave (normal behavior) and evaluated on a test series with injected anomalies. The LSTM branch and the TCN branch process the input sequence independently; their outputs are concatenated and passed through a fully connected layer to predict the next time step. The absolute prediction error serves as an anomaly score.

## Anomaly Detection Use Case

**When to Use Hybrid Forecasting:**

- **Fusing Complementary Features:**  
  By combining the strengths of LSTMs (which capture long-term dependencies) and TCNs (which leverage dilated convolutions), the hybrid model can capture a broader range of temporal patterns.

- **Improved Anomaly Detection:**  
  With a more robust forecast, prediction errors that deviate significantly from normal behavior can be used to flag anomalies.

- **Applications:**  
  Suitable for scenarios where time series data exhibits complex temporal dynamics, such as sensor monitoring, finance, and predictive maintenance.

## Setup

1. **Clone the repository** and navigate to the `time_series/hybrid_forecasting/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # (On Windows: venv\Scripts\activate)

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Generate a clean sine wave for training and a test series with injected anomalies.
    * Train the hybrid forecasting model.
    * Evaluate the model on the test set and plot the prediction error (used as the anomaly score).
    * Save the error plot to the outputs/ directory.