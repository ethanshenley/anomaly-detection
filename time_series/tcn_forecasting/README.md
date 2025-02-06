# TCN Forecasting for Anomaly Detection

This demo implements a Temporal Convolutional Network (TCN) for forecasting using PyTorch on synthetic time series data.
A clean sine wave is used for training, while the test series includes injected anomalies.
The TCN, built with dilated causal convolutions, forecasts the next time step from a fixed-length input.
Anomaly detection is performed by measuring the absolute prediction error: high errors indicate anomalous behavior.

## Anomaly Detection Use Case

**When to Use TCN Forecasting:**

- **Capturing Long-Range Dependencies:**  
  TCNs use dilated convolutions to capture long-range temporal dependencies efficiently.

- **Robust Forecasting:**  
  When trained on normal data, the model produces low forecast errors for normal behavior. Anomalies yield high errors that can be flagged.

- **Applications:**  
  Useful in sensor data monitoring, finance, and predictive maintenance where time series behavior deviates from normal patterns.

## Setup

1. **Clone the repository** and navigate to the `time_series/tcn_forecasting/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # (On Windows: venv\Scripts\activate)

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Generate a clean sine wave for training and a test series with injected anomalies.
    * Train the TCN forecasting model.
    * Evaluate the model on the test set and plot the prediction error as an anomaly score.
    * Save the error plot to the outputs/ directory.