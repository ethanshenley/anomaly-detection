# LSTM Forecasting for Anomaly Detection

This demo implements an LSTM-based forecasting model using PyTorch on synthetic time series data. The training series is a clean sine wave, while the test series has injected anomalies (spikes). The model learns to predict the next time step from a fixed-length sequence. During testing, the absolute prediction error is used as an anomaly score—large errors indicate potential anomalies.

## Anomaly Detection Use Case

**When to Use LSTM Forecasting:**

- **Time Series Forecasting:**  
  LSTM networks excel at capturing temporal dependencies, making them ideal for forecasting future values.

- **Anomaly Detection via Prediction Error:**  
  When trained on normal data, the model produces low prediction errors for normal samples. Anomalies (unexpected spikes or drops) yield high errors, which can be flagged as anomalies.

- **Applications:**  
  Suitable for sensor monitoring, financial forecasting, and predictive maintenance where deviations from normal behavior are critical.

## Setup

1. **Clone the repository** and navigate to the `time_series/lstm_forecasting/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # (On Windows: venv\Scripts\activate)

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Generate a clean sine wave for training and a test series with injected anomalies.
    * Train the LSTM forecasting model.
    * Evaluate the model on the test set and plot the prediction error as an anomaly score.
    * Save the error plot to the outputs/ directory.