# Time Series Transformer for Anomaly Detection

This demo implements a **Transformer-based forecasting model** using PyTorch on synthetic time series data.
A clean sine wave is used for training, while the test set includes injected anomalies (sudden spikes).
The model is trained to predict the next time step given a fixed-length input sequence.
At test time, the absolute prediction error is used as an anomaly score—large errors suggest anomalous behavior.

## Anomaly Detection Use Case

**When to Use a Time Series Transformer:**

- **Forecasting and Anomaly Detection:**  
  Transformer models can capture long-range dependencies in sequential data. By forecasting future values and measuring prediction errors, anomalies can be flagged when errors exceed normal bounds.

- **Handling Complex Temporal Patterns:**  
  Transformers are well-suited for time series with complex dynamics, making them useful in applications like sensor monitoring, financial forecasting, and predictive maintenance.

- **Unsupervised or Semi-supervised Settings:**  
  The model is trained on normal data only, so anomalies (which deviate from the learned pattern) yield higher errors.

## Setup

1. **Clone the repository** and navigate to the `transformers/time_series_transformer/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Generate a sine wave for training and a test series with injected anomalies.
    * Train a Transformer-based model to forecast the next value.
    * Evaluate the model on the test set and plot the prediction error, which serves as the anomaly score.
    * Save the error plot to the outputs/ directory.