# Hybrid Model for Anomaly Detection

This demo implements a unified hybrid model for anomaly detection on synthetic time series data. The model consists of:
- A shared encoder (an LSTM) that processes the input sequence.
- Two decoders:
  1. A Reconstruction Decoder that attempts to reconstruct the entire input sequence.
  2. A Forecasting Decoder that predicts the next time step.

The model is trained with a combined loss (reconstruction loss plus forecasting loss). During testing, the anomaly score is computed as a weighted sum of:
- The reconstruction error (mean squared error over the input sequence)
- The forecasting error (absolute error for the next time step)

High combined scores indicate potential anomalies.

## Anomaly Detection Use Case

**When to Use a Hybrid Model:**

- **Joint Learning:**  
  By jointly learning to reconstruct and forecast, the model captures complementary aspects of normal behavior.
  
- **Improved Anomaly Sensitivity:**  
  Anomalies that disrupt either the pattern of the input sequence or the future behavior will yield high reconstruction and/or forecasting errors.

- **Applications:**  
  Suitable for time series data where both the past sequence and future prediction are important, such as sensor monitoring, predictive maintenance, or financial data analysis.

## Setup

1. **Clone the repository** and navigate to the `ensemble_hybrid/hybrid_model_demo/` directory.
2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # (On Windows: venv\Scripts\activate)

3. Adjust hyperparameters in `config.yaml` as needed.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

* Generate synthetic time series data (a sine wave with injected anomalies).
* Train the hybrid model with a shared encoder and two decoders.
* Evaluate the model on the test set and compute a combined anomaly score.
* Plot and save the anomaly scores to the outputs/ directory.
