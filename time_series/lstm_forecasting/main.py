"""
LSTM Forecasting Demo for Anomaly Detection

This demo trains an LSTM-based forecasting model on synthetic time series data.
The training data is generated from a sine wave (normal behavior), while the test data includes injected anomalies.
The model is trained to predict the next time step given a fixed-length sequence.
Anomaly detection is performed by measuring the prediction error: high errors indicate potential anomalies.

Usage:
    python main.py --config config.yaml
"""

import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------------
# Dataset and Data Generation
# -------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len):
        """
        series: 1D numpy array of time series data.
        seq_len: Length of input sequence for forecasting.
        """
        self.series = series
        self.seq_len = seq_len

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.seq_len]
        y = self.series[idx + self.seq_len]
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

def generate_sine_series(n_points, noise_std=0.1, anomaly_indices=None, anomaly_magnitude=3.0):
    """
    Generates a sine wave with optional anomalies.
    anomaly_indices: List/array of indices where anomalies are injected.
    anomaly_magnitude: Amplitude added at anomaly points.
    """
    t = np.linspace(0, 4 * np.pi, n_points)
    series = np.sin(t)
    series += np.random.normal(0, noise_std, size=n_points)
    if anomaly_indices is not None:
        for idx in anomaly_indices:
            if 0 <= idx < n_points:
                series[idx] += anomaly_magnitude
    return series

# -------------------------------
# LSTM Forecasting Model
# -------------------------------
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)  # out: (batch, seq_len, hidden_size)
        out = out[:, -1, :]    # Use the last time step's hidden state
        out = self.fc(out)     # (batch, 1)
        return out.squeeze(-1)

# -------------------------------
# Training and Evaluation
# -------------------------------
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    predictions = []
    targets = []
    losses = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            losses.append(loss.item())
            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    return np.mean(losses), np.concatenate(predictions), np.concatenate(targets)

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    seq_len = config["seq_len"]

    # Generate training data (clean sine wave)
    train_series = generate_sine_series(config["n_train"], noise_std=config["noise_std"])
    train_dataset = TimeSeriesDataset(train_series, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # Generate test data (with anomalies injected)
    anomaly_indices = np.arange(int(config["n_test"] * 0.5), config["n_test"], config["anomaly_interval"])
    test_series = generate_sine_series(config["n_test"], noise_std=config["noise_std"],
                                       anomaly_indices=anomaly_indices, anomaly_magnitude=config["anomaly_magnitude"])
    test_dataset = TimeSeriesDataset(test_series, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize model
    model = LSTMForecast(input_size=1, hidden_size=config["hidden_size"], num_layers=config["num_layers"]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    # Training loop
    for epoch in range(1, config["num_epochs"] + 1):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        if epoch % config["print_every"] == 0:
            print(f"Epoch {epoch}/{config['num_epochs']}, Training Loss: {train_loss:.4f}")

    # Evaluate on test set
    test_loss, predictions, targets = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    # Compute absolute prediction error as anomaly score
    error = np.abs(predictions - targets)
    plt.figure(figsize=(10, 4))
    plt.plot(error, label="Prediction Error")
    plt.xlabel("Time Step")
    plt.ylabel("Absolute Error")
    plt.title("LSTM Forecasting Error (Anomaly Score)")
    plt.legend()
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], "lstm_error.png")
    plt.savefig(output_path)
    print("Error plot saved to", output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM Forecasting Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)
