"""
Time Series Transformer Demo for Anomaly Detection

This demo trains a Transformer-based forecasting model on synthetic time series data.
The training data is generated from a clean sine wave, while the test data includes injected anomalies.
The model is trained to predict the next time step given a fixed-length sequence.
At test time, high prediction errors (measured by MSE) are used as anomaly scores.

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
# Synthetic Time Series Dataset
# -------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len):
        """
        series: 1D numpy array of time series data.
        seq_len: Length of input sequence (window) to use for forecasting.
        """
        self.series = series
        self.seq_len = seq_len

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx : idx + self.seq_len]
        y = self.series[idx + self.seq_len]
        # Convert to tensor and add feature dimension
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

def generate_sine_series(n_points, noise_std=0.1, anomaly_indices=None, anomaly_magnitude=3.0):
    """
    Generates a sine wave with optional anomalies.
    anomaly_indices: list or array of indices where anomalies are injected.
    anomaly_magnitude: amplitude added to anomalies.
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
# Positional Encoding Module
# -------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)  # shape: (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 1:
            # if odd, handle last column separately
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: shape (batch_size, seq_len, d_model)
        x = x + self.pe[:, : x.size(1)]
        return x

# -------------------------------
# Time Series Transformer Model
# -------------------------------
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, seq_len):
        super(TimeSeriesTransformer, self).__init__()
        self.input_linear = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, max_len=seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(model_dim, 1)  # Predict scalar output

    def forward(self, x):
        # x: shape (batch_size, seq_len, input_dim)
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        # Transformer encoder
        x = self.transformer_encoder(x)
        # Use the representation of the last time step for prediction
        out = self.decoder(x[:, -1, :])
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
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    losses = []
    predictions = []
    targets = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = criterion(output, y)
            losses.append(loss.item())
            predictions.append(output.cpu().numpy())
            targets.append(y.cpu().numpy())
    return np.mean(losses), np.concatenate(predictions), np.concatenate(targets)

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    seq_len = config["seq_len"]

    # Generate training data (clean sine wave)
    n_train = config["n_train"]
    train_series = generate_sine_series(n_train, noise_std=config["noise_std"])
    train_dataset = TimeSeriesDataset(train_series, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # Generate test data (inject anomalies)
    n_test = config["n_test"]
    # Inject anomalies at specified indices (e.g., every 100th point)
    anomaly_indices = np.arange(int(n_test * 0.5), n_test, config["anomaly_interval"])
    test_series = generate_sine_series(n_test, noise_std=config["noise_std"], anomaly_indices=anomaly_indices, anomaly_magnitude=config["anomaly_magnitude"])
    test_dataset = TimeSeriesDataset(test_series, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize model
    model = TimeSeriesTransformer(
        input_dim=1,
        model_dim=config["model_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        seq_len=seq_len
    ).to(device)

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

    # Compute prediction error (absolute error) as anomaly score
    error = np.abs(predictions - targets)
    plt.figure(figsize=(10, 4))
    plt.plot(error, label="Prediction Error")
    plt.xlabel("Time Step")
    plt.ylabel("Absolute Error")
    plt.title("Time Series Prediction Error (Anomaly Score)")
    plt.legend()
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], "time_series_error.png")
    plt.savefig(output_path)
    print("Error plot saved to", output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Time Series Transformer Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)
