"""
Hybrid Model Demo for Anomaly Detection

This demo trains a joint hybrid model on synthetic time series data. The model consists of:
- A shared encoder (an LSTM) that processes the input sequence.
- Two decoders:
  1. A Reconstruction Decoder that attempts to reconstruct the input sequence.
  2. A Forecasting Decoder that predicts the next time step.
  
The training loss is a combination of reconstruction loss and forecasting loss.
At test time, the anomaly score is computed as a weighted sum of the reconstruction error and forecasting error.

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
# Data Generation and Dataset (reuse from ensemble_demo)
# -------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len):
        self.series = series
        self.seq_len = seq_len

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.seq_len]
        y = self.series[idx + self.seq_len]
        return (torch.tensor(x, dtype=torch.float32).unsqueeze(-1),
                torch.tensor(y, dtype=torch.float32))

def generate_sine_series(n_points, noise_std=0.1, anomaly_indices=None, anomaly_magnitude=3.0):
    t = np.linspace(0, 4*np.pi, n_points)
    series = np.sin(t)
    series += np.random.normal(0, noise_std, size=n_points)
    if anomaly_indices is not None:
        for idx in anomaly_indices:
            if 0 <= idx < n_points:
                series[idx] += anomaly_magnitude
    return series

# -------------------------------
# Hybrid Model with Shared Encoder and Two Decoders
# -------------------------------
class HybridAnomalyModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len):
        super(HybridAnomalyModel, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Reconstruction decoder: reconstruct entire sequence
        self.recon_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, seq_len)
        )
        # Forecasting decoder: predict next time step
        self.forecast_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)
        out, (hn, cn) = self.encoder(x)
        # Use the last hidden state from the encoder
        latent = hn[-1]  # shape: (batch, hidden_size)
        recon = self.recon_decoder(latent)  # (batch, seq_len)
        forecast = self.forecast_decoder(latent)  # (batch, 1)
        # Reshape reconstruction to (batch, seq_len, 1)
        recon = recon.view(batch_size, -1, 1)
        return recon, forecast.squeeze(-1)

# -------------------------------
# Training and Evaluation
# -------------------------------
def train_model(model, dataloader, criterion_recon, criterion_forecast, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        recon, forecast = model(x)
        loss_recon = criterion_recon(recon, x)
        loss_forecast = criterion_forecast(forecast, y)
        loss = loss_recon + loss_forecast
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, criterion_recon, criterion_forecast, device):
    model.eval()
    recon_errors = []
    forecast_errors = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            recon, forecast = model(x)
            err_recon = torch.mean((recon - x)**2, dim=[1,2]).cpu().numpy()
            err_forecast = torch.abs(forecast - y).cpu().numpy()
            recon_errors.extend(err_recon)
            forecast_errors.extend(err_forecast)
    return np.array(recon_errors), np.array(forecast_errors)

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    seq_len = config["seq_len"]

    # Generate training and test series
    train_series = generate_sine_series(config["n_train"], noise_std=config["noise_std"])
    anomaly_indices = np.arange(int(config["n_test"]*0.5), config["n_test"], config["anomaly_interval"])
    test_series = generate_sine_series(config["n_test"], noise_std=config["noise_std"],
                                       anomaly_indices=anomaly_indices, anomaly_magnitude=config["anomaly_magnitude"])
    
    train_dataset = TimeSeriesDataset(train_series, seq_len)
    test_dataset = TimeSeriesDataset(test_series, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    model = HybridAnomalyModel(input_size=1, hidden_size=config["hidden_size"],
                               num_layers=config["num_layers"], seq_len=seq_len).to(device)
    criterion_recon = nn.MSELoss()
    criterion_forecast = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(1, config["num_epochs"]+1):
        loss = train_model(model, train_loader, criterion_recon, criterion_forecast, optimizer, device)
        if epoch % config["print_every"] == 0:
            print(f"Epoch {epoch}/{config['num_epochs']} | Combined Loss: {loss:.4f}")

    # Evaluate model on test set
    recon_errors, forecast_errors = evaluate_model(model, test_loader, criterion_recon, criterion_forecast, device)
    # Ensemble anomaly score: weighted sum of reconstruction and forecasting errors
    beta = config["beta"]
    ensemble_score = beta * recon_errors + (1 - beta) * forecast_errors

    plt.figure(figsize=(10,4))
    plt.plot(ensemble_score, label="Hybrid Anomaly Score")
    plt.xlabel("Sample Index")
    plt.ylabel("Anomaly Score")
    plt.title("Hybrid Model Anomaly Score")
    plt.legend()
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], "hybrid_anomaly_score.png")
    plt.savefig(output_path)
    print("Hybrid anomaly score plot saved to", output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Model Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)