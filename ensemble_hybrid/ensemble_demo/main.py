"""
Ensemble Demo for Anomaly Detection

This demo trains two separate models on synthetic time series data:
1. An Autoencoder that reconstructs the input sequence.
2. An LSTM forecasting model that predicts the next time step.

During testing, the anomaly score for each sample is computed as a weighted combination of:
- The reconstruction error (from the autoencoder)
- The forecasting error (from the LSTM model)

High ensemble anomaly scores indicate potential anomalies.

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
from torchvision import transforms

# -------------------------------
# Data Generation and Dataset
# -------------------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, series, seq_len):
        """
        series: 1D numpy array of time series data.
        seq_len: Length of the input sequence.
        """
        self.series = series
        self.seq_len = seq_len

    def __len__(self):
        return len(self.series) - self.seq_len

    def __getitem__(self, idx):
        x = self.series[idx: idx + self.seq_len]
        y = self.series[idx + self.seq_len]
        # x is used for reconstruction; y is used for forecasting.
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
# Model 1: Autoencoder for Reconstruction
# -------------------------------
class Autoencoder(nn.Module):
    def __init__(self, seq_len):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, seq_len),
        )
    
    def forward(self, x):
        # x: (batch, seq_len, 1)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)  # shape: (batch, seq_len)
        latent = self.encoder(x_flat)
        recon = self.decoder(latent)
        recon = recon.view(batch_size, -1, 1)
        return recon

# -------------------------------
# Model 2: LSTM Forecasting
# -------------------------------
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Use last time step
        pred = self.fc(out)
        return pred.squeeze(-1)

# -------------------------------
# Training Functions
# -------------------------------
def train_autoencoder(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, _ in dataloader:
        x = x.to(device)
        optimizer.zero_grad()
        recon = model(x)
        loss = criterion(recon, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def train_lstm(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    seq_len = config["seq_len"]

    # Generate training and test series
    train_series = generate_sine_series(config["n_train"], noise_std=config["noise_std"])
    test_series = generate_sine_series(config["n_test"], noise_std=config["noise_std"],
                                       anomaly_indices=np.arange(int(config["n_test"]*0.5), config["n_test"], config["anomaly_interval"]),
                                       anomaly_magnitude=config["anomaly_magnitude"])
    
    train_dataset = TimeSeriesDataset(train_series, seq_len)
    test_dataset = TimeSeriesDataset(test_series, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize models
    autoencoder = Autoencoder(seq_len).to(device)
    lstm_model = LSTMForecast(input_size=1, hidden_size=config["lstm_hidden"], num_layers=config["lstm_layers"]).to(device)

    criterion_recon = nn.MSELoss()
    criterion_forecast = nn.MSELoss()

    optimizer_ae = optim.Adam(autoencoder.parameters(), lr=config["learning_rate"])
    optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=config["learning_rate"])

    # Train autoencoder and LSTM separately
    print("Training Autoencoder...")
    for epoch in range(1, config["num_epochs"]+1):
        loss_ae = train_autoencoder(autoencoder, train_loader, criterion_recon, optimizer_ae, device)
        if epoch % config["print_every"] == 0:
            print(f"Autoencoder Epoch {epoch}/{config['num_epochs']}, Loss: {loss_ae:.4f}")
    
    print("Training LSTM Forecasting Model...")
    for epoch in range(1, config["num_epochs"]+1):
        loss_lstm = train_lstm(lstm_model, train_loader, criterion_forecast, optimizer_lstm, device)
        if epoch % config["print_every"] == 0:
            print(f"LSTM Epoch {epoch}/{config['num_epochs']}, Loss: {loss_lstm:.4f}")

    # Evaluation on test set: compute anomaly scores for each sample
    autoencoder.eval()
    lstm_model.eval()
    ae_errors = []
    lstm_errors = []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            # Autoencoder reconstruction error (average MSE over sequence)
            recon = autoencoder(x)
            err_ae = torch.mean((recon - x)**2, dim=[1,2]).cpu().numpy()
            ae_errors.extend(err_ae)
            # LSTM forecasting error
            pred = lstm_model(x)
            err_lstm = torch.abs(pred - y.to(device)).cpu().numpy()
            lstm_errors.extend(err_lstm)
    
    ae_errors = np.array(ae_errors)
    lstm_errors = np.array(lstm_errors)

    # Ensemble anomaly score: weighted sum
    alpha = config["alpha"]
    ensemble_score = alpha * ae_errors + (1 - alpha) * lstm_errors

    # Plot ensemble anomaly scores
    plt.figure(figsize=(10,4))
    plt.plot(ensemble_score, label="Ensemble Anomaly Score")
    plt.xlabel("Sample Index")
    plt.ylabel("Anomaly Score")
    plt.title("Ensemble Hybrid Anomaly Score")
    plt.legend()
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], "ensemble_anomaly_score.png")
    plt.savefig(output_path)
    print("Ensemble anomaly score plot saved to", output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Hybrid Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)
