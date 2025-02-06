"""
Hybrid Forecasting Demo for Anomaly Detection

This demo trains a hybrid forecasting model that fuses features from an LSTM branch and a TCN branch
on synthetic time series data. The training series is generated from a sine wave (normal behavior),
while anomalies are injected into the test series. The model predicts the next time step from a fixed-length input,
and the prediction error is used as an anomaly score.

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
# Dataset and Data Generation (same as before)
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
        return torch.tensor(x, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

def generate_sine_series(n_points, noise_std=0.1, anomaly_indices=None, anomaly_magnitude=3.0):
    t = np.linspace(0, 4 * np.pi, n_points)
    series = np.sin(t)
    series += np.random.normal(0, noise_std, size=n_points)
    if anomaly_indices is not None:
        for idx in anomaly_indices:
            if 0 <= idx < n_points:
                series[idx] += anomaly_magnitude
    return series

# -------------------------------
# LSTM Branch
# -------------------------------
class LSTMBranch(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMBranch, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last time step
        out = self.fc(out)
        return out

# -------------------------------
# TCN Branch (simplified)
# -------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)
        out = self.dropout(out)
        # Remove extra padding at the end
        return out[:, :, :-self.conv.padding[0]]

class TCNBranch(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, seq_len):
        super(TCNBranch, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_channels[-1])
    
    def forward(self, x):
        # x: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # Use the last time step's output
        out = out[:, :, -1]
        out = self.fc(out)
        return out

# -------------------------------
# Hybrid Model: Fuse LSTM and TCN Branches
# -------------------------------
class HybridForecast(nn.Module):
    def __init__(self, input_size, lstm_hidden, lstm_layers, tcn_channels, kernel_size, dropout, seq_len):
        super(HybridForecast, self).__init__()
        self.lstm_branch = LSTMBranch(input_size, lstm_hidden, lstm_layers)
        self.tcn_branch = TCNBranch(input_size, tcn_channels, kernel_size, dropout, seq_len)
        fusion_dim = lstm_hidden + tcn_channels[-1]
        self.fc = nn.Linear(fusion_dim, 1)
    
    def forward(self, x):
        lstm_feat = self.lstm_branch(x)
        tcn_feat = self.tcn_branch(x)
        fused = torch.cat([lstm_feat, tcn_feat], dim=1)
        out = self.fc(fused)
        return out.squeeze(-1)

# -------------------------------
# Training and Evaluation (same as before)
# -------------------------------
def train_model(model, dataloader, criterion, optimizer, device):
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

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    preds = []
    targets = []
    losses = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            losses.append(loss.item())
            preds.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())
    return np.mean(losses), np.concatenate(preds), np.concatenate(targets)

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    seq_len = config["seq_len"]

    # Generate training and test data
    train_series = generate_sine_series(config["n_train"], noise_std=config["noise_std"])
    train_dataset = TimeSeriesDataset(train_series, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    anomaly_indices = np.arange(int(config["n_test"] * 0.5), config["n_test"], config["anomaly_interval"])
    test_series = generate_sine_series(config["n_test"], noise_std=config["noise_std"],
                                       anomaly_indices=anomaly_indices, anomaly_magnitude=config["anomaly_magnitude"])
    test_dataset = TimeSeriesDataset(test_series, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize hybrid model
    model = HybridForecast(
        input_size=1,
        lstm_hidden=config["lstm_hidden"],
        lstm_layers=config["lstm_layers"],
        tcn_channels=config["tcn_channels"],
        kernel_size=config["kernel_size"],
        dropout=config["dropout"],
        seq_len=seq_len
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    for epoch in range(1, config["num_epochs"] + 1):
        loss = train_model(model, train_loader, criterion, optimizer, device)
        if epoch % config["print_every"] == 0:
            print(f"Epoch {epoch}/{config['num_epochs']}, Training Loss: {loss:.4f}")

    test_loss, predictions, targets = evaluate_model(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    error = np.abs(predictions - targets)
    plt.figure(figsize=(10, 4))
    plt.plot(error, label="Prediction Error")
    plt.xlabel("Time Step")
    plt.ylabel("Absolute Error")
    plt.title("Hybrid Forecasting Error (Anomaly Score)")
    plt.legend()
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], "hybrid_error.png")
    plt.savefig(output_path)
    print("Error plot saved to", output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid Forecasting Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)