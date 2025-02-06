"""
TCN Forecasting Demo for Anomaly Detection

This demo trains a Temporal Convolutional Network (TCN) for forecasting on synthetic time series data.
The training series is generated from a sine wave (normal behavior), while the test series has injected anomalies.
The TCN uses dilated causal convolutions to capture temporal dependencies.
Prediction error is used as an anomaly score.

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
# Dataset (same as before)
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
# TCN Building Blocks
# -------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(TemporalBlock, self).__init__()
        # Calculate padding to maintain sequence length
        self.padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.init_weights()
    
    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight)
    
    def forward(self, x):
        # Save original input for residual
        res = x
        
        # First convolution block
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second convolution block
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Handle residual connection
        if self.downsample is not None:
            res = self.downsample(res)
        
        # Calculate total padding (from both convolutions)
        total_padding = 2 * self.padding  # Both conv1 and conv2 add padding
        
        # Trim the output and residual equally
        out_trim = out[:, :, total_padding:]
        res_trim = res[:, :, :out_trim.size(2)]  # Match the residual to output size
        
        return self.relu2(out_trim + res_trim)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, dilation=dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# -------------------------------
# TCN Forecasting Model
# -------------------------------
class TCNForecast(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, seq_len):
        super(TCNForecast, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.fc = nn.Linear(num_channels[-1], 1)
        self.seq_len = seq_len

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        y = self.tcn(x)        # (batch, num_channels[-1], seq_len) 
        # Use the last time step's output
        y = y[:, :, -1]
        y = self.fc(y)
        return y.squeeze(-1)

# -------------------------------
# Training and Evaluation
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

    # Training data (clean sine wave)
    train_series = generate_sine_series(config["n_train"], noise_std=config["noise_std"])
    train_dataset = TimeSeriesDataset(train_series, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)

    # Test data (with anomalies)
    anomaly_indices = np.arange(int(config["n_test"] * 0.5), config["n_test"], config["anomaly_interval"])
    test_series = generate_sine_series(config["n_test"], noise_std=config["noise_std"],
                                       anomaly_indices=anomaly_indices, anomaly_magnitude=config["anomaly_magnitude"])
    test_dataset = TimeSeriesDataset(test_series, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # Initialize TCN model
    model = TCNForecast(input_size=1, num_channels=config["num_channels"], kernel_size=config["kernel_size"],
                        dropout=config["dropout"], seq_len=seq_len).to(device)
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
    plt.title("TCN Forecasting Error (Anomaly Score)")
    plt.legend()
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], "tcn_error.png")
    plt.savefig(output_path)
    print("Error plot saved to", output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TCN Forecasting Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    main(config)