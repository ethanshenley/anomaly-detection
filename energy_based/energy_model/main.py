"""
Energy-Based Model Demo for Anomaly Detection

This demo trains an energy-based model on the MNIST dataset using PyTorch.
The model is a simple MLP that outputs a scalar “energy” for each input image.
During training, we generate negative samples (random noise) and use a margin-ranking loss
to enforce that normal samples receive lower energy than negatives. At test time,
the energy value serves as an anomaly score—higher energy indicates potential anomalies.

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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define the Energy-based Model
class EnergyNet(nn.Module):
    def __init__(self):
        super(EnergyNet, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Output a single scalar energy
        )
    
    def forward(self, x):
        return self.model(x)

def generate_negative_samples(x, device):
    # Generate negative samples as random images in [0,1]
    return torch.rand_like(x).to(device)

def train(model, dataloader, optimizer, device, config):
    margin = config["margin"]
    margin_loss_fn = nn.MarginRankingLoss(margin=margin)
    for epoch in range(1, config["num_epochs"] + 1):
        running_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            batch_size = x.size(0)
            # Get energy for normal samples
            energy_normal = model(x).squeeze()  # shape: (batch,)
            # Generate negative samples (random noise)
            x_neg = generate_negative_samples(x, device)
            energy_negative = model(x_neg).squeeze()
            # Target for MarginRankingLoss: we want energy_negative > energy_normal by at least margin
            target = torch.ones(batch_size, device=device)
            loss_margin = margin_loss_fn(energy_negative, energy_normal, target)
            # Also encourage low energy on normal samples
            loss_normal = torch.mean(energy_normal)
            loss = loss_normal + loss_margin
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{config['num_epochs']} | Loss: {avg_loss:.4f}")
    return model

def test(model, dataloader, device):
    model.eval()
    energies = []
    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)
            energy = model(x).squeeze().cpu().numpy()
            energies.extend(energy)
    return np.array(energies)

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    model = EnergyNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    print("Training Energy-Based Model...")
    model = train(model, train_loader, optimizer, device, config)
    
    print("Evaluating on test set...")
    energies = test(model, test_loader, device)
    
    # Plot histogram of energies as anomaly scores
    plt.figure(figsize=(8, 6))
    plt.hist(energies, bins=50, alpha=0.7)
    plt.xlabel("Energy")
    plt.ylabel("Frequency")
    plt.title("Histogram of Energy Scores on Test Data")
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], "energy_histogram.png")
    plt.savefig(output_path)
    print("Histogram saved to", output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Energy-Based Model Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)
