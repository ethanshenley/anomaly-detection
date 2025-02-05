"""
Deep SVDD Demo for Anomaly Detection

This demo trains a neural network to learn a compact representation of normal data by minimizing
the squared distance of the output from a fixed center. Anomalous samples—which do not conform
to the normal data distribution—tend to exhibit higher distances from this center.

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
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class DeepSVDDModel(nn.Module):
    def __init__(self):
        super(DeepSVDDModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.features(x)

def compute_center(model, dataloader, device):
    model.eval()
    center = None
    total = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            output = model(data)
            if center is None:
                center = torch.sum(output, dim=0)
            else:
                center += torch.sum(output, dim=0)
            total += output.size(0)
    center = center / total
    return center

def train(model, dataloader, optimizer, center, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        # Loss: Mean squared distance from the center
        loss = torch.mean(torch.sum((output - center) ** 2, dim=1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def test(model, dataloader, center, device):
    model.eval()
    distances = []
    labels = []
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            output = model(data)
            # Compute squared distance from the center for each sample
            batch_dist = torch.sum((output - center) ** 2, dim=1)
            distances.extend(batch_dist.cpu().numpy())
            labels.extend(label.numpy())
    return np.array(distances), np.array(labels)

def visualize_results(distances, labels, output_dir):
    plt.figure(figsize=(8, 6))
    plt.hist(distances, bins=50, alpha=0.7)
    plt.xlabel("Squared Distance from Center")
    plt.ylabel("Frequency")
    plt.title("Distribution of Deep SVDD Distances on Test Data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "deep_svdd_histogram.png")
    plt.savefig(output_path)
    print(f"Histogram saved to {output_path}")
    plt.show()

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    model = DeepSVDDModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Compute center using an initial pass over the training data
    center = compute_center(model, train_loader, device)
    print("Center computed.")
    
    # Training loop
    for epoch in range(1, config["num_epochs"] + 1):
        avg_loss = train(model, train_loader, optimizer, center, device)
        print(f"Epoch [{epoch}/{config['num_epochs']}]: Average Loss = {avg_loss:.4f}")
    
    distances, labels = test(model, test_loader, center, device)
    visualize_results(distances, labels, config["output_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep SVDD Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)
