#!/usr/bin/env python
"""
Convolutional Autoencoder Demo for Anomaly Detection

This script trains a convolutional autoencoder on the MNIST dataset.
The model learns to compress and reconstruct images. In an anomaly detection
scenario, images that are poorly reconstructed (i.e., high reconstruction error)
may be flagged as anomalous.

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

# ====================
# Define the Convolutional Autoencoder Model
# ====================
class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            # Input: (batch, 1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),  # -> (batch, 16, 14, 14)
            nn.ReLU(True),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1), # -> (batch, 32, 7, 7)
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (batch, 16, 14, 14)
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (batch, 1, 28, 28)
            nn.Sigmoid()  # Ensures output is between 0 and 1
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

# ====================
# Training Function
# ====================
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    avg_loss = running_loss / len(dataloader)
    return avg_loss

# ====================
# Testing and Visualization Function
# ====================
def test_and_visualize(model, dataloader, device, output_dir):
    model.eval()
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
    
    images = images.cpu().numpy()
    outputs = outputs.cpu().numpy()
    
    n = 8  # Number of images to display
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        # Original images on the top row
        axes[0, i].imshow(np.squeeze(images[i]), cmap='gray')
        axes[0, i].axis('off')
        # Reconstructed images on the bottom row
        axes[1, i].imshow(np.squeeze(outputs[i]), cmap='gray')
        axes[1, i].axis('off')
    
    plt.suptitle("Top: Original Images | Bottom: Reconstructed Images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "conv_reconstruction.png")
    plt.savefig(output_path)
    print(f"Reconstruction plot saved to {output_path}")
    plt.show()

# ====================
# Main Function
# ====================
def main(config):
    # Set device (use CUDA if available and configured)
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transforms and loading MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Instantiate model, loss function, and optimizer
    model = ConvAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Training loop
    for epoch in range(1, config["num_epochs"] + 1):
        avg_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch}/{config['num_epochs']}]: Loss = {avg_loss:.4f}")
    
    # Visualize reconstruction on test data
    test_and_visualize(model, test_loader, device, config["output_dir"])

# ====================
# Entry Point
# ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convolutional Autoencoder Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    args = parser.parse_args()
    
    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)
