#!/usr/bin/env python
"""
Autoencoder Demo for Anomaly Detection

This script trains a simple autoencoder on the MNIST dataset.
After training, it visualizes a few original and reconstructed images.
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
# Define the Autoencoder Model
# ====================
class Autoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        # Encoder: 28x28 input images flattened to 784, then reduce to latent_dim
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(True),
            nn.Linear(128, latent_dim),
        )
        # Decoder: Map latent_dim back to 784 and reshape to 28x28 image
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()  # Outputs between 0 and 1
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        reconstructed = reconstructed.view(x.size())
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
        output = model(data)
        loss = criterion(output, data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    avg_loss = running_loss / len(dataloader)
    return avg_loss

# ====================
# Testing / Visualization Function
# ====================
def test_and_visualize(model, dataloader, device, output_dir):
    model.eval()
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
    
    # Move data to CPU and convert to numpy arrays for plotting
    images = images.cpu().numpy()
    outputs = outputs.cpu().numpy()
    
    n = 8  # Number of images to display
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        # Original images on top row
        axes[0, i].imshow(np.squeeze(images[i]), cmap='gray')
        axes[0, i].axis('off')
        # Reconstructed images on bottom row
        axes[1, i].imshow(np.squeeze(outputs[i]), cmap='gray')
        axes[1, i].axis('off')
    
    plt.suptitle("Top: Original Images | Bottom: Reconstructed Images")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "reconstruction.png")
    plt.savefig(output_path)
    print(f"Reconstruction plot saved to {output_path}")
    plt.show()

# ====================
# Main Function
# ====================
def main(config):
    # Set device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transforms and loading MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    # Instantiate model, loss function, and optimizer
    model = Autoencoder(latent_dim=config["latent_dim"]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Training loop
    for epoch in range(1, config["num_epochs"] + 1):
        avg_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch}/{config['num_epochs']}], Loss: {avg_loss:.4f}")
    
    # After training, visualize some reconstructions
    test_and_visualize(model, test_loader, device, config["output_dir"])

# ====================
# Entry Point
# ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autoencoder Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    args = parser.parse_args()
    
    # Load configuration from YAML file
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)
