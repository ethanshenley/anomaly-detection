#!/usr/bin/env python
"""
Denoising Autoencoder Demo for Anomaly Detection

This script trains a convolutional denoising autoencoder on the MNIST dataset.
It adds random noise to input images and trains the model to reconstruct the original images.
In anomaly detection, denoising autoencoders can help filter out noise and highlight anomalies by
learning to recover the underlying clean signal.

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
# Define the Denoising Autoencoder Model
# ====================
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),  
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),  
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def add_noise(imgs, noise_factor):
    """Adds Gaussian noise to a batch of images and clamps the results to [0, 1]."""
    noisy_imgs = imgs + noise_factor * torch.randn_like(imgs)
    noisy_imgs = torch.clamp(noisy_imgs, 0., 1.)
    return noisy_imgs

# ====================
# Training Function
# ====================
def train(model, dataloader, criterion, optimizer, device, noise_factor):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        # Add noise to the input images
        noisy_data = add_noise(data, noise_factor)
        optimizer.zero_grad()
        outputs = model(noisy_data)
        loss = criterion(outputs, data)  # Compare reconstruction to original clean images
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
def test_and_visualize(model, dataloader, device, noise_factor, output_dir):
    model.eval()
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images.to(device)
    noisy_images = add_noise(images, noise_factor)
    
    with torch.no_grad():
        outputs = model(noisy_images)
    
    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()
    
    n = 8  # Number of images to display
    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 4))
    for i in range(n):
        # Row 1: Original images
        axes[0, i].imshow(np.squeeze(images[i]), cmap='gray')
        axes[0, i].axis('off')
        # Row 2: Noisy images
        axes[1, i].imshow(np.squeeze(noisy_images[i]), cmap='gray')
        axes[1, i].axis('off')
        # Row 3: Denoised (reconstructed) images
        axes[2, i].imshow(np.squeeze(outputs[i]), cmap='gray')
        axes[2, i].axis('off')
    
    plt.suptitle("Row 1: Original | Row 2: Noisy | Row 3: Denoised")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "denoising_reconstruction.png")
    plt.savefig(output_path)
    print(f"Reconstruction plot saved to {output_path}")
    plt.show()

# ====================
# Main Function
# ====================
def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    model = DenoisingAutoencoder().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    noise_factor = config.get("noise_factor", 0.5)
    
    for epoch in range(1, config["num_epochs"] + 1):
        avg_loss = train(model, train_loader, criterion, optimizer, device, noise_factor)
        print(f"Epoch [{epoch}/{config['num_epochs']}]: Loss = {avg_loss:.4f}")
    
    test_and_visualize(model, test_loader, device, noise_factor, config["output_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Denoising Autoencoder Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)
