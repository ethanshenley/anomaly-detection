"""
Memory Augmented Autoencoder Demo for Anomaly Detection

This demo implements a simple memory-augmented autoencoder using PyTorch on the MNIST dataset.
The model consists of an encoder, a learnable memory module, and a decoder.
The encoder maps input images to a latent representation. The memory module contains a set of learnable
memory vectors that are used to refine the latent representation via a weighted combination (based on similarity).
The decoder reconstructs the image from the refined latent vector. The network is trained using a reconstruction
loss (MSE) along with an auxiliary loss that encourages consistency between the original and refined latent features.
At test time, the reconstruction error serves as an anomaly score.

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

# Define the Memory Augmented Autoencoder
class MemoryAutoencoder(nn.Module):
    def __init__(self, latent_dim, memory_size):
        super(MemoryAutoencoder, self).__init__()
        self.latent_dim = latent_dim
        # Encoder: maps image to latent vector
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(True),
            nn.Linear(256, latent_dim)
        )
        # Memory module: learnable memory bank of size (memory_size, latent_dim)
        self.memory = nn.Parameter(torch.randn(memory_size, latent_dim))
        # Decoder: reconstruct image from refined latent vector
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(True),
            nn.Linear(256, 28 * 28),
            nn.Sigmoid()  # Output in [0,1]
        )
    
    def forward(self, x):
        z = self.encoder(x)  # shape: (batch, latent_dim)
        # Compute similarity between z and each memory vector (using dot product)
        sim = torch.matmul(z, self.memory.t())  # shape: (batch, memory_size)
        weights = torch.softmax(sim, dim=1)      # shape: (batch, memory_size)
        # Refine latent vector as weighted sum of memory items
        z_refined = torch.matmul(weights, self.memory)  # shape: (batch, latent_dim)
        recon = self.decoder(z_refined)
        recon = recon.view(-1, 1, 28, 28)
        return recon, z, z_refined

def train(model, dataloader, optimizer, device, config):
    mse_loss = nn.MSELoss()
    lambda_memory = config["lambda_memory"]
    for epoch in range(1, config["num_epochs"] + 1):
        running_loss = 0.0
        for x, _ in dataloader:
            x = x.to(device)
            optimizer.zero_grad()
            recon, z, z_refined = model(x)
            loss_recon = mse_loss(recon, x)
            loss_memory = mse_loss(z, z_refined)
            loss = loss_recon + lambda_memory * loss_memory
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{config['num_epochs']} | Loss: {avg_loss:.4f}")
    return model

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    model = MemoryAutoencoder(latent_dim=config["latent_dim"], memory_size=config["memory_size"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    print("Training Memory Augmented Autoencoder...")
    model = train(model, train_loader, optimizer, device, config)
    
    # Evaluate on a test image
    test_img, _ = test_dataset[0]
    test_img = test_img.to(device)
    with torch.no_grad():
        recon, _, _ = model(test_img.unsqueeze(0))
    # Convert images for visualization
    test_img_np = test_img.cpu().numpy().squeeze()
    recon_np = recon.cpu().numpy().squeeze()
    
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(test_img_np, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(recon_np, cmap="gray")
    axes[1].set_title("Reconstructed Image")
    axes[1].axis("off")
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], "memory_augmented_reconstruction.png")
    plt.savefig(output_path)
    print("Reconstruction saved to", output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Memory Augmented Autoencoder Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)