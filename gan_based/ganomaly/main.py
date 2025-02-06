"""
Ganomaly Demo for Anomaly Detection

This demo implements a simplified version of **Ganomaly** using PyTorch on the MNIST dataset.
Ganomaly consists of an encoder-decoder-encoder architecture, where the generator reconstructs the input
and a secondary encoder extracts features from the reconstruction. The anomaly score is computed as the difference
between the latent representations of the input and the reconstruction.
High differences indicate that the input deviates from the normal data distribution and may be anomalous.

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

# Generator: Encoder-Decoder structure
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(True),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )
    def forward(self, x):
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        reconstruction = reconstruction.view(-1, 1, 28, 28)
        return latent, reconstruction

# Secondary encoder: encodes the reconstructed image
class Encoder2(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder2, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(True),
            nn.Linear(512, latent_dim)
        )
    def forward(self, x):
        return self.model(x)

def train_ganomaly(generator, encoder2, dataloader, device, config):
    criterion_recon = nn.MSELoss()
    criterion_latent = nn.MSELoss()
    optimizer = optim.Adam(list(generator.parameters()) + list(encoder2.parameters()), lr=config["lr"])
    num_epochs = config["num_epochs"]
    
    for epoch in range(1, num_epochs+1):
        running_loss = 0.0
        for imgs, _ in dataloader:
            imgs = imgs.to(device)
            imgs = (imgs - 0.5) * 2  # scale to [-1,1]
            optimizer.zero_grad()
            latent_real, recon = generator(imgs)
            latent_recon = encoder2(recon)
            loss_recon = criterion_recon(recon, imgs)
            loss_latent = criterion_latent(latent_recon, latent_real)
            loss = loss_recon + loss_latent
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{num_epochs} | Loss: {avg_loss:.4f}")
    return generator, encoder2

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    latent_dim = config["latent_dim"]
    generator = Generator(latent_dim).to(device)
    encoder2 = Encoder2(latent_dim).to(device)
    
    print("Training Ganomaly...")
    generator, encoder2 = train_ganomaly(generator, encoder2, dataloader, device, config)
    
    # Test on a sample image
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_img, label = test_dataset[0]
    test_img = test_img.to(device)
    test_img_scaled = (test_img - 0.5) * 2
    with torch.no_grad():
        latent_real, recon = generator(test_img_scaled.unsqueeze(0))
        latent_recon = encoder2(recon)
    # Anomaly score: L2 distance between latent representations
    anomaly_score = torch.mean((latent_real - latent_recon)**2).item()
    
    # Scale images back for visualization
    test_img_np = test_img.cpu().numpy().squeeze()
    recon_np = recon.cpu().numpy().squeeze()
    recon_np = (recon_np + 1) / 2  # from [-1,1] to [0,1]
    
    fig, axes = plt.subplots(1, 2, figsize=(6,3))
    axes[0].imshow(test_img_np, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(recon_np, cmap="gray")
    axes[1].set_title(f"Reconstructed\nScore: {anomaly_score:.4f}")
    axes[1].axis("off")
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], "ganomaly_reconstruction.png")
    plt.savefig(output_path)
    print("Reconstruction saved to", output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ganomaly Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)