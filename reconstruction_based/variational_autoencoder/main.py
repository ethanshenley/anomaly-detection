"""
Variational Autoencoder (VAE) Demo for Anomaly Detection

This script trains a variational autoencoder on the MNIST dataset.
The VAE learns a probabilistic latent space of the data, and its reconstruction error
combined with the KL divergence can be used to flag anomalous inputs.
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
# Define the VAE Model
# ====================
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # Mean vector
        self.fc22 = nn.Linear(400, latent_dim)  # Log variance vector
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28 * 28)
    
    def encode(self, x):
        h1 = torch.relu(self.fc1(x))
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    # KL Divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# ====================
# Training Function
# ====================
def train(model, dataloader, optimizer, device):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        data = data.view(-1, 28 * 28)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item()/len(data):.4f}")
    avg_loss = train_loss / len(dataloader.dataset)
    return avg_loss

# ====================
# Testing and Visualization Function
# ====================
def test_and_visualize(model, dataloader, device, output_dir):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            data = data.view(-1, 28 * 28)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            break  # Use one batch for visualization
    test_loss /= len(data)
    print(f"Test Loss per image: {test_loss:.4f}")
    
    # Visualize reconstructions
    data = data.view(-1, 1, 28, 28).cpu().numpy()
    recon = recon_batch.view(-1, 1, 28, 28).cpu().numpy()
    
    n = 8  # Number of images to display
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        # Original image
        axes[0, i].imshow(np.squeeze(data[i]), cmap='gray')
        axes[0, i].axis('off')
        # Reconstructed image
        axes[1, i].imshow(np.squeeze(recon[i]), cmap='gray')
        axes[1, i].axis('off')
    
    plt.suptitle("Top: Original | Bottom: Reconstructed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "vae_reconstruction.png")
    plt.savefig(output_path)
    print(f"Reconstruction plot saved to {output_path}")
    plt.show()

# ====================
# Main Function
# ====================
def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    model = VAE(latent_dim=config["latent_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    for epoch in range(1, config["num_epochs"] + 1):
        avg_loss = train(model, train_loader, optimizer, device)
        print(f"Epoch [{epoch}/{config['num_epochs']}]: Average Loss per image = {avg_loss:.4f}")
    
    test_and_visualize(model, test_loader, device, config["output_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Variational Autoencoder Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)