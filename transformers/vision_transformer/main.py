"""
Vision Transformer Autoencoder Demo for Anomaly Detection

This demo trains a simplified Vision Transformer (ViT) autoencoder on the MNIST dataset.
The model splits each image into patches, embeds them, processes them with transformer encoder layers,
and then reconstructs the image via a linear decoder.
Anomaly detection is achieved by computing the reconstruction error: high error indicates an anomaly.

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
from tqdm import tqdm

# -------------------------------
# Patch Embedding Module
# -------------------------------
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch, in_channels, img_size, img_size)
        x = self.proj(x)  # (batch, embed_dim, num_patches_sqrt, num_patches_sqrt)
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x

# -------------------------------
# Vision Transformer Autoencoder
# -------------------------------
class ViTAutoencoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_layers, num_heads):
        super(ViTAutoencoder, self).__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Decoder: project back to patch pixels
        self.decoder = nn.Linear(embed_dim, patch_size * patch_size * in_channels)

        self.img_size = img_size
        self.patch_size = patch_size

    def forward(self, x):
        # x: (batch, in_channels, img_size, img_size)
        patches = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        patches = patches + self.pos_embedding  # Add positional encoding

        encoded = self.transformer_encoder(patches)  # (batch, num_patches, embed_dim)
        # Decode each patch
        decoded = self.decoder(encoded)  # (batch, num_patches, patch_size*patch_size*in_channels)
        # Reshape patches back to image
        batch_size = x.size(0)
        decoded = decoded.view(batch_size, self.num_patches, self.patch_size, self.patch_size, -1)
        decoded = decoded.permute(0, 4, 1, 2, 3)  # (batch, in_channels, num_patches, patch_size, patch_size)
        # Rearrange patches into image
        num_patches_side = self.img_size // self.patch_size
        decoded = decoded.reshape(batch_size, -1, num_patches_side, num_patches_side, self.patch_size, self.patch_size)
        decoded = decoded.permute(0, 1, 2, 4, 3, 5).contiguous()
        decoded = decoded.view(batch_size, -1, self.img_size, self.img_size)
        return decoded

# -------------------------------
# Training and Evaluation
# -------------------------------
def train_model(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    pbar = tqdm(dataloader, desc="Training")
    for i, (imgs, _) in enumerate(pbar):
        imgs = imgs.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, imgs)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
        
        # Print every 50 batches
        if (i + 1) % 50 == 0:
            print(f"\nBatch {i+1}/{len(dataloader)} | Loss: {loss.item():.4f}")
    
    return running_loss / len(dataloader)

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    print("Loading dataset...")
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    print(f"Dataset loaded. Total batches: {len(dataloader)}")
    
    print("Initializing model...")
    model = ViTAutoencoder(
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        in_channels=1,
        embed_dim=config["embed_dim"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"]
    ).to(device)
    print("Model initialized")
    
    print("Setting up training...")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    print("Starting training...")
    for epoch in range(1, config["num_epochs"] + 1):
        print(f"Epoch {epoch}/{config['num_epochs']}")
        loss = train_model(model, dataloader, criterion, optimizer, device)
        if epoch % config["print_every"] == 0:
            print(f"Epoch {epoch}/{config['num_epochs']} | Loss: {loss:.4f}")
    
    print("Training complete. Evaluating...")
    # Test reconstruction
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_img, _ = test_dataset[0]
    test_img = test_img.to(device)
    
    print("Generating reconstruction...")
    with torch.no_grad():
        recon = model(test_img.unsqueeze(0))
    
    print("Saving results...")
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
    output_path = os.path.join(config["output_dir"], "vit_reconstruction.png")
    plt.savefig(output_path)
    print(f"Reconstruction saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vision Transformer Autoencoder Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)