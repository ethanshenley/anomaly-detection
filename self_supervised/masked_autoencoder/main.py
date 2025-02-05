"""
Masked Autoencoder Demo for Anomaly Detection

This script trains a masked autoencoder on the MNIST dataset.
A portion of each input image is randomly masked, and the model is trained to reconstruct the original image.
In anomaly detection, if the reconstruction error (especially in the masked regions) is high,
it may indicate that the input deviates from the learned normal patterns.

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

def mask_image(img, mask_ratio):
    """
    Randomly masks a fraction of the pixels in the image.
    img: Tensor of shape (C, H, W)
    mask_ratio: Fraction of pixels to mask (set to 0)
    Returns the masked image and the mask used.
    """
    mask = torch.bernoulli(torch.full(img.shape, 1 - mask_ratio)).to(img.device)
    masked_img = img * mask
    return masked_img, mask

class MaskedAutoencoder(nn.Module):
    def __init__(self):
        super(MaskedAutoencoder, self).__init__()
        # Encoder: Downsample the image
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(True)
        )
        # Decoder: Upsample back to the original size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # 7 -> 14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),   # 14 -> 28
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

def train(model, dataloader, criterion, optimizer, device, mask_ratio):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        masked_data, _ = mask_image(data, mask_ratio)
        optimizer.zero_grad()
        outputs = model(masked_data)
        loss = criterion(outputs, data)  # Compare reconstruction with original image
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def test_and_visualize(model, dataloader, device, mask_ratio, output_dir):
    model.eval()
    data_iter = iter(dataloader)
    images, _ = next(data_iter)
    images = images.to(device)
    masked_images, _ = mask_image(images, mask_ratio)
    
    with torch.no_grad():
        outputs = model(masked_images)
    
    images = images.cpu().numpy()
    masked_images = masked_images.cpu().numpy()
    outputs = outputs.cpu().numpy()
    
    n = 8  # Number of images to display
    fig, axes = plt.subplots(3, n, figsize=(n * 1.5, 4))
    for i in range(n):
        # Row 1: Original image
        axes[0, i].imshow(np.squeeze(images[i]), cmap='gray')
        axes[0, i].axis('off')
        # Row 2: Masked image
        axes[1, i].imshow(np.squeeze(masked_images[i]), cmap='gray')
        axes[1, i].axis('off')
        # Row 3: Reconstructed image
        axes[2, i].imshow(np.squeeze(outputs[i]), cmap='gray')
        axes[2, i].axis('off')
    
    plt.suptitle("Row 1: Original | Row 2: Masked | Row 3: Reconstructed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "masked_reconstruction.png")
    plt.savefig(output_path)
    print(f"Reconstruction plot saved to {output_path}")
    plt.show()

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
    
    model = MaskedAutoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    mask_ratio = config["mask_ratio"]
    
    for epoch in range(1, config["num_epochs"] + 1):
        avg_loss = train(model, train_loader, criterion, optimizer, device, mask_ratio)
        print(f"Epoch [{epoch}/{config['num_epochs']}]: Loss = {avg_loss:.4f}")
    
    test_and_visualize(model, test_loader, device, mask_ratio, config["output_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Masked Autoencoder Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)