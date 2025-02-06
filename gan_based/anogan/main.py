"""
AnoGAN Demo for Anomaly Detection

This demo trains a GAN on the MNIST dataset using PyTorch.
After training, the generator is used to perform latent space optimization (inversion)
to find the latent vector that best reconstructs a given test image.
The reconstruction error serves as an anomaly score. High reconstruction error
may indicate that the input is anomalous compared to the training data.

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

# Define the Generator network
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28*28),
            nn.Tanh()  # Output between -1 and 1
        )
    def forward(self, z):
        x = self.model(z)
        x = x.view(-1, 1, 28, 28)
        return x

# Define the Discriminator network
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

def train_gan(generator, discriminator, dataloader, device, config):
    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(generator.parameters(), lr=config["lr"])
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config["lr"])
    latent_dim = config["latent_dim"]
    num_epochs = config["num_epochs"]
    
    for epoch in range(1, num_epochs+1):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            real = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            
            imgs = imgs.to(device)
            imgs = (imgs - 0.5) * 2  # Scale images to [-1,1]
            
            # Train Generator
            optimizer_G.zero_grad()
            z = torch.randn(batch_size, latent_dim, device=device)
            gen_imgs = generator(z)
            validity = discriminator(gen_imgs)
            g_loss = criterion(validity, real)
            g_loss.backward()
            optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()
            real_loss = criterion(discriminator(imgs), real)
            fake_loss = criterion(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{num_epochs} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")
    return generator, discriminator

def latent_optimization(generator, test_img, device, config):
    # Initialize latent vector randomly and optimize it to reconstruct test_img
    latent_dim = config["latent_dim"]
    z = torch.randn(1, latent_dim, device=device, requires_grad=True)
    optimizer = optim.Adam([z], lr=config["inversion_lr"])
    test_img = test_img.to(device)
    test_img = (test_img - 0.5) * 2  # scale to [-1,1]
    num_steps = config["inversion_steps"]
    criterion = nn.MSELoss()
    
    for step in range(num_steps):
        optimizer.zero_grad()
        gen_img = generator(z)
        loss = criterion(gen_img, test_img.unsqueeze(0))
        loss.backward()
        optimizer.step()
        if (step+1) % 100 == 0:
            print(f"Inversion step {step+1}/{num_steps}, Loss: {loss.item():.4f}")
    return z, loss.item(), gen_img

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    generator = Generator(config["latent_dim"]).to(device)
    discriminator = Discriminator().to(device)
    
    print("Training GAN...")
    generator, discriminator = train_gan(generator, discriminator, dataloader, device, config)
    
    # Select a test image
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_img, label = test_dataset[0]  # for demonstration, take the first image
    print("Performing latent space optimization on a test image...")
    z_opt, recon_loss, gen_img = latent_optimization(generator, test_img, device, config)
    
    # Scale images back to [0,1] for visualization
    test_img_np = test_img.numpy().squeeze()
    gen_img_np = gen_img.detach().cpu().numpy().squeeze()
    gen_img_np = (gen_img_np + 1) / 2  # from [-1,1] to [0,1]
    
    # Plot original and reconstructed images
    fig, axes = plt.subplots(1, 2, figsize=(6,3))
    axes[0].imshow(test_img_np, cmap="gray")
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    axes[1].imshow(gen_img_np, cmap="gray")
    axes[1].set_title(f"Reconstructed (Loss: {recon_loss:.4f})")
    axes[1].axis("off")
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], "anogan_reconstruction.png")
    plt.savefig(output_path)
    print("Reconstruction saved to", output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnoGAN Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)
