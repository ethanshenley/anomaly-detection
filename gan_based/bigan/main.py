"""
BiGAN Demo for Anomaly Detection

This demo implements a simplified version of BiGAN (Bidirectional GAN) using PyTorch on the MNIST dataset.
BiGAN extends traditional GANs by incorporating an encoder that learns to map images into the latent space.
The discriminator is trained to differentiate between joint pairs of (image, latent vector).
After training, the encoder and generator can be used together to reconstruct an image.
The reconstruction error is used as an anomaly score.

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

# Generator: maps latent vector z to image
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    def forward(self, z):
        x = self.model(z)
        x = x.view(-1, 1, 28, 28)
        return x

# Encoder: maps image to latent vector
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, latent_dim)
        )
    
    def forward(self, x):
        # Set to eval mode during inference
        if not self.training and x.size(0) == 1:
            self.eval()
        return self.model(x)

# Discriminator: distinguishes between (x, z) pairs
class Discriminator(nn.Module):
    def __init__(self, latent_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28*28 + latent_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1)  # Remove sigmoid for Wasserstein
        )
    def forward(self, x, z):
        x_flat = x.view(x.size(0), -1)
        combined = torch.cat([x_flat, z], dim=1)
        return self.model(combined)

def train_bigan(generator, encoder, discriminator, dataloader, device, config):
    # Initialize optimizers with lower learning rate
    optimizer_D = optim.Adam(discriminator.parameters(), lr=config["lr"] * 0.1)
    optimizer_G_E = optim.Adam(
        list(generator.parameters()) + list(encoder.parameters()), 
        lr=config["lr"] * 0.1
    )
    
    # Add learning rate scheduling
    scheduler_D = optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.995)
    scheduler_G_E = optim.lr_scheduler.ExponentialLR(optimizer_G_E, gamma=0.995)
    
    # Modified Wasserstein loss with gradient penalty
    def wasserstein_loss(real_output, fake_output):
        # Scale down the outputs to prevent explosion
        real_output = real_output * 0.01
        fake_output = fake_output * 0.01
        return torch.mean(fake_output) - torch.mean(real_output)  # Note the sign flip
    
    def compute_gradient_penalty(real_samples, fake_samples, real_z, fake_z, discriminator):
        batch_size = real_samples.size(0)
        alpha_x = torch.rand(batch_size, 1, 1, 1).to(real_samples.device)
        alpha_z = torch.rand(batch_size, 1).to(real_samples.device)
        
        # Interpolate between real and fake samples for both x and z
        interpolated_x = alpha_x * real_samples + (1 - alpha_x) * fake_samples
        interpolated_z = alpha_z * real_z + (1 - alpha_z) * fake_z
        
        interpolated_x.requires_grad_(True)
        interpolated_z.requires_grad_(True)
        
        # Calculate discriminator output for interpolated pairs
        d_interpolated = discriminator(interpolated_x, interpolated_z)
        
        # Calculate gradients
        grad_outputs = torch.ones_like(d_interpolated)
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=[interpolated_x, interpolated_z],
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True
        )
        
        # Calculate gradient penalty
        gradients_x = gradients[0].view(batch_size, -1)
        gradients_z = gradients[1].view(batch_size, -1)
        gradient_penalty = ((gradients_x.norm(2, dim=1) - 1) ** 2).mean() + \
                          ((gradients_z.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    # Add reconstruction loss function
    reconstruction_criterion = nn.MSELoss()

    latent_dim = config["latent_dim"]
    num_epochs = config["num_epochs"]
    
    for epoch in range(1, num_epochs+1):
        for i, (imgs, _) in enumerate(dataloader):
            batch_size = imgs.size(0)
            imgs = imgs.to(device)
            imgs = (imgs - 0.5) * 2  # Normalize to [-1, 1]
            
            # Train Discriminator
            optimizer_D.zero_grad()
            z_random = torch.randn(batch_size, config["latent_dim"], device=device)
            
            with torch.no_grad():
                x_fake = generator(z_random)
                z_encoded = encoder(imgs)
            
            d_real = discriminator(imgs, z_encoded.detach())
            d_fake = discriminator(x_fake.detach(), z_random)
            
            d_loss = wasserstein_loss(d_real, d_fake)
            gp = compute_gradient_penalty(imgs, x_fake, z_encoded.detach(), z_random, discriminator)
            d_loss = d_loss + 0.1 * gp
            
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()
            
            # Train Generator and Encoder
            optimizer_G_E.zero_grad()
            x_fake = generator(z_random)
            z_encoded = encoder(imgs)
            x_recon = generator(z_encoded)  # Generate reconstruction
            
            d_fake = discriminator(x_fake, z_random)
            d_real = discriminator(imgs, z_encoded)
            
            # Combine adversarial and reconstruction losses
            ge_loss = -wasserstein_loss(d_real, d_fake)
            recon_loss = reconstruction_criterion(x_recon, imgs)
            total_loss = ge_loss + 10.0 * recon_loss  # Weight reconstruction more heavily
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(generator.parameters()) + list(encoder.parameters()), 
                max_norm=1.0
            )
            optimizer_G_E.step()
            
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch}/{num_epochs} | Batch {i+1} | "
                      f"D_loss: {d_loss.item():.4f} | GE_loss: {ge_loss.item():.4f} | "
                      f"Recon_loss: {recon_loss.item():.4f}")
        
            # Step the schedulers
            scheduler_D.step()
            scheduler_G_E.step()
    
    return generator, encoder, discriminator

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    latent_dim = config["latent_dim"]
    generator = Generator(latent_dim).to(device)
    encoder = Encoder(latent_dim).to(device)
    discriminator = Discriminator(latent_dim).to(device)
    
    print("Training BiGAN...")
    generator, encoder, discriminator = train_bigan(generator, encoder, discriminator, dataloader, device, config)
    
    # After training, evaluate reconstruction
    generator.eval()
    encoder.eval()
    
    with torch.no_grad():
        # Process multiple test images
        test_dataset = datasets.MNIST(root="data", train=False, transform=transforms.ToTensor())
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
        test_batch = next(iter(test_loader))[0].to(device)
        
        # Get reconstructions
        z_encoded = encoder(test_batch)
        x_recon = generator(z_encoded)
        
        # Display first 8 original and reconstructed images
        fig, axes = plt.subplots(2, 8, figsize=(16, 4))
        for i in range(8):
            # Original
            axes[0,i].imshow(test_batch[i].cpu().squeeze(), cmap='gray')
            axes[0,i].axis('off')
            # Reconstruction
            axes[1,i].imshow(x_recon[i].cpu().squeeze(), cmap='gray')
            axes[1,i].axis('off')
        
        plt.tight_layout()
        os.makedirs(config["output_dir"], exist_ok=True)
        plt.savefig(os.path.join(config["output_dir"], "bigan_reconstructions.png"))
        plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BiGAN Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)