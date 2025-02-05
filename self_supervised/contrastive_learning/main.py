#!/usr/bin/env python
"""
Contrastive Learning Demo for Anomaly Detection

This script implements a simple contrastive learning framework (SimCLR-like) using PyTorch on the MNIST dataset.
Two augmented views of each image are generated and used to train an encoder network with a projection head.
The learned representations can be used for anomaly detection by identifying samples that do not conform
to the learned clusters of normal data.

Usage:
    python main.py --config config.yaml
"""

import os
import yaml
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn.functional as F

# ----- Data Augmentation for Contrastive Learning -----
class ContrastiveTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
    
    def __call__(self, x):
        # Return two different augmented versions of the same image
        return self.base_transform(x), self.base_transform(x)

class ContrastiveMNIST(Dataset):
    def __init__(self, train=True, transform=None):
        self.dataset = datasets.MNIST(root="./data", train=train, download=True)
        self.transform = transform
    
    def __getitem__(self, index):
        img, label = self.dataset[index]
        if self.transform:
            xi, xj = self.transform(img)
        else:
            xi, xj = img, img
        return xi, xj, label  # label is returned only for visualization
    
    def __len__(self):
        return len(self.dataset)

# ----- Model Definition -----
class Encoder(nn.Module):
    def __init__(self, feature_dim):
        super(Encoder, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28 -> 14
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 14 -> 7
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc = nn.Linear(64 * 7 * 7, feature_dim)
    
    def forward(self, x):
        x = self.conv_net(x)
        x = self.fc(x)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, input_dim, proj_dim):
        super(ProjectionHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, proj_dim)
        )
    
    def forward(self, x):
        return self.net(x)

class ContrastiveModel(nn.Module):
    def __init__(self, feature_dim=128):
        super(ContrastiveModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.projector = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim)
        )

    def forward(self, x):
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        z = self.projector(h)
        return F.normalize(z, dim=1)

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = torch.exp(torch.mm(z, z.t()) / self.temperature)
        mask = torch.zeros_like(sim_matrix)
        mask[range(batch_size), range(batch_size, 2*batch_size)] = 1.
        mask[range(batch_size, 2*batch_size), range(batch_size)] = 1.
        
        sim_matrix = sim_matrix / (sim_matrix.sum(dim=1, keepdim=True) + 1e-8)
        loss = -(torch.log(sim_matrix + 1e-8) * mask).sum() / (2 * batch_size)
        return loss

def get_augmentation_transform():
    return transforms.Compose([
        transforms.RandomAffine(15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
    ])

def visualize_embeddings(model, dataloader, device, output_dir):
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, target in dataloader:
            images = images.to(device)
            embedding = model(images)
            embeddings.append(embedding.cpu().numpy())
            labels.append(target.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    # Use t-SNE for visualization
    tsne = TSNE(n_components=2, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title('t-SNE visualization of learned embeddings')
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'embeddings.png'))
    plt.close()

def train(model, dataloader, optimizer, device, temperature):
    model.train()
    running_loss = 0.0
    criterion = ContrastiveLoss(temperature=temperature)
    
    for batch_idx, (xi, xj, _) in enumerate(dataloader):
        xi, xj = xi.to(device), xj.to(device)
        
        optimizer.zero_grad()
        
        # Get embeddings for both augmented views
        zi = model(xi)
        zj = model(xj)
        
        # Calculate contrastive loss
        loss = criterion(zi, zj)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx+1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    base_transform = transforms.Compose([
        transforms.RandomResizedCrop(28, scale=(0.8, 1.0)),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    contrastive_transform = ContrastiveTransform(base_transform)
    
    train_dataset = ContrastiveMNIST(train=True, transform=contrastive_transform)
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    
    # For visualization, we use the standard MNIST test set without augmentation.
    test_transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    model = ContrastiveModel(feature_dim=config["feature_dim"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    temperature = config["temperature"]
    
    for epoch in range(1, config["num_epochs"] + 1):
        avg_loss = train(model, train_loader, optimizer, device, temperature)
        print(f"Epoch [{epoch}/{config['num_epochs']}]: Average Loss = {avg_loss:.4f}")
    
    visualize_embeddings(model, test_loader, device, config["output_dir"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Contrastive Learning Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)
