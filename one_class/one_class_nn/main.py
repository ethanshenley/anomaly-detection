"""
One-Class Neural Network Demo for Anomaly Detection

This demo implements a simple one-class neural network using PyTorch on the MNIST dataset.
The model is trained solely on normal data (digit 0) to output a value close to 0.
During testing, the absolute value of the network's output is used as an anomaly score.
Samples with higher anomaly scores are considered anomalous.

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
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

class OneClassNN(nn.Module):
    def __init__(self):
        super(OneClassNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # 14x14 -> 7x7
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU()
        )
        self.fc = nn.Linear(64, 1)  # Output a single scalar
    def forward(self, x):
        feat = self.features(x)
        out = self.fc(feat)
        return out

def filter_normal(dataset, normal_class=0):
    # Filter dataset to include only samples of the designated normal class
    indices = [i for i, (img, label) in enumerate(dataset) if label == normal_class]
    return Subset(dataset, indices)

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        target = torch.zeros_like(output)  # Target is 0 for normal data
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx + 1}/{len(dataloader)}: Loss = {loss.item():.4f}")
    avg_loss = running_loss / len(dataloader)
    return avg_loss

def test(model, dataloader, device):
    model.eval()
    scores = []
    labels = []
    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)
            output = model(data)
            # Use the absolute output as the anomaly score
            score = torch.abs(output).squeeze().cpu().numpy()
            scores.extend(score)
            labels.extend(label.numpy())
    return np.array(scores), np.array(labels)

def visualize_scores(scores, labels, output_dir):
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=50, alpha=0.7)
    plt.xlabel("Anomaly Score (Absolute Output)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Anomaly Scores on Test Data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "one_class_nn_scores.png")
    plt.savefig(output_path)
    print(f"Anomaly score histogram saved to {output_path}")
    plt.show()

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([transforms.ToTensor()])
    full_train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    # Filter training data to include only the normal class (digit 0 by default)
    train_dataset = filter_normal(full_train_dataset, normal_class=config["normal_class"])
    
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    
    model = OneClassNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    for epoch in range(1, config["num_epochs"] + 1):
        avg_loss = train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch}/{config['num_epochs']}]: Average Loss = {avg_loss:.4f}")
    
    scores, labels = test(model, test_loader, device)
    visualize_scores(scores, labels, config["output_dir"])
    
    # Additionally, print average scores for normal and anomalous classes
    normal_scores = scores[labels == config["normal_class"]]
    anomaly_scores = scores[labels != config["normal_class"]]
    print(f"Average score for normal class ({config['normal_class']}): {np.mean(normal_scores):.4f}")
    print(f"Average score for anomalies (other digits): {np.mean(anomaly_scores):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-Class Neural Network Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)