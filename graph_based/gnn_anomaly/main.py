"""
GNN-based Anomaly Detection Demo for Graphs

This demo generates a synthetic graph with node features and anomaly labels. Normal nodes are drawn
from one distribution, while anomalous nodes come from a different distribution. A simple GCN classifier
is trained to distinguish between normal and anomalous nodes by predicting an anomaly probability.
The resulting distribution of anomaly scores is visualized to show the separation between the two groups.

Usage:
    python main.py --config config.yaml
"""

import os
import yaml
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

def generate_synthetic_graph(num_nodes=100, feature_dim=16, anomaly_ratio=0.1):
    num_anomalies = int(num_nodes * anomaly_ratio)
    num_normal = num_nodes - num_anomalies
    # Generate node features: normal nodes from N(0,1), anomalies from N(5,1)
    features_normal = torch.randn(num_normal, feature_dim)
    features_anomaly = torch.randn(num_anomalies, feature_dim) + 5
    x = torch.cat([features_normal, features_anomaly], dim=0)
    # Create labels: 0 for normal, 1 for anomaly
    y = torch.cat([torch.zeros(num_normal, dtype=torch.long),
                   torch.ones(num_anomalies, dtype=torch.long)], dim=0)
    # Generate edges: for simplicity, iterate over node pairs and add an edge based on a probability
    edge_index = []
    for i in range(num_nodes):
        for j in range(i+1, num_nodes):
            if y[i] == 0 and y[j] == 0:
                p = 0.2
            elif y[i] == 1 and y[j] == 1:
                p = 0.1
            else:
                p = 0.05
            if random.random() < p:
                edge_index.append([i, j])
                edge_index.append([j, i])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return Data(x=x, edge_index=edge_index, y=y)

class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes):
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
    
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Generate a synthetic graph with specified parameters
    data = generate_synthetic_graph(num_nodes=config["num_nodes"],
                                    feature_dim=config["feature_dim"],
                                    anomaly_ratio=config["anomaly_ratio"])
    data = data.to(device)
    
    model = GCNClassifier(in_channels=config["feature_dim"],
                          hidden_channels=config["hidden_channels"],
                          num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(1, config["num_epochs"]+1):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Evaluation: compute anomaly probability (class 1 probability) for each node
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        probs = F.softmax(logits, dim=1)[:, 1]  # probability for anomaly class
        anomaly_scores = probs.cpu().numpy()
        labels = data.y.cpu().numpy()
    
    # Plot histograms of anomaly scores for normal and anomalous nodes
    plt.figure(figsize=(8, 6))
    plt.hist(anomaly_scores[labels == 0], bins=20, alpha=0.7, label="Normal")
    plt.hist(anomaly_scores[labels == 1], bins=20, alpha=0.7, label="Anomaly")
    plt.xlabel("Anomaly Probability")
    plt.ylabel("Frequency")
    plt.title("Distribution of Anomaly Scores")
    plt.legend()
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], "gnn_anomaly_histogram.png")
    plt.savefig(output_path)
    print("Histogram saved to", output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN-based Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)