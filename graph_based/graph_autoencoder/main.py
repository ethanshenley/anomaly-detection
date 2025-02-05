"""
Graph Autoencoder Demo for Anomaly Detection

This demo trains a graph autoencoder on the Cora dataset using PyTorch Geometric.
The encoder uses two GCN layers to learn node embeddings, and the decoder reconstructs
the graph's adjacency matrix via an inner product. Nodes with high reconstruction error
may be considered anomalous.

Usage:
    python main.py --config config.yaml
"""

import os
import yaml
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj

class GraphAutoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphAutoencoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        z = self.conv2(x, edge_index)
        return z
    
    def decode(self, z):
        # Inner product decoder with sigmoid activation
        adj_rec = torch.sigmoid(torch.matmul(z, z.t()))
        return adj_rec
    
    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        adj_rec = self.decode(z)
        return z, adj_rec

def main(config):
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # Load the Cora dataset
    dataset = Planetoid(root="./data", name="Cora")
    data = dataset[0].to(device)
    # Convert edge_index to a dense adjacency matrix
    adj = to_dense_adj(data.edge_index)[0]
    
    model = GraphAutoencoder(in_channels=dataset.num_features,
                             hidden_channels=config["hidden_channels"],
                             out_channels=config["out_channels"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    
    # Training loop
    for epoch in range(1, config["num_epochs"]+1):
        model.train()
        optimizer.zero_grad()
        _, adj_rec = model(data.x, data.edge_index)
        loss = F.binary_cross_entropy(adj_rec, adj)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Evaluation: compute per-node reconstruction error
    model.eval()
    with torch.no_grad():
        _, adj_rec = model(data.x, data.edge_index)
    # For each node, compute the mean squared error between its true and reconstructed connections
    rec_errors = ((adj - adj_rec) ** 2).mean(dim=1).cpu().numpy()
    
    # Plot a histogram of reconstruction errors
    plt.figure(figsize=(8, 6))
    plt.hist(rec_errors, bins=50, alpha=0.7)
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.title("Histogram of Node Reconstruction Errors")
    os.makedirs(config["output_dir"], exist_ok=True)
    output_path = os.path.join(config["output_dir"], "graph_autoencoder_histogram.png")
    plt.savefig(output_path)
    print("Histogram saved to", output_path)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Graph Autoencoder Anomaly Detection Demo")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    main(config)
