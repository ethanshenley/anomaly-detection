# Graph Autoencoder for Anomaly Detection

This demo implements a **graph autoencoder** using PyTorch Geometric on the Cora dataset. The model employs two GCN layers as an encoder to learn node embeddings and reconstructs the graph's adjacency matrix via an inner product decoder. Nodes that exhibit high reconstruction error are flagged as potential anomalies, which may indicate unusual or unexpected connectivity patterns in the network.

## Anomaly Detection Use Case

**When to Use a Graph Autoencoder:**

- **Structural Anomaly Detection:**  
  Graph autoencoders capture the underlying structure of graph data. In applications such as fraud detection, social network analysis, or cybersecurity, unusual patterns (e.g., unexpected connections or missing links) lead to high reconstruction errors that can be used to flag anomalies.

- **Unsupervised Learning:**  
  This approach does not require labeled anomalies; it is trained solely on normal graph structure. Deviations from the learned patterns naturally emerge as high reconstruction errors.

- **Network Monitoring:**  
  In network security or monitoring applications, a graph autoencoder can help identify abnormal behaviors in a network by spotting nodes whose connectivity deviates from the norm.

## Setup

1. **Clone the repository** and navigate to the `graph_based/graph_autoencoder/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Download the Cora dataset.
    * Train the graph autoencoder to reconstruct the adjacency matrix.
    * Compute the reconstruction error for each node.
    * Plot and save a histogram of reconstruction errors to the outputs/ directory.