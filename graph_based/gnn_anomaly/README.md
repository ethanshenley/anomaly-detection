# GNN-based Anomaly Detection for Graphs

This demo demonstrates **GNN-based anomaly detection** using a synthetic graph. A synthetic graph is generated where normal nodes are drawn from one distribution (e.g., features from N(0,1)) and anomalous nodes from a different distribution (e.g., features from N(5,1)). A simple GCN classifier is then trained to predict anomaly probabilities (with label 1 for anomalies and 0 for normal nodes). The resulting distribution of anomaly scores is visualized to highlight the separation between normal and anomalous nodes.

## Anomaly Detection Use Case

**When to Use GNN-based Anomaly Detection:**

- **Graph-Structured Data:**  
  When your data is naturally represented as a graph (e.g., social networks, transaction networks), GNNs can leverage both node features and connectivity patterns to detect anomalies.

- **Supervised or Semi-Supervised Settings:**  
  This approach is useful when some labels (or synthetic labels) are available, allowing the model to learn to distinguish nodes that deviate from normal patterns.

- **Network Security and Fraud Detection:**  
  In applications such as fraud detection or cybersecurity, GNN-based models can detect anomalous behaviors by learning both local and global patterns in the graph.

## Setup

1. **Clone the repository** and navigate to the `graph_based/gnn_anomaly/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Generate a synthetic graph with node features and anomaly labels.
    * Train a GCN classifier to predict the probability of a node being anomalous.
    * Visualize the distribution of anomaly scores for normal and anomalous nodes.
    * Save a histogram of anomaly scores to the outputs/ directory.