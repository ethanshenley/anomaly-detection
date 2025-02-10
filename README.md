# Anomaly Detection Demos Repository

Welcome to the **Anomaly Detection Demos Repository**! This repository provides a comprehensive suite of self-contained demos covering a wide spectrum of anomaly detection techniques using modern AI/ML methods (primarily implemented in PyTorch). Each demo includes end-to-end code, configuration files, and detailed README instructions.

## Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [How to Use the Demos](#how-to-use-the-demos)
- [Techniques & Use Cases Guide](#techniques--use-cases-guide)
- [Contributing](#contributing)
- [License](#license)

## Overview

This repository is a one-stop resource for anomaly detection demos. It includes examples for various paradigms such as:

- **Reconstruction-Based Methods:** Autoencoders (vanilla, convolutional, denoising, variational)
- **Adversarial (GAN)-Based Methods:** AnoGAN, BiGAN, Ganomaly
- **One-Class Methods:** Deep SVDD, One-Class Neural Networks
- **Self-Supervised Methods:** Contrastive Learning, Masked Autoencoders
- **Transformer-Based Methods:** Time Series Transformer, Vision Transformer
- **Graph-Based Methods:** Graph Autoencoder, GNN-Based Anomaly Detection
- **Time Series Forecasting:** LSTM Forecasting, TCN Forecasting, Hybrid Forecasting
- **Energy-Based & Memory-Augmented Methods:** Energy Models, Memory-Augmented Autoencoders
- **Ensemble & Hybrid Approaches:** Ensemble Demo and Hybrid Model Demo

Each category addresses different types of anomaly detection challenges—from image data to time series, graph data, and beyond.


Each demo folder contains:
- **main.py** – the entry point for the demo.
- **config.yaml** – configuration file with hyperparameters.
- **README.md** – detailed instructions specific to that demo.
- **requirements.txt** – list of required Python packages.

## Getting Started

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/anomaly-detection-demos.git
   cd anomaly-detection-demos

2. **Set Up a Virtual Environment (Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**

   ```bash
   cd reconstruction_based/autoencoder
   pip install -r requirements.txt
   ```

4. **Launch the Dashboard:**

   ```bash
   # From the root directory
   pip install streamlit
   streamlit run dashboard.py
   ```

   This will open a web interface where you can explore all demos interactively.

## How to Use the Demos

There are two ways to run the demos:

### 1. Using the Central Dashboard (Recommended)

The easiest way to explore all demos is through our Streamlit dashboard:

```bash
streamlit run dashboard.py
```

This will:
- Launch a web interface at http://localhost:8501
- Allow you to select different categories of anomaly detection methods
- Run demos with visualizations and real-time output
- Display results and plots in an organized manner

### 2. Running Individual Demos

Each demo is also fully self-contained and can be run independently:

1. Navigate to the desired demo's folder
2. (Optional) Adjust the hyperparameters in config.yaml
3. Run the demo with:

```bash
python main.py --config config.yaml
```
Refer to the README inside each folder for demo-specific instructions and sample outputs.

## Techniques & Use Cases Guide

1. **Reconstruction-Based Methods**
        Use When: You need to learn a compact representation of normal data (e.g., images, sensor data).
        Example: For industrial defect detection, try reconstruction_based/conv_autoencoder.

2. **Adversarial (GAN)-Based Methods**
        Use When: You need to model complex data distributions and leverage generative models for anomaly detection.
        Example: For detecting unusual patterns in medical imaging, see gan_based/ganomaly.

3. **One-Class Methods**
        Use When: You have only normal data and want to detect anomalies as outliers.
        Use When: You have only normal data and want to detect anomalies as outliers.
        Example: For network intrusion detection, check out one_class/deep_svdd.

4. **Self-Supervised Methods**
        Use When: You need robust feature representations without requiring labeled anomalies.
        Use When: You need robust feature representations without requiring labeled anomalies.
        Example: For unsupervised video surveillance anomaly detection, explore self_supervised/contrastive_learning.

5. **Transformer-Based Methods**
        Use When: Your data exhibits long-range dependencies or requires capturing global context.
        Use When: Your data exhibits long-range dependencies or requires capturing global context.
        Example: For sensor time series forecasting, use transformers/time_series_transformer. For image-based anomalies, try transformers/vision_transformer.

6. **Graph-Based Methods**
        Use When: Your data is structured as a graph (e.g., social networks, transaction networks).
        Use When: Your data is structured as a graph (e.g., social networks, transaction networks).
        Example: For detecting fraud in financial networks, refer to graph_based/gnn_anomaly.

7. **Time Series Forecasting Methods**
        Use When: You want to detect anomalies via forecasting errors.
        Use When: You want to detect anomalies via forecasting errors.
        Example: For financial time series, try time_series/lstm_forecasting or time_series/tcn_forecasting. For a fusion of approaches, use time_series/hybrid_forecasting.

8. **Energy-Based & Memory-Augmented Methods**
        Use When: You prefer unsupervised methods that assign an energy score or use external memory for anomaly detection.
        Use When: You prefer unsupervised methods that assign an energy score or use external memory for anomaly detection.
        Example: For sensor data with high noise levels, see energy_based/memory_augmented.

9. **Ensemble & Hybrid Approaches**
        Use When: You want to combine multiple models for more robust anomaly detection.
        Use When: You want to combine multiple models for more robust anomaly detection.
        Use When: You want to combine multiple models for more robust anomaly detection.
        Example: For critical applications in manufacturing quality control, explore ensemble_hybrid/ensemble_demo or ensemble_hybrid/hybrid_model_demo.

## Contributing

Contributions, bug fixes, and feature enhancements are welcome! If you have ideas or improvements, please fork the repository and submit a pull request.


