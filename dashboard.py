import streamlit as st
import subprocess
import os
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Anomaly Detection Demos Dashboard", layout="wide")

st.title("Anomaly Detection Demos Dashboard")
st.write("""
Welcome to the unified dashboard for exploring various AI anomaly detection methods.
Select a category from the sidebar to explore demos for that technique.
""")

# Sidebar for selecting demo category
category = st.sidebar.selectbox(
    "Select Demo Category",
    [
        "Reconstruction-Based Methods",
        "GAN-Based Methods",
        "One-Class Methods",
        "Self-Supervised Methods",
        "Transformer-Based Methods",
        "Graph-Based Methods",
        "Time Series Forecasting",
        "Energy-Based Methods",
        "Ensemble & Hybrid Approaches",
    ]
)

# Mapping category to available demos
demos = {
    "Reconstruction-Based Methods": {
        "Autoencoder": "reconstruction_based/autoencoder",
        "Conv Autoencoder": "reconstruction_based/conv_autoencoder",
        "Denoising Autoencoder": "reconstruction_based/denoising_autoencoder",
        "Variational Autoencoder": "reconstruction_based/variational_autoencoder",
    },
    "GAN-Based Methods": {
        "AnoGAN": "gan_based/anogan",
        "BiGAN": "gan_based/bigan",
        "Ganomaly": "gan_based/ganomaly",
    },
    "One-Class Methods": {
        "Deep SVDD": "one_class/deep_svdd",
        "One-Class NN": "one_class/one_class_nn",
    },
    "Self-Supervised Methods": {
        "Contrastive Learning": "self_supervised/contrastive_learning",
        "Masked Autoencoder": "self_supervised/masked_autoencoder",
    },
    "Transformer-Based Methods": {
        "Time Series Transformer": "transformers/time_series_transformer",
        "Vision Transformer": "transformers/vision_transformer",
    },
    "Graph-Based Methods": {
        "Graph Autoencoder": "graph_based/graph_autoencoder",
        "GNN Anomaly": "graph_based/gnn_anomaly",
    },
    "Time Series Forecasting": {
        "LSTM Forecasting": "time_series/lstm_forecasting",
        "TCN Forecasting": "time_series/tcn_forecasting",
        "Hybrid Forecasting": "time_series/hybrid_forecasting",
    },
    "Energy-Based Methods": {
        "Energy Model": "energy_based/energy_model",
        "Memory-Augmented Autoencoder": "energy_based/memory_augmented",
    },
    "Ensemble & Hybrid Approaches": {
        "Ensemble Demo": "ensemble_hybrid/ensemble_demo",
        "Hybrid Model Demo": "ensemble_hybrid/hybrid_model_demo",
    },
}

# Display available demos for the selected category
st.sidebar.markdown("### Available Demos")
if category in demos:
    demo_options = list(demos[category].keys())
    selected_demo = st.sidebar.selectbox("Select a Demo", demo_options)
    
    st.header(f"{selected_demo}")
    
    # Display a short description for the demo (customize as needed)
    descriptions = {
        "Autoencoder": "Learn a compact representation of normal data and detect anomalies via reconstruction error.",
        "Conv Autoencoder": "Uses convolutional layers to capture spatial features for image anomaly detection.",
        "Denoising Autoencoder": "Trains to remove noise from images, highlighting anomalies that deviate from normal patterns.",
        "Variational Autoencoder": "Employs a probabilistic framework to model normal data distributions and flag anomalies.",
        "AnoGAN": "Trains a GAN on normal data; uses latent space optimization to detect anomalies based on reconstruction error.",
        "BiGAN": "Incorporates an encoder into the GAN framework for joint representation learning and anomaly detection.",
        "Ganomaly": "Uses an encoder-decoder-encoder architecture to detect anomalies via discrepancies in latent representations.",
        "Deep SVDD": "Maps normal data into a compact region of feature space; anomalies fall outside this region.",
        "One-Class NN": "Trains on normal data only and flags anomalies as deviations from the learned baseline.",
        "Contrastive Learning": "Learns robust representations by maximizing agreement between augmented views of the same input.",
        "Masked Autoencoder": "Trains to reconstruct missing parts of an image; anomalies yield high reconstruction errors.",
        "Time Series Transformer": "Uses self-attention for forecasting time series data, with prediction error as anomaly score.",
        "Vision Transformer": "Processes image patches with a transformer encoder; reconstruction error flags anomalies.",
        "Graph Autoencoder": "Learns node embeddings to reconstruct graph structure; high errors indicate anomalous nodes.",
        "GNN Anomaly": "Utilizes graph neural networks to detect anomalies in graph-structured data.",
        "LSTM Forecasting": "Uses an LSTM to forecast time series values; prediction errors serve as anomaly scores.",
        "TCN Forecasting": "Employs dilated causal convolutions for time series forecasting and anomaly detection.",
        "Hybrid Forecasting": "Fuses LSTM and TCN features for improved forecasting and anomaly detection.",
        "Energy Model": "Assigns an energy score to inputs; higher energy suggests anomalous behavior.",
        "Memory-Augmented Autoencoder": "Uses a learnable memory bank to improve reconstruction of normal patterns, highlighting anomalies.",
        "Ensemble Demo": "Combines reconstruction and forecasting models by fusing their anomaly scores.",
        "Hybrid Model Demo": "A unified model that jointly reconstructs and forecasts; anomaly score is a weighted sum of errors.",
    }
    st.write(descriptions.get(selected_demo, "Description coming soon..."))

    # Add this after the imports (around line 7)
    def run_demo_with_visualization(demo_path, config_path):
        # Create columns for output
        col1, col2 = st.columns(2)
        
        try:
            # Capture and redirect matplotlib calls
            plt.close('all')
            
            # Run the demo as subprocess with output capture
            process = subprocess.run(
                ["python", os.path.join(demo_path, "main.py"), "--config", config_path],
                capture_output=True,
                text=True,
                env={**os.environ, 'MPLBACKEND': 'Agg'}
            )
            
            # Display console output
            with col1:
                st.text("Console Output:")
                st.code(process.stdout)
            
            # Display any generated plots
            with col2:
                st.text("Generated Visualizations:")
                output_dir = "outputs"
                
                # Demo-specific plot files mapping
                plot_files = {
                    "one_class/one_class_nn": {
                        "one_class_nn_scores.png": "One-Class NN Anomaly Scores"
                    },
                    "reconstruction_based/autoencoder": {
                        "reconstruction.png": "Reconstruction Results"
                    },
                    "reconstruction_based/conv_autoencoder": {
                        "conv_reconstruction.png": "Convolutional Autoencoder Results"
                    },
                    "reconstruction_based/denoising_autoencoder": {
                        "denoising_results.png": "Denoising Results"
                    },
                    "reconstruction_based/variational_autoencoder": {
                        "vae_results.png": "VAE Reconstruction Results"
                    },
                    "gan_based/anogan": {
                        "anogan_results.png": "AnoGAN Detection Results"
                    },
                    "gan_based/bigan": {
                        "bigan_results.png": "BiGAN Anomaly Scores"
                    },
                    "gan_based/ganomaly": {
                        "ganomaly_results.png": "Ganomaly Detection Results"
                    },
                    "one_class/deep_svdd": {
                        "deep_svdd_scores.png": "Deep SVDD Anomaly Scores"
                    },
                    "self_supervised/contrastive_learning": {
                        "contrastive_results.png": "Contrastive Learning Results"
                    },
                    "self_supervised/masked_autoencoder": {
                        "masked_ae_results.png": "Masked Autoencoder Results"
                    },
                    "transformers/time_series_transformer": {
                        "transformer_forecast.png": "Time Series Prediction Results"
                    },
                    "transformers/vision_transformer": {
                        "vit_results.png": "Vision Transformer Results"
                    },
                    "graph_based/graph_autoencoder": {
                        "graph_ae_results.png": "Graph Reconstruction Results"
                    },
                    "graph_based/gnn_anomaly": {
                        "gnn_anomaly_scores.png": "GNN Anomaly Scores"
                    },
                    "time_series/lstm_forecasting": {
                        "lstm_forecast.png": "LSTM Forecasting Results"
                    },
                    "time_series/tcn_forecasting": {
                        "tcn_forecast.png": "TCN Forecasting Results"
                    },
                    "time_series/hybrid_forecasting": {
                        "hybrid_forecast.png": "Hybrid Forecasting Results"
                    },
                    "energy_based/energy_model": {
                        "energy_scores.png": "Energy-Based Anomaly Scores"
                    },
                    "energy_based/memory_augmented": {
                        "memory_ae_results.png": "Memory-Augmented Results"
                    },
                    "ensemble_hybrid/ensemble_demo": {
                        "ensemble_scores.png": "Ensemble Model Scores"
                    },
                    "ensemble_hybrid/hybrid_model_demo": {
                        "hybrid_anomaly_score.png": "Hybrid Model Anomaly Score"
                    }
                }
                
                # Get the correct plot files for this demo
                demo_plots = plot_files.get(demo_path, {})
                
                for plot_file, caption in demo_plots.items():
                    plot_path = os.path.join(output_dir, plot_file)
                    if os.path.exists(plot_path):
                        st.image(plot_path, caption=caption)
            
            if process.returncode != 0:
                st.error(f"Demo exited with error code {process.returncode}")
                if process.stderr:
                    st.error(process.stderr)
            else:
                st.success("Demo executed successfully!")
            
        except Exception as e:
            st.error(f"Error running demo: {e}")

    # Then the demo execution code
    if st.button("Run Demo"):
        demo_path = demos[category][selected_demo]
        config_path = os.path.join(demo_path, "config.yaml")
        run_demo_with_visualization(demo_path, config_path)

else:
    st.write("No demos available for the selected category.")

st.sidebar.markdown("## About")
st.sidebar.info("""
This dashboard aggregates various anomaly detection demos covering a range of techniques from reconstruction-based methods to GANs, transformers, and more.
It is designed for educational and exploratory purposes. Happy anomaly detecting!
""")
