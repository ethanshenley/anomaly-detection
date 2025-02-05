# Convolutional Autoencoder for Anomaly Detection

This demo implements a **convolutional autoencoder** using PyTorch on the MNIST dataset. The network is built with convolutional layers to better capture spatial features from images. This makes it particularly well-suited for anomaly detection in image data.

## Anomaly Detection Use Case

**When to Use a Convolutional Autoencoder:**

- **Spatially Correlated Data:**  
  Convolutional autoencoders excel when dealing with images or other data that have strong spatial correlations. For example, in industrial inspection, you might use a convolutional autoencoder to detect defects in manufactured products (e.g., scratches, dents, or misalignments) by learning the normal appearance of the product.

- **Feature Preservation:**  
  Unlike fully connected networks, convolutional networks preserve the spatial hierarchy of features, making them ideal for tasks where the location and structure of features are important.

- **Image Reconstruction:**  
  In anomaly detection, the model is trained only on "normal" data. When presented with an image that significantly deviates from the norm, the network’s inability to accurately reconstruct the image leads to a high reconstruction error. This error can be used as a signal to flag potential anomalies.

## Setup

1. **Clone the repository** (if you haven't already) and navigate to the `reconstruction_based/conv_autoencoder/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Adjust hyperparameters in config.yaml if desired.


## Running the Demo

Run the demo with: `python main.py --config config.yaml`

The script will:

* Download the MNIST dataset.
* Train the convolutional autoencoder for the specified number of epochs.
* Visualize original and reconstructed images.
* Save the reconstruction plot to the outputs/ directory.


## Why This Technique?

Convolutional autoencoders are highly effective when anomalies are related to spatial or structural changes in images. They are widely used in applications such as:

    Medical imaging (detecting tumors or lesions)
    Industrial quality control (spotting defects in products)
    Surveillance (identifying unusual patterns or objects)

By leveraging the spatial feature extraction capabilities of convolutional layers, these models can learn a rich representation of normal data and thereby more effectively highlight anomalous deviations.

Enjoy exploring and extending this demo!