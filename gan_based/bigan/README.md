# BiGAN for Anomaly Detection

This demo implements a simplified version of **BiGAN (Bidirectional GAN)** using PyTorch on the MNIST dataset.
BiGAN extends traditional GANs by incorporating an encoder that maps images into the latent space.
After training, the encoder and generator can be used together to reconstruct an image.
The reconstruction error can serve as an anomaly score, with higher errors indicating potential anomalies.

## Anomaly Detection Use Case

**When to Use BiGAN:**

- **Joint Representation Learning:**  
  BiGAN learns both the generative and inference models simultaneously, providing meaningful latent representations.
  
- **Anomaly Scoring:**  
  By comparing the original image to its reconstruction (obtained via the encoder and generator), anomalies can be identified as samples with high reconstruction error.
  
- **Data with Complex Structures:**  
  This method is suitable for complex data where learning a bidirectional mapping helps in capturing the underlying data distribution.

## Setup

1. **Clone the repository** and navigate to the `gan_based/bigan/` directory.

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Adjust hyperparameters in config.yaml if desired.

4. Run the demo with: `python main.py --config config.yaml`

The script will:

    * Train the BiGAN model on the MNIST dataset.
    * Encode and reconstruct a test image.
    * Display and save the original and reconstructed images, where high reconstruction error may indicate an anomaly.