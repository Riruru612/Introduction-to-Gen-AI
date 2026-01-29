# Variational Autoencoder (VAE) – Fashion-MNIST  
**Latent Space Analysis With and Without KL Divergence**


## 1. Objective

The objective of this assignment is to understand and implement a **Variational Autoencoder (VAE)** and to analyze the effect of **KL Divergence** on:

- Latent space structure  
- Reconstruction quality  
- Generative capability  

A comparative study is performed between:
- A model trained **without KL divergence** (Autoencoder-like behavior)
- A model trained **with KL divergence** (True VAE behavior)


## 2. Learning Outcomes

After completing this assignment, the following concepts are clearly understood:

- Difference between **Autoencoders** and **Variational Autoencoders**
- Role of **latent space regularization**
- Importance of **KL divergence** in generative models
- Reconstruction vs generated samples
- Visualization and interpretation of latent space distributions
- Model saving, loading, and reproducibility


## 3. Dataset

- **Dataset:** Fashion-MNIST  
- **Image Size:** 28 × 28 grayscale  
- **Classes:** 10 (T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot)

The dataset is automatically downloaded using `torchvision.datasets.FashionMNIST`.



## 4. Model Architecture

### Encoder
- Fully connected layers
- Outputs:
  - Mean vector (μ)
  - Log variance vector (log σ²)

### Latent Space
- Latent dimension = **2**
- Enables direct visualization of learned representations

### Decoder
- Fully connected layers
- Reconstructs image from latent vector

### Reparameterization Trick
Used to enable backpropagation through stochastic sampling:
\[
z = \mu + \sigma \cdot \epsilon,\quad \epsilon \sim \mathcal{N}(0,1)
\]



## 5. Loss Functions

### 5.1 Reconstruction Loss
Binary Cross Entropy (BCE) is used to measure pixel-wise reconstruction error.

### 5.2 KL Divergence Loss
KL divergence regularizes the latent distribution to match a standard normal distribution:
\[
D_{KL}(q(z|x) || p(z))
\]

### Training Variants
- **Without KL Divergence:**  
  Uses only reconstruction loss (behaves like a standard autoencoder)
- **With KL Divergence:**  
  Uses reconstruction loss + KL divergence (true VAE)


## 6. Training Procedure

- Optimizer: Adam  
- Epochs: 20  
- Batch Size: 128  
- Learning Rate: 0.001  

During training:
- Reconstructed images are saved after each epoch
- Generated samples are saved after each epoch
- Models are saved for later reuse

## 7. Reconstruction vs Generated Samples

### Reconstructed Images
- Input image → Encoder → Latent space → Decoder
- Used during training
- Measures how well the model preserves information

### Generated Samples
- Random latent vector sampled from \( \mathcal{N}(0,1) \)
- Passed directly to decoder
- Used only for evaluation and visualization
- Demonstrates generative capability


## 8. Latent Space Visualization

Two separate latent space plots are generated and saved:

1. **Latent Space WITHOUT KL Divergence**
   - Discrete clusters
   - Large empty regions
   - Poor sampling performance

2. **Latent Space WITH KL Divergence**
   - Smooth, continuous distribution
   - Gaussian-like structure
   - Enables meaningful random sampling

These plots visually demonstrate the regularization effect of KL divergence.


## 9. Results and Observations

| Aspect | Without KL | With KL |
| Latent space | Discontinuous | Smooth & continuous |
| Reconstruction | Sharp | Slightly blurry |
| Sampling | Poor | Meaningful |
| Generative ability | Low | High |

**Key Observation:**  
KL divergence trades reconstruction sharpness for latent space smoothness, which is essential for generative modeling.

All outputs are compressed into a ZIP file for easy submission.


## 11. Conclusion

This assignment demonstrates that **KL divergence is the key component that transforms an autoencoder into a true generative model**.  
Without KL divergence, the latent space becomes fragmented and unsuitable for sampling.  
With KL divergence, the latent space becomes smooth and continuous, enabling realistic image generation.


## 12. Technologies Used

- Python  
- PyTorch  
- Torchvision  
- Matplotlib  


## 13. How to Run

1. Install dependencies:
   ```bash
   pip install torch torchvision matplotlib