# Cat Species Generation & Classification

## Overview
This project explores the complete pipeline of **synthetic dataset generation** and **image classification** using deep learning. The work is divided into three main stages:

1. Synthetic image generation using a pre-trained generative model.
2. Image classification using both a custom CNN and a pre-trained ResNet model.
3. Comparative analysis of Zero-Shot, Few-Shot, and Continual Learning strategies.

The objective is to understand the limitations of training from scratch and the advantages of transfer learning.

---

## Project Files

### 1. dataset_generation.ipynb
- Generates a synthetic dataset of cat species using **Stable Diffusion**.
- Uses prompt-based image generation.
- Creates labeled folders for each cat species.
- This dataset is later used for all classification experiments.

---

### 2. CustomCNN.ipynb 
- Implements a **custom CNN with attention and residual connections**.
- Trained from scratch on the generated dataset.
- Serves as a **baseline model** to evaluate performance without prior knowledge.

**Accuracy achieved:**
- Custom CNN Accuracy: **20.24%**

This low accuracy highlights the difficulty of fine-grained classification when training from scratch on a synthetic dataset.

---

### 3. resnet_inference_accuracy.ipynb
- Uses **ResNet-18 pre-trained **.
- Performs inference without training on the cat species dataset.
- Accuracy is calculated using semantic matching.

**Accuracy achieved:**
- Zero-Shot Accuracy: **94.05%**

This shows the strong generalization capability of pre-trained models.

---

### 4. Few-Shot Learning
- Uses a small number of samples per class.
- Demonstrates rapid adaptation with limited data.

**Accuracy achieved:**
- Few-Shot Accuracy: **72.62%**

---

### 5. Continual Learning
- Trains the model incrementally by introducing new data in phases.
- Demonstrates the effect of learning new classes over time.

**Accuracy achieved:**
- Continual Learning Accuracy: **66.67%**

---

**Accuracy achieved:**
- Retrained ResNet Accuracy: **87%**

---

## Final Accuracy Comparison

Custom CNN Accuracy       : 20.24%
Zero-Shot Accuracy        : 94.05%
Few-Shot Accuracy         : 72.62%
Continual Learning Accuracy: 66.67%
Retrained ResNet Accuracy : 87%

---

## Model Details

- Generative Model: Stable Diffusion (Hugging Face)
- Classification Models:
  - Custom CNN (trained from scratch)
---

## Key Observations

- Training a CNN from scratch on a fine-grained dataset results in low accuracy.
- Zero-shot learning performs well due to strong pre-trained representations.
- Few-shot learning significantly improves accuracy with minimal data.
- Continual learning introduces challenges such as partial forgetting.

---

## Note
The dataset is synthetic and generated using a diffusion model. Accuracy values may vary depending on data quality, augmentation, and training configuration.


