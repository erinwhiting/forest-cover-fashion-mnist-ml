# Forest Cover & Fashion MNIST — Neural Network & CNN Models (PyTorch, Google Colab)

A machine learning project built in Google Colab using Python, PyTorch, Pandas, and NumPy. This project implements multiple neural network architectures — including fully connected ANNs and convolutional neural networks — to classify both tabular and image datasets. Models were trained on the Forest Cover dataset and the Fashion MNIST dataset, demonstrating skills in data preprocessing, deep learning, GPU acceleration, and model evaluation.

---

## Features
- Load and preprocess real datasets using Pandas and NumPy
- Build fully connected neural networks (ANNs) using PyTorch
- Build convolutional neural networks (CNNs) for image classification
- Train models with GPU acceleration in Google Colab
- Visualize training curves (loss & accuracy)
- Achieve strong validation accuracy on both datasets
- Explore hyperparameter tuning and model stability
-Work with both tabular and image‑based machine learning workflows

--- 

## Tech Stack

### Languages & Tools
- Python
- Google Colab
- GPU (T4) acceleration

### Libraries
- PyTorch
- Pandas
- NumPy
- Matplotlib

### Datasets
- Forest Cover Dataset
- Fashion MNIST (CSV version)

---

## Model Summaries

### Forest Cover ANN Model
- Input: 10 numerical features
- Architecture: Multiple Linear + ReLU layers
- Validation Accuracy: ~0.75+
- Techniques:
  - Feature extraction
  - Label encoding
  - ANN construction with nn.Sequential
  - Model training using custom NNet wrapper
  - Training curve visualization

### Fashion MNIST ANN Model
- Input: 784 scaled pixel values (flattened 28×28 images)
- Architecture: Linear → ReLU → Linear → ReLU → Output
- Validation Accuracy: ~0.88+
- Techniques:
  - Feature scaling (0–1)
  - ANN classification
  - Hyperparameter tuning
  - GPU‑accelerated training

### Fashion MNIST CNN Model
- Input: 28×28 grayscale images reshaped to (1, 28, 28)
- Architecture:
  - Conv2D → ReLU → MaxPool
  - Conv2D → ReLU → MaxPool
  - Flatten → Linear → ReLU → Output
- Validation Accuracy: ~0.91+
- Techniques:
  - Convolutional layers
  - Pooling layers
  - Image tensor manipulation
  - GPU‑accelerated training
  - Model stability evaluation

---

## How to Run the Project

### 1. Open the Notebook in Google Colab
Upload the .ipynb file to Google Colab.

### 2. Install Dependencies
Run the setup cell:

```
!rm dsci_408 -r
!git clone https://github.com/drbeane/dsci_408.git
!wget -nc https://github.com/drbeane/datasets/raw/.../fashion-mnist_train.csv.zip
!unzip -n fashion-mnist_train.csv.zip
```

### 3. Import Libraries
```
python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from dsci_408.scripts.nnet import NNet
```

### 4. Enable GPU (Recommended for CNN)
Go to:
Runtime → Change runtime type → Hardware accelerator → GPU (T4)

### 5. Run Each Model Section
- Part 1: Forest Cover ANN
- Part 2: Fashion MNIST ANN
- Part 3: Fashion MNIST CNN

Each section includes:
- Data loading
- Preprocessing
- Model building
- Training
- Training curve visualization

--- 

## Future Improvements
- Add hyperparameter search (GridSearch or Optuna)
- Add dropout or batch normalization
- Add confusion matrices and classification reports
- Train on full Fashion MNIST dataset (non‑CSV version)
- Deploy models using Flask or FastAPI
- Convert notebooks into modular Python scripts
- Add model saving/loading with torch.save()
