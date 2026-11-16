# EuroSAT Land Cover Classification using Custom CNN

🚀 **Live Demo:** [Try the EuroSAT CNN Classifier](https://cvpr-cnn-eurosat-yqyd23ekjrevkcjtfxrrfc.streamlit.app)

## Overview
This project implements a custom Convolutional Neural Network (CNN) in PyTorch for land-cover classification using the EuroSAT satellite image dataset.

The goal is to classify satellite images into 10 different land-cover categories using an end-to-end deep learning pipeline including data preprocessing, augmentation, CNN training, validation, testing, and performance evaluation.

## Dataset
The EuroSAT dataset used in this project contains **27,000 RGB satellite images** of size **64 × 64 pixels**.

Dataset split:
- Training: 18,900 images
- Validation: 5,400 images
- Testing: 2,700 images

Classes:
1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

## Model
A custom CNN was built from scratch using PyTorch.

The network uses:
- Convolution layers
- Batch Normalization
- ReLU activation
- Max Pooling
- Dropout
- Fully connected layers

The convolutional blocks progressively increase the feature channels from 32 to 256.

## Training
Main training configuration:
- Image size: 64 × 64
- Batch size: 64
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: CrossEntropyLoss
- Maximum epochs: 20
- Early stopping patience: 5
- Learning-rate scheduler: ReduceLROnPlateau

Training data augmentation includes random horizontal/vertical flipping and random rotation.

## Results
The trained model achieved:

- **Test Accuracy: 94.33%**
- **Weighted Precision: 94.46%**
- **Weighted Recall: 94.33%**
- **Weighted F1-score: 94.35%**
- **Macro F1-score: 94.22%**

The model was further analyzed using a classification report, confusion matrix, training/validation curves, and example predictions.

## Repository Files

- `CNN_23_50712_1.ipynb` — complete implementation, training, and evaluation
- `EuroSAT_CustomCNN_weights.pth` — trained model weights
- `README.md` — project summary and documentation

## Technologies
Python, PyTorch, Torchvision, NumPy, Pandas, Matplotlib, Scikit-learn, Pillow, Google Colab.

## Conclusion
This project demonstrates a complete deep-learning workflow for multi-class satellite image classification. The custom CNN achieved strong performance on the EuroSAT test set without using a pretrained architecture.

Possible future improvements include transfer learning with models such as ResNet or EfficientNet, hyperparameter optimization, and Grad-CAM-based model explainability.
