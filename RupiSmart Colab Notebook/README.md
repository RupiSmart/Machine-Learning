# **RupiSmart Project**

## **💵 Currency Classification with Deep Learning**

Welcome to the RupiSmart Project, where state-of-the-art deep learning meets real-world challenges! This repository showcases a robust machine learning solution to classify Indonesian currency (both coins and banknotes) using computer vision. Whether you're tackling counterfeit detection, building an educational tool, or automating financial tasks, this project has you covered!

### 🚀 Features

- **Pretrained Model Integration:** Utilizes EfficientNet for feature extraction and fine-tuning, ensuring high accuracy and generalization.
- **Real-Time Inference:** Ready-to-deploy pipeline for classifying new images with confidence scores.
- **Augmentation Pipeline:** Handles real-world data variability using random flips, rotations, brightness adjustments, and more.
- **Robust Evaluation:** Includes confusion matrix, classification report, and visualization tools for thorough performance analysis.

### 📁 Dataset

- **A balanced dataset of 12 classes, including various denominations of Indonesian currency:**
  - Coins: 100COIN, 200COIN, 500COIN, 1000COIN
  - Banknotes: 1RIBU, 2RIBU, 5RIBU, 10RIBU, 20RIBU, 50RIBU, 75RIBU, 100RIBU
- **Preprocessed and augmented for real-world robustness:**
  - Image Size: 120x120 pixels

 ### 🧠 Model Architecture

- **Base Model:** Fine-tuned EfficientNetB0, leveraging pretrained weights from ImageNet.
- **Custom Layers:**
  - Fully connected dense layers for classification.
  - Dropout layers to prevent overfitting.
- **Output:** Softmax activation for multi-class classification across 12 classes.

### 📊 Model Performance

The model achieved impressive performance metrics on the test set:

- **Accuracy:** 99.0%
- **F1-Score:** 99.0%
- **Support:** 1296 samples

**Precision, Recall, and F1-Score Breakdown:**

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| 1000COIN    | 0.99      | 1.00   | 0.99     | 92      |
| 100COIN     | 0.98      | 0.99   | 0.99     | 108     |
| 100RIBU     | 1.00      | 1.00   | 1.00     | 87      |
| 10RIBU      | 1.00      | 1.00   | 1.00     | 119     |
| 1RIBU       | 1.00      | 1.00   | 1.00     | 94      |
| 200COIN     | 0.98      | 0.96   | 0.97     | 130     |
| 20RIBU      | 1.00      | 1.00   | 1.00     | 123     |
| 2RIBU       | 1.00      | 1.00   | 1.00     | 98      |
| 500COIN     | 0.98      | 0.99   | 0.99     | 105     |
| 50RIBU      | 1.00      | 1.00   | 1.00     | 114     |
| 5RIBU       | 1.00      | 1.00   | 1.00     | 123     |
| 75RIBU      | 1.00      | 1.00   | 1.00     | 103     |

### 🔍 Key Highlights

- **Training Accuracy:** 99.96%
- **Validation Accuracy:** 99.46%
- Implemented data **augmentation** to enhance model robustness.
- **Confusion Matrix** confirms consistent performance across all classes.

### 🚀 Getting Started

To reproduce the results:

**1. Clone this repository.**

**2. Install required dependencies: pip install -r requirements.txt.**

**3. Prepare the dataset and follow the training steps provided in the code.**
