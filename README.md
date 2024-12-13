# **RupiSmart Project**

## **💵 Currency Classification and Authenticity Verification**

![3](https://github.com/user-attachments/assets/f8b337a0-cbee-456f-8498-a470a7b5c7eb)

Welcome to the RupiSmart Project! This repository provides solutions for two key tasks related to Indonesian currency:
1. **Currency Denomination Classification**: Identify the denomination of coins and banknotes.
2. **Authenticity Verification**: Detect counterfeit currency.

With these functionalities, the project combines cutting-edge machine learning with practical applications, including educational tools, automation of financial tasks, and enhanced counterfeit detection systems.

---

## 🚀 **Features**

### Denomination Classification
- **Pretrained Model Integration**: Utilizes EfficientNetB0 for robust feature extraction and fine-tuning.
- **Augmentation Pipeline**: Ensures real-world robustness through flips, rotations, brightness adjustments, etc.
- **Thorough Evaluation**: Confusion matrix, classification report, and visualization tools for performance analysis.

### Authenticity Verification
- **Custom CNN Architecture**: Tailored for binary classification of authentic and counterfeit currency.
- **Rescaling and Augmentation**: Standardized preprocessing with rescaling and data augmentation.
- **Model Performance Insights**: Detailed analysis of training, validation, and test results.

---

## 📁 **Datasets**

### Denomination Dataset
- **12 Classes**: Includes 100COIN, 200COIN, 500COIN, 1000COIN (coins) and 1RIBU to 100RIBU (banknotes).
- **Balanced Distribution**: 790 images per class.
- **Image Size**: Resized to 120x120 pixels for model compatibility.
- **Training-Validation Split**: 80-20 split with data augmentation applied to training.

### Authenticity Dataset
- **2 Classes**: Authentic and Counterfeit.
- **Data Distribution**: 154 images (77 authentic, 77 counterfeit).
- **Image Size**: Resized to 224x224 pixels for model compatibility.
- **Training-Validation Split**: 80-20 split with validation on unseen data.

---

## 🧠 **Model Architectures**

### Denomination Classification
- **Base Model**: Fine-tuned EfficientNetB0 pretrained on ImageNet.
- **Custom Layers**:
  - Fully connected dense layers for classification.
  - Dropout layers to prevent overfitting.
- **Output**: Softmax activation for multi-class classification across 12 classes.

### Authenticity Verification
- **Custom CNN**:
  - Convolutional layers with ReLU activation.
  - MaxPooling for down-sampling.
  - Dense layers for binary classification.
- **Output**: Sigmoid activation for authenticity detection.

---

## 📊 **Performance Metrics**

### Denomination Classification
- **Accuracy**: 99.0%
- **F1-Score**: 99.0%
- **Support**: 1895 samples.

| Class       | Precision | Recall | F1-Score | Support |
|-------------|-----------|--------|----------|---------|
| 1000COIN    | 0.99      | 1.00   | 1.00     | 151     |
| 100COIN     | 0.93      | 0.99   | 0.96     | 165     |
| 100RIBU     | 0.99      | 0.99   | 0.99     | 164     |
| 10RIBU      | 0.99      | 1.00   | 0.99     | 152     |
| 1RIBU       | 1.00      | 0.99   | 1.00     | 162     |
| 200COIN     | 0.98      | 0.91   | 0.94     | 158     |
| 20RIBU      | 1.00      | 1.00   | 1.00     | 158     |
| 2RIBU       | 0.99      | 1.00   | 1.00     | 160     |
| 500COIN     | 0.99      | 0.99   | 0.99     | 147     |
| 50RIBU      | 1.00      | 0.99   | 1.00     | 157     |
| 5RIBU       | 0.99      | 0.99   | 0.99     | 157     |
| 75RIBU      | 1.00      | 1.00   | 1.00     | 164     |
| **Accuracy**    |               |       | 0.99     | 1895     |
| **Macro avg**   | 0.99          | 0.99  | 0.99     | 1895     |
| **Weighted avg**| 0.99          | 0.99  | 0.99     | 1895     |

### Authenticity Verification
- **Training Accuracy**: 95.0%
- **Validation Accuracy**: 92.4%
- **Test Accuracy**: 96.7%

---

## 🔍 **Key Highlights**

- **Denomination Classification**:
  - Training Accuracy: 99.96%
  - Validation Accuracy: 99.46%.
  - Confusion Matrix confirms consistent performance across all classes.

- **Authenticity Verification**:
  - Rata-rata akurasi training: 95.0%.
  - Validation Akurasi: 92.4%.
  - Akurasi Pengujian: 96.7%.

---

## 🚀 **Getting Started**

### Denomination Classification
1. Clone this repository.
2. Install required dependencies: `pip install -r requirements.txt`.
3. Use the dataset for denomination.
4. Train the model using `model_denominatin.ipynb` and save results.

### Authenticity Verification
1. Clone this repository.
2. Use the dataset for authenticity.
3. Train the model using `model_autenticity.ipynb` and save results.

---

## 🎯 **Future Plans**
- Extend the dataset for better generalization.
- Improve real-time inference for both tasks.
- Explore ensemble models for increased robustness.

  
