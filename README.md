# Breed Classification of Cats and Dogs using Transformer Architecture

![Tech Stack](https://img.shields.io/badge/tech%20stack-Python%20%7C%20Transformers%20%7C%20Deep%20Learning-blue)

![Status](https://img.shields.io/badge/status-Project%20Complete-brightgreen)

## Overview

This project introduces a robust and scalable deep learning framework tailored for **fine-grained classification of cat and dog breeds**, a task that presents challenges due to high inter-class similarity (between breeds) and intra-class variation (within the same breed). The proposed pipeline leverages the **latest advancements in computer vision**, specifically **Transformer-based architectures** like **Vision Transformer (ViT)**, **Swin Transformer**, **DeiT**, **BEiT**, and **ConvNeXt**, to extract rich, high-level semantic features from pet images. These models are known for their ability to capture long-range dependencies and fine-grained details, enabling better differentiation between visually similar breeds.

The extracted features are fused and passed into several **deep neural classifiers**, with a focus on a customized **Residual Multi-Layer Perceptron (Residual MLP)** architecture. This classifier includes skip connections inspired by ResNet, which facilitate stable and efficient training of deeper models by preserving gradient flow.

To further enhance classification performance and generalization, we employ a **Genetic Algorithm (GA)** for **hyperparameter optimization**. This metaheuristic approach intelligently searches the hyperparameter space—tuning parameters such as learning rate, dropout rate, and weight decay—to find the most optimal configuration for training. This results in significantly improved validation accuracy while avoiding overfitting.

---

## Key Features

- Transformer-based feature extraction using:
  - ViT-B/16
  - Swin Transformer
  - BEiT
  - DeiT
  - ConvNeXt

- Classifier Architectures:
  - Basic MLP
  - Deep MLP
  - Residual MLP (Best Accuracy)
  - ResNet-style, EfficientNet-style, DenseNet-style classifiers

- Metaheuristic Optimization:
  - Genetic Algorithm (GA) for hyperparameter tuning

- Robust Evaluation:
  - Accuracy, Precision, Recall, F1-score, Confusion Matrix
  - CAM-based feature visualization
  - t-SNE plots for class separability
  - Edge-based shape and texture analysis

---

## Visual Architecture

![image](https://github.com/user-attachments/assets/5e35a3e4-8011-4e00-b3ae-1d72d4fd9c88)

*Figure: Proposed Feature Extraction and Classification Pipeline*

---

## Dataset

- **Oxford-IIIT Pet Dataset**
  - 7,349 images across 35 cat and dog breeds
  - Approx. 200 images per class
  - Includes bounding boxes, labels, and segmentation masks

---

## Tools and Technologies

- Python, PyTorch
- NumPy, Scikit-learn, Matplotlib, Seaborn
- DEAP (for Genetic Algorithm)
- Pretrained models from `timm` or `huggingface`

---

## How It Works

1. **Preprocessing**
   - Resize to 224×224
   - Normalize using ImageNet mean and std
   - Label encode breed classes
   
![image](https://github.com/user-attachments/assets/393c74b5-6841-4450-804c-7838ca29f685)

2. **Feature Extraction**
   - Use pretrained transformer backbones
   - Concatenate features to form a composite matrix of shape `(7349 × 4352)`

![image](https://github.com/user-attachments/assets/f1cd9b66-6466-4c8c-9f99-8e0565d4593b) 

3. **Classification**
   - Train classifiers on extracted features
   - Residual MLP yields highest validation accuracy (96.41%)
    
![image](https://github.com/user-attachments/assets/8d1a69c5-c4c8-4c68-87cc-3aeb8c14c50d)

4. **Optimization**
   - GA tunes learning rate, dropout, and weight decay
   - Best configuration:
     - LR: 0.00138 | Dropout: 0.2869 | WD: 0.00025

---

## Deep Learning Model Performance

| Classifier | Accuracy (%) |
|------------|--------------|
| Basic MLP | 96.14 |
| Deep MLP | 96.35 |
| **Residual MLP** | **96.35** [HIGHEST]|
| MLP + Dropout/BatchNorm | 96.14 |
| DenseNet-style | 96.32 |
| EfficientNet-style | 96.28 |
| ResNet-style | 95.94 |

---

## Evaluation Visuals

### Class Activation Maps (CAM)

![image](https://github.com/user-attachments/assets/e83a1a70-f608-45de-9849-a31e6f08f20f)

*Figure: Class Activation Map showing important features of an Abyssinian cat*

### t-SNE Plot

![image](https://github.com/user-attachments/assets/1ab1eff2-6e0b-475a-9d69-67330e7e64d0)

*Figure: Class-wise separability using t-SNE*

### Model Predictions

![image](https://github.com/user-attachments/assets/c0b85a53-7042-4164-ba18-daab92fc3b40)

*Prediction: Abyssinian | Confidence: 100.00%*

![image](https://github.com/user-attachments/assets/17b53a27-cc3e-44b8-9d52-8f020ef383a5)

*Prediction: Beagle | Confidence: 99.98%*

---

## Future Work

- AutoML tools (Optuna, Ray Tune)
- Expand to wildlife and ecological datasets
- Use of Explainable AI (Grad-CAM, SHAP, LIME)
- Ensemble learning and data augmentation
- Video-based temporal classification (CNN+LSTM)

---

## How to Run

```bash
git clone https://github.com/Bevinaa/Breed-Classification-of-Cats-and-Dogs-Using-Transformer-Architecture
cd Breed-Classification-of-Cats-and-Dogs-Using-Transformer-Architecture

# Run feature extraction
python extract_features.py

# Train classifier
python train_classifier.py

# Evaluate model
python evaluate_model.py

# Predict the breed
python predict_pet.py
```

## Contact

**Bevina R.**  
Email: bevina2110@gmail.com  
GitHub: [Bevinaa](https://github.com/Bevinaa)

---

