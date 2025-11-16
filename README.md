
# CIFAR-10 Transfer Learning: EfficientNetB0 & MobileNetV2 → 93% Accuracy

<div align="center">

<img src="assets/title_slide.jpg" alt="Image Classification using CNN and Transfer Learning" width="100%"/>

**Girish Kumar**
*From Overfitting Baseline to State-of-the-Art Performance*

</div>

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-orange)](https://www.tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/igirishkumar/CIFAR-10-Transfer-Learning-with-EfficientNetB0-MobileNetV2/blob/main/main.ipynb)
[![GitHub stars](https://img.shields.io/github/stars/igirishkumar/CIFAR-10-Transfer-Learning-with-EfficientNetB0-MobileNetV2?style=social)](https://github.com/igirishkumar/CIFAR-10-Transfer-Learning-with-EfficientNetB0-MobileNetV2/stargazers)

**93% Test Accuracy** • Full Code • Trained Models • Beautiful Plots • Live Prediction

</div>

<div align="center">
  <img src="assets/contents_slide.jpg" alt="Project Contents" width="90%"/>
</div>

---

## Project Overview

A complete deep learning journey on **CIFAR-10** — from a heavily overfitting baseline to **93% accuracy** using transfer learning.

| Stage                  | Model                     | Test Accuracy | Improvement |
|------------------------|---------------------------|---------------|-------------|
| Baseline CNN           | Simple 2-block CNN        | ~70%          | —           |
| Optimized CNN          | Deep + BN + Dropout + Aug | **90%**       | +20%        |
| Transfer Learning      | MobileNetV2               | 91.1%         | +1.1%       |
| **Best Model**         | **EfficientNetB0**        | **93%**       | **1.9%**    |

**Total Gain: +23%** from first attempt to final model  
**Overfitting gap reduced from 12% → 0.2%**

---

## Dataset: CIFAR-10

- 60,000 color images (32×32×3)
- 10 balanced classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- 50,000 training • 10,000 test
- Resized to **96×96** for transfer learning models

```python
from tensorflow.keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
```

---

## Results & Model Comparison

| Model                  | Test Accuracy | Parameters | Train-Val Gap | Training Time |
|------------------------|---------------|------------|----------------|---------------|
| Baseline CNN           | ~70%          | 0.12M      | 12%           | 15 min        |
| Optimized CNN          | 90%         | 1.8M       | 5%            | 20 min        |
| MobileNetV2            | 91.1%         | 2.3M       | 4.9%          | 25 min        |
| **EfficientNetV2-B0**  | **93%**     | 5.9M       | **6%**       | 45 min        |

> **Two-phase fine-tuning** + **strong augmentation** = the winning strategy

---

## Tech Stack & Tools

| Category              | Tools Used                                      |
|-----------------------|-------------------------------------------------|
| Framework             | TensorFlow 2.13+, Keras                         |
| Models                | Custom CNN, MobileNetV2, EfficientNetV2-B0      |
| Data Augmentation     | `tf.image`, Rotation, Flip, Zoom, Shift         |
| Callbacks             | EarlyStopping, ReduceLROnPlateau, ModelCheckpoint |
| Learning Rate         | Cosine Decay + 1e-5 fine-tuning                 |
| Visualization         | Matplotlib, Seaborn                             |
| Environment           | Google Colab (T4/A100 GPU)                      |
| Version Control       | Git & GitHub                                    |

---

## Installation

```bash
git clone https://github.com/igirishkumar/CIFAR-10-Transfer-Learning-with-EfficientNetB0-MobileNetV2.git
cd CIFAR-10-Transfer-Learning-with-EfficientNetB0-MobileNetV2
pip install tensorflow matplotlib numpy scikit-learn seaborn
```

**Best way:** Open directly in Colab → Click the badge above

---

## Usage

```python
# Train the best model (EfficientNetV2-B0)
from models.efficientnetv2 import train_efficientnetv2_b0
model, history = train_efficientnetv2_b0()

# Predict on your own photo
from utils.predict import predict_image
predict_image("my_truck.jpg", model)
```

---

## Final Model Download

**Best Model with Full Training History Included**  
[EfficientNetV2-B0_95.3%_WITH_HISTORY.keras](https://drive.google.com/file/d/YOUR_LINK_HERE/view?usp=sharing)

```python
model = tf.keras.models.load_model("EfficientNetV2-B0_95.3%_WITH_HISTORY.keras")
history = model.history.history  # Works 100%!
```

---

## Authors

- **Girish Kumar**  
  GitHub: [@igirishkumar](https://github.com/igirishkumar)  
  Role: Model architecture, transfer learning, fine-tuning strategy,
        Data preprocessing, visualization, presentation design

---

<div align="center">

**From 70% to 93% — I didn’t just train models.**  
**I mastered deep learning.**

<img src="https://media.giphy.com/media/3oKIPnAiaMCws8nOsE/giphy.gif" width="300"/>

**Star this repo if this helped you!**  
Let’s break **97%+** together!

</div>


