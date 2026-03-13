# Breast Ultrasound Image Segmentation using U-Net

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![macOS](https://img.shields.io/badge/macOS-Apple_Silicon_Optimized-lightgrey.svg)

## 📌 Project Overview
This repository contains a deep learning pipeline designed to automatically segment and outline breast tumors in ultrasound scans. Built from scratch using the Breast Ultrasound Images Dataset (BUSI), the project implements a custom **U-Net architecture** to translate raw medical imagery into precise, binary diagnostic masks. 

This project was developed with a focus on **high recall** to minimize false negatives, which is a critical safety requirement in medical AI diagnostics.

## 📂 Repository Structure
* **`Breast Ultrasound Image Segmentation using U-Net (keras).ipynb`**: The primary Jupyter Notebook containing the complete start-to-finish pipeline. This includes data loading, preprocessing, model construction, training, and evaluation.
* **`Breast Ultrasound Image Segmentation using U-Net (pytorch).py`**: The secondary PyTorch implementation utilizing Apple's Metal Performance Shaders (MPS) for hardware acceleration.
* **`unet_architecture.png`**: The generated visual graph of the U-Net layer structure.

## 📊 Dataset & Preprocessing
The model is trained on the **Breast Ultrasound Images Dataset (BUSI)**, which contains scans categorized as Normal, Benign, and Malignant. 

**Preprocessing Pipeline:**
1. **Standardization:** All images and ground-truth masks are resized to 128x128 pixels and normalized.
2. **Multi-Mask Handling:** Custom logic implemented to detect and mathematically merge multiple tumor masks belonging to a single ultrasound scan.
3. **OS Sanitization:** Strict `.png` file filtering to prevent training crashes caused by hidden macOS `.DS_Store` files.

## 🧠 Model Architecture
The project utilizes **U-Net**, a Convolutional Neural Network (CNN) specifically designed for biomedical image segmentation.
* **Encoder:** Extracts deep spatial features and context from the ultrasound.
* **Decoder:** Rebuilds the image resolution.
* **Skip Connections:** Transfers exact spatial boundaries directly from the encoder to the decoder, ensuring the predicted tumor shape is structurally accurate.

## 📈 Performance & Results
The primary Keras/TensorFlow model was evaluated using a 20% holdout test set. The model prioritizes Recall to ensure potential tumors are not missed.

* **Precision:** 62.2% *(Accuracy of positive tumor predictions)*
* **Recall:** 77.5% *(Success rate of finding all actual tumor pixels)*
* **F1-Score:** 69.0% *(Harmonic balance of precision and recall)*

*Note: An experimental PyTorch version was also developed to test framework performance differences, highlighting the necessity of advanced data augmentation when dealing with high-precision/low-recall local minima.*

## ⚙️ Local Setup & Installation

**1. Clone the repository**
```bash
git clone [https://github.com/abrarimtiyaz01/Medical-Image-Segmentation.git](https://github.com/abrarimtiyaz01/Medical-Image-Segmentation.git)
cd Medical-Image-Segmentation
