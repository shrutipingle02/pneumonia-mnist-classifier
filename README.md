# PneumoniaMNIST Classifier

A deep learning project that uses a Convolutional Neural Network (CNN) built with PyTorch to classify chest X-ray images as **Normal** or **Pneumonia** using the PneumoniaMNIST (MedMNIST v2) dataset.

---

## Overview

This project demonstrates an end-to-end medical image classification pipeline, including data preprocessing, model training,evaluation and prediction using PyTorch.

---

## Dataset

- **Name:** PneumoniaMNIST (MedMNIST v2)
- **Image Size:** 28 × 28 grayscale
- **Classes:** Normal, Pneumonia

**Dataset:** https://medmnist.com/

---

## Project Workflow

1. Prepare the PneumoniaMNIST dataset.
2. Preprocess chest X-ray images.
3. Train a Convolutional Neural Network (CNN).
4. Evaluate the trained model.
5. Generate predictions on unseen test images.
6. Save the trained model for future inference.

---

## Tech Stack

- Python
- PyTorch
- Torchvision
- MedMNIST
- NumPy
- Matplotlib

---

## Model

A custom CNN built with PyTorch for binary chest X-ray classification.

---

## Results

- Successfully trained a CNN for binary chest X-ray classification.
- Saved the trained model as `pneumonia_model.pth`.
- Generated sample predictions on unseen test images (`sample_predictions.png`).
- Demonstrated an end-to-end deep learning workflow using PyTorch.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python train.py
```

---

## Future Work

- Improve model accuracy through hyperparameter tuning.
- Apply data augmentation to improve generalization.
- Experiment with deeper CNN architectures and pretrained models (e.g., ResNet18).
- Report additional evaluation metrics.


---

## Contact and Feedback


**Data Analyst:** Shruti Pingle

**LinkedIn:** [Profile](https://www.linkedin.com/in/shruti-pingle-aa8034196)

**Email:** [shrutipingle02\@gmail.com](mailto:shrutipingle02@gmail.com)


