# Pneumonia Detection — PneumoniaMNIST

A PyTorch project that loads the PneumoniaMNIST dataset (chest X-ray images labeled pneumonia/normal) and prepares it for training a classifier.

## Setup

```bash
conda create -n pneumonia python=3.10
conda activate pneumonia
pip install -r requirements.txt
```

## Run

```bash
python train.py
```

This downloads the PneumoniaMNIST dataset automatically and prepares train/validation/test DataLoaders.

## Dataset
[PneumoniaMNIST](https://www.tensorflow.org/datasets/catalog/pneumonia_mnist) — pediatric chest X-ray images for binary pneumonia classification, part of the [MedMNIST](https://medmnist.com/) collection.

