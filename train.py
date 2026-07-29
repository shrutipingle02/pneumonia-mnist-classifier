# core imports: numpy for arrays, torch for building/training the model,
# DataLoader for batching data, transforms for preprocessing images
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from medmnist import PneumoniaMNIST

# use GPU if available, otherwise fall back to CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# preprocessing: convert images to tensors, then normalize pixel values
# to roughly [-1, 1] — helps the model train faster and more stably
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# load the three splits: train (to learn from), val (to check progress),
# test (final, unseen accuracy check) — download=True grabs the data automatically
train_dataset = PneumoniaMNIST(split='train', transform=data_transform, download=True)
val_dataset = PneumoniaMNIST(split='val', transform=data_transform, download=True)
test_dataset = PneumoniaMNIST(split='test', transform=data_transform, download=True)

# sanity check — confirm how many images are in each split
print(f'train: {len(train_dataset)} | val: {len(val_dataset)} | test: {len(test_dataset)}')

# wrap each dataset in a DataLoader — batches of 64 images at a time,
# shuffled for training so the model doesn't memorize order
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
