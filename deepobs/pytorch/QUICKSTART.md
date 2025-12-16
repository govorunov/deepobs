# DeepOBS PyTorch - Quick Start Guide

This guide shows how to use the PyTorch implementation of DeepOBS datasets.

---

## Installation

```bash
# Clone the repository
git clone https://github.com/fsschneider/DeepOBS.git
cd DeepOBS

# Install PyTorch (if not already installed)
pip install torch torchvision

# Install DeepOBS
pip install -e .
```

---

## Basic Usage

### 1. Import Datasets

```python
from deepobs.pytorch.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
```

### 2. Create a Dataset Instance

```python
# MNIST with batch size 32
dataset = MNIST(batch_size=32)

# CIFAR-10 with batch size 128 and data augmentation
dataset = CIFAR10(batch_size=128, data_augmentation=True)
```

### 3. Iterate Over Data

```python
# Training loop
for epoch in range(num_epochs):
    for images, labels in dataset.train_loader:
        # images: (batch_size, channels, height, width) float32 tensor
        # labels: (batch_size,) int64 tensor with class indices

        # Your training code here
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## Complete Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from deepobs.pytorch.datasets import MNIST

# 1. Create dataset
dataset = MNIST(batch_size=64, train_eval_size=5000)

# 2. Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = SimpleNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training loop
num_epochs = 5
for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    for images, labels in dataset.train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Evaluation on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in dataset.test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {train_loss/len(dataset.train_loader):.4f}")
    print(f"  Test Loss: {test_loss/len(dataset.test_loader):.4f}")
    print(f"  Test Accuracy: {accuracy:.2f}%")
```

---

## Dataset Reference

### MNIST

```python
from deepobs.pytorch.datasets import MNIST

dataset = MNIST(
    batch_size=32,           # Mini-batch size
    train_eval_size=10000,   # Size of training evaluation subset
    num_workers=4            # Number of data loading workers
)

# Data loaders
dataset.train_loader       # Training data (shuffled)
dataset.train_eval_loader  # Training eval data (not shuffled, subset)
dataset.test_loader        # Test data (not shuffled)

# Properties
dataset.batch_size         # Returns 32
```

**Data Format**:
- Images: `(batch_size, 1, 28, 28)` float32, range [0, 1]
- Labels: `(batch_size,)` int64, values [0-9]

---

### Fashion-MNIST

```python
from deepobs.pytorch.datasets import FashionMNIST

dataset = FashionMNIST(batch_size=64)
```

**Data Format**:
- Images: `(batch_size, 1, 28, 28)` float32, range [0, 1]
- Labels: `(batch_size,)` int64, values [0-9]

**Classes**: T-shirt, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

### CIFAR-10

```python
from deepobs.pytorch.datasets import CIFAR10

dataset = CIFAR10(
    batch_size=128,
    data_augmentation=True,   # Enable data augmentation (training only)
    train_eval_size=10000,
    num_workers=4
)
```

**Data Format**:
- Images: `(batch_size, 3, 32, 32)` float32, per-image standardized
- Labels: `(batch_size,)` int64, values [0-9]

**Data Augmentation** (training only):
- Random crop (pad to 36×36, crop to 32×32)
- Random horizontal flip
- Color jitter (brightness, saturation, contrast)
- Per-image standardization

**Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

---

### CIFAR-100

```python
from deepobs.pytorch.datasets import CIFAR100

dataset = CIFAR100(
    batch_size=128,
    data_augmentation=True,
    train_eval_size=10000
)
```

**Data Format**:
- Images: `(batch_size, 3, 32, 32)` float32, per-image standardized
- Labels: `(batch_size,)` int64, values [0-99]

**Same augmentation as CIFAR-10**

---

## Advanced Usage

### Custom Training Evaluation Size

```python
# Use only 1000 training samples for evaluation
dataset = MNIST(batch_size=32, train_eval_size=1000)

# Evaluate on training subset
for images, labels in dataset.train_eval_loader:
    # Only 1000 samples will be used
    pass
```

### Data Augmentation Control

```python
# CIFAR-10 without augmentation
dataset = CIFAR10(batch_size=128, data_augmentation=False)

# CIFAR-10 with augmentation (default)
dataset = CIFAR10(batch_size=128, data_augmentation=True)
```

### Parallel Data Loading

```python
# Use 8 worker processes for faster data loading
dataset = CIFAR10(batch_size=128, num_workers=8)
```

---

## GPU Usage

```python
import torch
from deepobs.pytorch.datasets import CIFAR10

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create dataset
dataset = CIFAR10(batch_size=128)

# Move model to GPU
model = model.to(device)

# Training loop
for images, labels in dataset.train_loader:
    # Move data to GPU
    images = images.to(device)
    labels = labels.to(device)

    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

**Note**: DataLoaders automatically use `pin_memory=True` when CUDA is available for faster data transfer.

---

## Configuration

### Data Directory

```python
from deepobs.pytorch import config

# Get current data directory
data_dir = config.get_data_dir()
print(f"Data stored in: {data_dir}")

# Set custom data directory
config.set_data_dir("/path/to/my/data")

# Now all datasets will use this directory
dataset = MNIST(batch_size=32)
```

### Default Dtype

```python
from deepobs.pytorch import config

# Get current dtype
dtype = config.get_dtype()  # torch.float32 by default

# Set to float64 for higher precision
config.set_dtype(torch.float64)
# or
config.set_dtype('float64')
```

---

## Troubleshooting

### Import Error

```python
# If you get: ModuleNotFoundError: No module named 'deepobs.pytorch'
# Make sure you've installed the package:
pip install -e .
```

### PyTorch Not Found

```bash
# Install PyTorch and torchvision
pip install torch torchvision
```

### Dataset Download Issues

```python
# Datasets are automatically downloaded on first use
# They will be stored in the data directory (default: 'data')
# Make sure you have internet connection and write permissions
```

### Out of Memory

```python
# Reduce batch size
dataset = CIFAR10(batch_size=64)  # Instead of 128

# Reduce number of workers
dataset = CIFAR10(batch_size=128, num_workers=2)  # Instead of 4
```

---

## Differences from TensorFlow Implementation

| Aspect | TensorFlow | PyTorch |
|--------|------------|---------|
| Data Format | NHWC (batch, height, width, channels) | NCHW (batch, channels, height, width) |
| Label Format | One-hot vectors | Class indices |
| Phase Switching | `train_init_op` / `test_init_op` | Separate DataLoaders |
| Iteration | `sess.run()` | Python iterator (`for` loop) |

---

## What's Next?

- **Phase 3**: Simple neural network architectures (logreg, mlp, 2c2d, 3c3d)
- **Phase 4**: Training runner for orchestrating experiments
- **Phase 5**: Remaining datasets (SVHN, ImageNet, Tolstoi, etc.)
- **Phase 6**: Advanced architectures (VGG, ResNet, Inception, VAE)

---

## Getting Help

- **Documentation**: See `deepobs/pytorch/datasets/README.md`
- **Tests**: See `tests/test_pytorch_datasets.py` for usage examples
- **Original Paper**: https://openreview.net/forum?id=rJg6ssC5Y7
- **GitHub Issues**: https://github.com/fsschneider/DeepOBS/issues

---

**Last Updated**: 2025-12-13
