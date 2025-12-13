# DeepOBS PyTorch Datasets

This directory contains PyTorch implementations of DeepOBS datasets.

## Implemented Datasets (Phase 2)

### MNIST (`mnist.py`)
- **Dataset**: Handwritten digits (0-9)
- **Size**: 60,000 training images, 10,000 test images
- **Image Shape**: 28x28 grayscale (output: 1×28×28 in NCHW format)
- **Normalization**: [0, 1] range (no standardization)
- **Augmentation**: None
- **Source**: `torchvision.datasets.MNIST`

**Usage**:
```python
from deepobs.pytorch.datasets import MNIST

dataset = MNIST(batch_size=32)

# Training loop
for images, labels in dataset.train_loader:
    # images: (32, 1, 28, 28) float32 tensor
    # labels: (32,) int64 tensor with class indices [0-9]
    pass
```

---

### Fashion-MNIST (`fmnist.py`)
- **Dataset**: Fashion items (10 categories)
- **Size**: 60,000 training images, 10,000 test images
- **Image Shape**: 28x28 grayscale (output: 1×28×28 in NCHW format)
- **Normalization**: [0, 1] range (no standardization)
- **Augmentation**: None
- **Source**: `torchvision.datasets.FashionMNIST`

**Usage**:
```python
from deepobs.pytorch.datasets import FashionMNIST

dataset = FashionMNIST(batch_size=64, train_eval_size=5000)

# Evaluation on training subset
for images, labels in dataset.train_eval_loader:
    # images: (64, 1, 28, 28) float32 tensor
    # labels: (64,) int64 tensor with class indices [0-9]
    pass
```

---

### CIFAR-10 (`cifar10.py`)
- **Dataset**: 10 object classes (airplane, automobile, bird, etc.)
- **Size**: 50,000 training images, 10,000 test images
- **Image Shape**: 32x32 RGB (output: 3×32×32 in NCHW format)
- **Normalization**: Per-image standardization (zero mean, unit variance)
- **Augmentation** (training only, when enabled):
  - Pad to 36×36, random crop to 32×32
  - Random horizontal flip
  - Color jitter (brightness, saturation, contrast)
- **Source**: `torchvision.datasets.CIFAR10`

**Usage**:
```python
from deepobs.pytorch.datasets import CIFAR10

# With augmentation (default)
dataset = CIFAR10(batch_size=128, data_augmentation=True)

# Without augmentation
dataset = CIFAR10(batch_size=128, data_augmentation=False)

for images, labels in dataset.train_loader:
    # images: (128, 3, 32, 32) float32 tensor
    # labels: (128,) int64 tensor with class indices [0-9]
    pass
```

---

### CIFAR-100 (`cifar100.py`)
- **Dataset**: 100 fine-grained object classes
- **Size**: 50,000 training images, 10,000 test images
- **Image Shape**: 32x32 RGB (output: 3×32×32 in NCHW format)
- **Normalization**: Per-image standardization (zero mean, unit variance)
- **Augmentation** (training only, when enabled):
  - Pad to 36×36, random crop to 32×32
  - Random horizontal flip
  - Color jitter (brightness, saturation, contrast)
- **Source**: `torchvision.datasets.CIFAR100`

**Usage**:
```python
from deepobs.pytorch.datasets import CIFAR100

dataset = CIFAR100(batch_size=128, data_augmentation=True, train_eval_size=10000)

for images, labels in dataset.test_loader:
    # images: (128, 3, 32, 32) float32 tensor
    # labels: (128,) int64 tensor with class indices [0-99]
    pass
```

---

## Key Differences from TensorFlow Implementation

1. **Label Format**: PyTorch datasets return class indices (int64) instead of one-hot vectors
2. **Data Format**: NCHW (channels-first) instead of NHWC (channels-last)
3. **No Phase Variable**: Train/eval mode handled automatically via DataLoader selection
4. **Automatic Downloads**: torchvision handles dataset downloading automatically
5. **Simpler API**: No `train_init_op` / `test_init_op` - just iterate over loaders

---

## Base Class API

All datasets inherit from `deepobs.pytorch.datasets.DataSet` and provide:

**Constructor Parameters**:
- `batch_size` (int): Mini-batch size
- `train_eval_size` (int, optional): Size of training evaluation subset
- `num_workers` (int): Number of data loading workers (default: 4)

**Attributes**:
- `train_loader` (DataLoader): Training data (shuffled, with augmentation if enabled)
- `train_eval_loader` (DataLoader): Training evaluation data (not shuffled, subset)
- `test_loader` (DataLoader): Test data (not shuffled, no augmentation)
- `batch_size` (int): The configured batch size

---

## Testing

Run unit tests with pytest:
```bash
pytest tests/test_pytorch_datasets.py -v
```

Run manual tests (requires PyTorch installed):
```bash
python tests/manual_test_datasets.py
```

---

## Implementation Status

- [x] MNIST
- [x] Fashion-MNIST
- [x] CIFAR-10
- [x] CIFAR-100
- [ ] SVHN (Phase 5)
- [ ] ImageNet (Phase 5)
- [ ] Tolstoi (Phase 5)
- [ ] Quadratic (Phase 5)
- [ ] Two-D (Phase 5)

---

**Last Updated**: 2025-12-13
