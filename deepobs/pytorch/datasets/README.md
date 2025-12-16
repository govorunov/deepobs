# DeepOBS PyTorch Datasets

This directory contains PyTorch implementations of DeepOBS datasets.

## Implemented Datasets (All Phases - COMPLETE)

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

---

### SVHN (`svhn.py`) - Phase 5
- **Dataset**: Street View House Numbers (digits 0-9)
- **Size**: 73,257 training images, 26,032 test images
- **Image Shape**: 32x32 RGB (output: 3×32×32 in NCHW format)
- **Normalization**: Per-image standardization (zero mean, unit variance)
- **Augmentation** (training only, when enabled):
  - Pad to 36×36, random crop to 32×32
  - Color jitter (brightness, saturation, contrast)
- **Source**: `torchvision.datasets.SVHN`

**Usage**:
```python
from deepobs.pytorch.datasets import SVHN

dataset = SVHN(batch_size=128, data_augmentation=True)

for images, labels in dataset.train_loader:
    # images: (128, 3, 32, 32) float32 tensor
    # labels: (128,) int64 tensor with class indices [0-9]
    pass
```

---

### ImageNet (`imagenet.py`) - Phase 5
- **Dataset**: Large-scale image classification (1000 classes)
- **Size**: ~1.28M training images, 50,000 validation images
- **Image Shape**: Resized to 224×224 RGB (output: 3×224×224 in NCHW format)
- **Normalization**: Per-image standardization (zero mean, unit variance)
- **Augmentation** (training only, when enabled):
  - Resize to 256px (aspect-preserving)
  - Random crop to 224×224
  - Random horizontal flip
- **Source**: `torchvision.datasets.ImageNet`
- **Note**: Requires manual download

**Usage**:
```python
from deepobs.pytorch.datasets import ImageNet

dataset = ImageNet(batch_size=64, data_augmentation=True)

for images, labels in dataset.train_loader:
    # images: (64, 3, 224, 224) float32 tensor
    # labels: (64,) int64 tensor with class indices [0-999]
    pass
```

---

### Tolstoi (`tolstoi.py`) - Phase 5
- **Dataset**: Character-level sequences from War and Peace
- **Size**: Pre-processed character sequences
- **Vocabulary**: 83 unique characters
- **Sequence Length**: 50 (configurable)
- **Format**: Character indices (int64)
- **Source**: Custom .npy files
- **Note**: Requires data preparation script

**Usage**:
```python
from deepobs.pytorch.datasets import Tolstoi

dataset = Tolstoi(batch_size=64, seq_length=50)

for x, y in dataset.train_loader:
    # x: (64, 50) int64 tensor with character indices
    # y: (64, 50) int64 tensor (x shifted by 1)
    pass
```

---

### Quadratic (`quadratic.py`) - Phase 5
- **Dataset**: Synthetic n-dimensional Gaussian samples
- **Size**: Configurable (default: 1000 samples)
- **Dimensions**: Configurable (default: 100)
- **Distribution**: N(0, noise_level²)
- **Purpose**: Quadratic optimization test problems
- **Source**: Generated on-the-fly with fixed seeds

**Usage**:
```python
from deepobs.pytorch.datasets import Quadratic

dataset = Quadratic(batch_size=32, dim=100, train_size=1000, noise_level=0.6)

for (x,) in dataset.train_loader:
    # x: (32, 100) float32 tensor with Gaussian samples
    pass
```

---

### Two-D (`two_d.py`) - Phase 5
- **Dataset**: Synthetic 2D Gaussian samples
- **Size**: Configurable (default: 10,000 samples)
- **Distribution**: Two independent N(0, noise_level²) variables
- **Purpose**: 2D optimization test functions (Rosenbrock, Beale, Branin)
- **Source**: Generated on-the-fly with fixed seeds
- **Note**: Test set uses zeros for deterministic evaluation

**Usage**:
```python
from deepobs.pytorch.datasets import TwoD

dataset = TwoD(batch_size=64, train_size=10000, noise_level=1.0)

for x, y in dataset.train_loader:
    # x: (64,) float32 tensor
    # y: (64,) float32 tensor
    pass
```

---

## Implementation Status

- [x] MNIST (Phase 2)
- [x] Fashion-MNIST (Phase 2)
- [x] CIFAR-10 (Phase 2)
- [x] CIFAR-100 (Phase 2)
- [x] SVHN (Phase 5)
- [x] ImageNet (Phase 5)
- [x] Tolstoi (Phase 5)
- [x] Quadratic (Phase 5)
- [x] Two-D (Phase 5)

**Status**: ✅ All 9 datasets implemented (100% complete)

---

**Last Updated**: 2025-12-14
