# DeepOBS - A Deep Learning Optimizer Benchmark Suite

----

> **Note**: DeepOBS is now a **PyTorch-only** project. All TensorFlow code has been removed. For large-scale benchmarking, see the [AlgoPerf benchmark suite](https://github.com/mlcommons/algorithmic-efficiency).

-----

![DeepOBS](docs/deepobs_banner.png "DeepOBS")

[![PyPI version](https://badge.fury.io/py/deepobs.svg)](https://badge.fury.io/py/deepobs)
[![Documentation Status](https://readthedocs.org/projects/deepobs/badge/?version=stable)](https://deepobs.readthedocs.io/en/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)


**DeepOBS** is a benchmarking suite that drastically simplifies, automates and
improves the evaluation of deep learning optimizers.

It can evaluate the performance of new optimizers on a variety of
**real-world test problems** and automatically compare them with
**realistic baselines**.

DeepOBS automates several steps when benchmarking deep learning optimizers:

  - Downloading and preparing data sets.
  - Setting up test problems consisting of contemporary data sets and realistic
    deep learning architectures.
  - Running the optimizers on multiple test problems and logging relevant
    metrics.
  - Reporting and visualizing the results of the optimizer benchmark.

![DeepOBS Output](docs/deepobs.jpg "DeepOBS_output")

## Table of Contents

- [Quick Start](#quick-start)
- [Available Test Problems](#available-test-problems)
- [Features](#features)
- [Installation](#installation)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [FAQ](#faq)
- [Documentation](#documentation)
- [Paper](#paper)
- [Requirements](#requirements)
- [Contributing](#contributing)

## Quick Start

```python
import torch
from deepobs.pytorch import testproblems

# Create a test problem
tproblem = testproblems.mnist_mlp(batch_size=128)
tproblem.set_up()

# Train with any PyTorch optimizer
optimizer = torch.optim.Adam(tproblem.model.parameters(), lr=0.001)

for epoch in range(10):
    # Training loop
    for batch in tproblem.train_loader:
        optimizer.zero_grad()
        losses, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
        loss = losses.mean()
        loss.backward()
        optimizer.step()
```

## Available Test Problems

All 26 test problems are available:

| Dataset | Problems |
|---------|----------|
| **MNIST** | `mnist_logreg`, `mnist_mlp`, `mnist_2c2d`, `mnist_vae` |
| **Fashion-MNIST** | `fmnist_logreg`, `fmnist_mlp`, `fmnist_2c2d`, `fmnist_vae` |
| **CIFAR-10** | `cifar10_3c3d`, `cifar10_vgg16`, `cifar10_vgg19` |
| **CIFAR-100** | `cifar100_3c3d`, `cifar100_allcnnc`, `cifar100_vgg16`, `cifar100_vgg19`, `cifar100_wrn404` |
| **SVHN** | `svhn_3c3d`, `svhn_wrn164` |
| **ImageNet** | `imagenet_vgg16`, `imagenet_vgg19`, `imagenet_inception_v3` |
| **Text** | `tolstoi_char_rnn` |
| **Synthetic** | `quadratic_deep`, `two_d_rosenbrock`, `two_d_beale`, `two_d_branin` |

### Problem Categories

**Classification Problems**: mnist_*, fmnist_*, cifar10_*, cifar100_*, svhn_*, imagenet_*
**Generative Models**: mnist_vae, fmnist_vae
**Sequential Models**: tolstoi_char_rnn
**Optimization Benchmarks**: quadratic_deep, two_d_*

---

## Features

- **26 Test Problems**: Realistic deep learning benchmarks across multiple domains
- **9 Architectures**: MLPs, CNNs, ResNets, VGG, Inception, VAE, RNN
- **9 Datasets**: MNIST, CIFAR, ImageNet, and more
- **PyTorch Implementation**: Modern PyTorch-based framework
- **Automated Benchmarking**: Run experiments with minimal code
- **Baseline Comparisons**: Compare against established optimizers
- **Publication-Ready Plots**: Automatic visualization and analysis
- **Per-Example Losses**: Access individual sample losses for advanced optimizer development

## Installation

### Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package manager. Install it first:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

Then install DeepOBS:

```bash
# Basic installation
uv pip install deepobs

# PyTorch support (recommended)
uv pip install deepobs[pytorch]

# All dependencies
uv pip install deepobs[all]

# Development installation
uv pip install deepobs[dev]
```

### From Source with UV

```bash
# Clone repository
git clone https://github.com/fsschneider/DeepOBS.git
cd DeepOBS

# Create virtual environment
uv venv

# Activate virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Install in development mode
uv pip install -e .

# With PyTorch
uv pip install -e ".[pytorch]"

# Sync all dependencies including dev
uv pip install -e ".[all,dev]"
```

### Using pip

```bash
# Basic installation
pip install deepobs

# PyTorch support (recommended)
pip install deepobs[pytorch]

# All dependencies
pip install deepobs[all]

# Development installation
pip install deepobs[dev]
```

### From Source with pip

```bash
# Clone repository
git clone https://github.com/fsschneider/DeepOBS.git
cd DeepOBS

# Install in development mode
pip install -e .

# With PyTorch
pip install -e ".[pytorch]"
```

## Usage Examples

### Basic Training

```python
import torch
from deepobs.pytorch import testproblems

# Create test problem
tp = testproblems.cifar10_3c3d(batch_size=128)
tp.set_up()

# Use with any optimizer
optimizer = torch.optim.SGD(tp.model.parameters(), lr=0.1, momentum=0.9)

# Training loop
for epoch in range(100):
    for batch in tp.train_loader:
        optimizer.zero_grad()
        losses, accuracy = tp.get_batch_loss_and_accuracy(batch)
        loss = losses.mean() + tp.get_regularization_loss()
        loss.backward()
        optimizer.step()
```

### GPU Training

```python
import torch
from deepobs.pytorch import testproblems

# Specify GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create test problem with GPU
tproblem = testproblems.cifar10_3c3d(batch_size=128, device=device)
tproblem.set_up()

# Optimizer and training loop
optimizer = torch.optim.Adam(tproblem.model.parameters(), lr=1e-3)

for epoch in range(100):
    tproblem.model.train()
    for batch in tproblem.train_loader:
        # Data is automatically moved to device by test problem
        optimizer.zero_grad()
        loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
        loss.mean().backward()
        optimizer.step()
```

### Using StandardRunner

The `StandardRunner` automates the entire training workflow:

```python
import torch.optim as optim
from deepobs.pytorch.runners import StandardRunner

# Define optimizer and hyperparameters
optimizer_class = optim.SGD
hyperparams = [
    {"name": "momentum", "type": float, "default": 0.0},
    {"name": "nesterov", "type": bool, "default": False}
]

# Create runner
runner = StandardRunner(optimizer_class, hyperparams)

# Run benchmark
runner.run(
    testproblem='mnist_mlp',
    batch_size=128,
    num_epochs=10,
    learning_rate=0.01,
    momentum=0.9,
    random_seed=42,
    output_dir='./results'
)
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import MultiStepLR

# Create test problem
tproblem = testproblems.cifar100_wrn404(batch_size=128)
tproblem.set_up()

# Create optimizer
optimizer = torch.optim.SGD(
    tproblem.model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

# Create scheduler: reduce LR by 0.1 at epochs 60, 120, 160
scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)

# Training loop
for epoch in range(200):
    # Train
    tproblem.model.train()
    for batch in tproblem.train_loader:
        optimizer.zero_grad()
        loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
        loss.mean().backward()
        optimizer.step()

    # Step scheduler at epoch end
    scheduler.step()

    print(f'Epoch {epoch}, LR: {scheduler.get_last_lr()[0]:.6f}')
```

### Accessing Per-Example Losses

DeepOBS returns per-example losses for advanced optimizer development:

```python
tproblem = testproblems.mnist_mlp(batch_size=128)
tproblem.set_up()

optimizer = torch.optim.SGD(tproblem.model.parameters(), lr=0.01)

for batch in tproblem.train_loader:
    optimizer.zero_grad()

    # Get per-example losses (shape: [batch_size])
    losses, accuracy = tproblem.get_batch_loss_and_accuracy(batch)

    # Can use individual losses for custom weighting, sampling, etc.
    print(f'Per-example losses shape: {losses.shape}')
    print(f'Loss statistics: min={losses.min():.4f}, '
          f'max={losses.max():.4f}, mean={losses.mean():.4f}')

    # Compute mean for standard training
    loss = losses.mean()
    loss.backward()
    optimizer.step()
```

## Configuration

DeepOBS uses a global configuration system for data and baseline directories.

### Setting Data Directory

```python
from deepobs.pytorch import config

# Set custom data directory
config.set_data_dir('/path/to/data')

# Get current data directory
data_dir = config.get_data_dir()
print(f'Data directory: {data_dir}')
```

### Data Directory Structure

DeepOBS automatically downloads and organizes datasets:

```
<data_dir>/
├── mnist/
│   ├── MNIST/
│   │   └── raw/
├── fashion-mnist/
│   └── FashionMNIST/
├── cifar-10/
│   └── cifar-10-batches-py/
├── cifar-100/
│   └── cifar-100-python/
├── svhn/
│   ├── train_32x32.mat
│   └── test_32x32.mat
├── imagenet/
│   ├── train/  (user must provide)
│   └── val/    (user must provide)
└── tolstoi/
    └── war_and_peace.txt
```

**Note**: ImageNet requires manual download due to licensing. Place training images in `<data_dir>/imagenet/train/` and validation images in `<data_dir>/imagenet/val/`.

### Setting Data Type

```python
from deepobs.pytorch import config
import torch

# Use float32 (default)
config.set_dtype(torch.float32)

# Use float64 for higher precision
config.set_dtype(torch.float64)

# Get current dtype
dtype = config.get_dtype()
print(f'Data type: {dtype}')
```

## FAQ

### How do I use a specific GPU?

```python
import torch

# Use specific GPU
device = torch.device('cuda:0')  # GPU 0
tproblem = testproblems.mnist_mlp(batch_size=128, device=device)
tproblem.set_up()
```

### How do I ensure reproducibility?

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Can I use mixed precision training?

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in tproblem.train_loader:
    optimizer.zero_grad()

    with autocast():
        loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
        loss = loss.mean()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### How do I save and load checkpoints?

```python
# Save checkpoint
checkpoint = {
    'epoch': epoch,
    'model_state_dict': tproblem.model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}
torch.save(checkpoint, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
tproblem.model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']
```

### How do I handle out-of-memory errors?

- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Use gradient checkpointing for very deep models

```python
# Gradient accumulation example
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(tproblem.train_loader):
    loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
    loss = loss.mean() / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Can I access the dataset directly without test problems?

Yes! Datasets can be used independently:

```python
from deepobs.pytorch.datasets import MNIST
from torch.utils.data import DataLoader

# Create dataset
dataset = MNIST(batch_size=128)

# Access train/test loaders
for batch in dataset.train_loader:
    images, labels = batch
    # Your training code here
```

## Documentation

### Additional Resources

- **[README_PYTORCH.md](docs/README_PYTORCH.md)** - Detailed PyTorch usage guide with advanced topics
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** - API documentation
- **[examples/](examples/)** - Usage examples and tutorials
- **[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)** - Known issues and limitations
- **[docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)** - Migrating from TensorFlow

**Online Documentation**: https://deepobs.readthedocs.io/

### Example Scripts

The `examples/` directory contains complete, runnable examples:

- `basic_usage.py` - Simple end-to-end example
- `custom_optimizer_benchmark.py` - How to benchmark a custom optimizer
- `multiple_test_problems.py` - Running multiple test problems
- `learning_rate_schedule.py` - Using learning rate schedules
- `result_analysis.py` - Analyzing and plotting results

### Migrating from TensorFlow

If you're migrating from the TensorFlow version of DeepOBS, see [README_PYTORCH.md](docs/README_PYTORCH.md#migration-from-tensorflow) for detailed migration instructions including:

- API changes
- Training loop differences
- Batch normalization handling
- Common gotchas

## Paper

The DeepOBS paper was accepted at ICLR 2019:

**DeepOBS: A Deep Learning Optimizer Benchmark Suite**
Frank Schneider, Lukas Balles, Philipp Hennig
[https://openreview.net/forum?id=rJg6ssC5Y7](https://openreview.net/forum?id=rJg6ssC5Y7)

If you use DeepOBS in your research, please cite:

```bibtex
@inproceedings{schneider2019deepobs,
  title={DeepOBS: A Deep Learning Optimizer Benchmark Suite},
  author={Schneider, Frank and Balles, Lukas and Hennig, Philipp},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=rJg6ssC5Y7}
}
```

## Requirements

- Python >= 3.8
- PyTorch >= 1.9.0 (PyTorch >= 2.0 recommended)
- torchvision >= 0.10.0
- NumPy >= 1.19.0
- Pandas >= 1.1.0
- Matplotlib >= 3.3.0
- Seaborn >= 0.11.0

## Project Status

- **Current Version**: PyTorch-only (v1.2.0), actively maintained
- **Legacy**: TensorFlow code has been removed
- **Successor Project**: [AlgoPerf](https://github.com/mlcommons/algorithmic-efficiency) for large-scale benchmarking

## Contributing

We welcome contributions! Please see [CONTRIBUTORS.md](CONTRIBUTORS.md) for guidelines.

If you find any bugs or have suggestions, please:
1. Check existing [GitHub issues](https://github.com/fsschneider/DeepOBS/issues)
2. Create a new issue with details
3. Or contact: frank.schneider@tue.mpg.de

## License

DeepOBS is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

**Original DeepOBS Authors**:
- Frank Schneider
- Lukas Balles
- Philipp Hennig

**PyTorch Implementation**:
- Aaron Bahde (DeepOBS 1.2.0 development lead)
- PyTorch migration team (2025)

See [CONTRIBUTORS.md](CONTRIBUTORS.md) for full acknowledgments.

## Related Projects

- **[AlgoPerf](https://github.com/mlcommons/algorithmic-efficiency)** - MLCommons algorithmic efficiency benchmark
- **[TorchVision](https://github.com/pytorch/vision)** - PyTorch vision models and datasets

---

**Last Updated**: 2025-12-16
**Version**: 1.2.0-pytorch
**Framework**: PyTorch >= 1.9.0
