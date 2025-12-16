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

---

## Features

- **26 Test Problems**: Realistic deep learning benchmarks across multiple domains
- **9 Architectures**: MLPs, CNNs, ResNets, VGG, Inception, VAE, RNN
- **9 Datasets**: MNIST, CIFAR, ImageNet, and more
- **PyTorch Implementation**: Modern PyTorch-based framework
- **Automated Benchmarking**: Run experiments with minimal code
- **Baseline Comparisons**: Compare against established optimizers
- **Publication-Ready Plots**: Automatic visualization and analysis

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

## Usage Example

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

## Documentation

- **[README_PYTORCH.md](README_PYTORCH.md)** - Complete usage guide
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** - API documentation
- **[examples/](examples/)** - Usage examples and tutorials
- **[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)** - Known issues and limitations

**Online Documentation**: https://deepobs.readthedocs.io/

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
- PyTorch >= 1.9.0
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
