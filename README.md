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

The easiest way to use DeepOBS is through the command-line interface:

### 1. Run a Benchmark

```bash
# Run benchmark with default configuration
uv run deepobs benchmark

# Or with a custom configuration
uv run deepobs benchmark my_config.yaml
```

### 2. Analyze Results

```bash
# Generate interactive plots and statistics
uv run deepobs analyze

# Or analyze results from a custom directory
uv run deepobs analyze ./my_results
```

That's it! DeepOBS will run your benchmarks and generate an interactive HTML report with all results.

See [Quick Start Guide](docs/QUICK_START_BENCHMARK.md) for detailed instructions.

### Programmatic Usage

You can also use DeepOBS as a Python library:

```python
import torch
from deepobs.pytorch import testproblems

# Create a test problem
tproblem = testproblems.mnist_mlp(batch_size=128)
tproblem.set_up()

# Train with any PyTorch optimizer
optimizer = torch.optim.Adam(tproblem.model.parameters(), lr=0.001)

for epoch in range(10):
    for batch in tproblem.train_loader:
        optimizer.zero_grad()
        losses, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
        loss = losses.mean()
        loss.backward()
        optimizer.step()
```

For more programmatic examples, see [PyTorch Usage Guide](docs/README_PYTORCH.md).

## Available Test Problems

All **27 test problems** are currently implemented and available:

### MNIST (4 problems)
| Problem | Architecture | Parameters | Description |
|---------|-------------|------------|-------------|
| `mnist_logreg` | Logistic Regression | 7,850 | Single linear layer |
| `mnist_mlp` | Multi-Layer Perceptron | 1,134,410 | 4-layer fully connected |
| `mnist_2c2d` | 2Conv+2Dense | 2,949,120 | 2 conv + 2 dense layers |
| `mnist_vae` | Variational Autoencoder | ~500,000 | Encoder-decoder architecture |

### Fashion-MNIST (4 problems)
| Problem | Architecture | Parameters | Description |
|---------|-------------|------------|-------------|
| `fmnist_logreg` | Logistic Regression | 7,850 | Single linear layer |
| `fmnist_mlp` | Multi-Layer Perceptron | 1,134,410 | 4-layer fully connected |
| `fmnist_2c2d` | 2Conv+2Dense | 2,949,120 | 2 conv + 2 dense layers |
| `fmnist_vae` | Variational Autoencoder | ~500,000 | Encoder-decoder architecture |

### CIFAR-10 (3 problems)
| Problem | Architecture | Parameters | Description |
|---------|-------------|------------|-------------|
| `cifar10_3c3d` | 3Conv+3Dense | 1,411,850 | 3 conv + 3 dense layers |
| `cifar10_vgg16` | VGG-16 | 14,987,722 | 13 conv + 3 FC layers |
| `cifar10_vgg19` | VGG-19 | 20,040,522 | 16 conv + 3 FC layers |

### CIFAR-100 (5 problems)
| Problem | Architecture | Parameters | Description |
|---------|-------------|------------|-------------|
| `cifar100_3c3d` | 3Conv+3Dense | 1,461,700 | 3 conv + 3 dense layers (100 classes) |
| `cifar100_allcnnc` | All-CNN-C | 1,387,108 | All-convolutional network |
| `cifar100_vgg16` | VGG-16 | 15,002,212 | 13 conv + 3 FC layers |
| `cifar100_vgg19` | VGG-19 | 20,055,012 | 16 conv + 3 FC layers |
| `cifar100_wrn404` | Wide ResNet 40-4 | 8,952,420 | 40-layer ResNet, width factor 4 |

### SVHN (2 problems)
| Problem | Architecture | Parameters | Description |
|---------|-------------|------------|-------------|
| `svhn_3c3d` | 3Conv+3Dense | 1,411,850 | 3 conv + 3 dense layers |
| `svhn_wrn164` | Wide ResNet 16-4 | 2,748,218 | 16-layer ResNet, width factor 4 |

### ImageNet (3 problems)
| Problem | Architecture | Parameters | Description |
|---------|-------------|------------|-------------|
| `imagenet_vgg16` | VGG-16 | 138,357,544 | VGG-16 on ImageNet (1001 classes) |
| `imagenet_vgg19` | VGG-19 | 143,667,240 | VGG-19 on ImageNet (1001 classes) |
| `imagenet_inception_v3` | Inception V3 | ~27,000,000 | Multi-branch architecture |

**Note**: ImageNet problems require manual dataset download due to licensing restrictions.

### Text Generation (2 problems)
| Problem | Architecture | Parameters | Description |
|---------|-------------|------------|-------------|
| `textgen` | 2-Layer LSTM | ~500,000 | Penn Treebank (recommended) |
| `tolstoi_char_rnn` | 2-Layer LSTM | ~500,000 | War and Peace (deprecated) |

### Synthetic Optimization (4 problems)
| Problem | Parameters | Description |
|---------|------------|-------------|
| `quadratic_deep` | 100 | 100D quadratic with deep learning eigenspectrum |
| `two_d_rosenbrock` | 2 | Rosenbrock function (narrow valley) |
| `two_d_beale` | 2 | Beale function (flat regions) |
| `two_d_branin` | 2 | Branin function (multi-modal) |

### Problem Categories

- **Classification**: mnist_*, fmnist_*, cifar10_*, cifar100_*, svhn_*, imagenet_*
- **Generative Models**: mnist_vae, fmnist_vae
- **Sequential Models**: textgen (recommended), tolstoi_char_rnn (deprecated)
- **Optimization Benchmarks**: quadratic_deep, two_d_*

---

## Available Datasets

All **10 datasets** are currently implemented:

### Vision Datasets
- **MNIST**: 60,000 train / 10,000 test, 28×28 grayscale, 10 classes (digits)
- **Fashion-MNIST**: 60,000 train / 10,000 test, 28×28 grayscale, 10 classes (fashion items)
- **CIFAR-10**: 50,000 train / 10,000 test, 32×32 RGB, 10 classes
- **CIFAR-100**: 50,000 train / 10,000 test, 32×32 RGB, 100 classes
- **SVHN**: 73,257 train / 26,032 test, 32×32 RGB, 10 classes (street view house numbers)
- **ImageNet**: 1,281,167 train / 50,000 val, variable size→224×224, 1001 classes

### Text Datasets
- **Penn Treebank**: Text generation corpus for language modeling
- **Tolstoi**: War and Peace by Leo Tolstoy, 83 unique characters

### Synthetic Datasets
- **Quadratic**: Synthetic Gaussian samples for quadratic optimization
- **TwoD**: 2D noisy samples for optimization benchmarks

---

## Features

- **27 Test Problems**: Realistic deep learning benchmarks across multiple domains
- **9 Architectures**: MLPs, CNNs, ResNets, VGG, Inception, VAE, RNN
- **10 Datasets**: MNIST, Fashion-MNIST, CIFAR-10/100, SVHN, ImageNet, Penn Treebank, Tolstoi, and synthetic datasets
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

### Command-Line Interface

The recommended way to use DeepOBS:

```bash
# Run a benchmark suite
uv run deepobs benchmark my_config.yaml

# Analyze and visualize results
uv run deepobs analyze ./results
```

See [Quick Start Guide](docs/QUICK_START_BENCHMARK.md) for configuration examples.

### Programmatic API

DeepOBS can also be used as a Python library for custom training loops:

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
        loss = losses.mean()
        loss.backward()
        optimizer.step()
```

For detailed programmatic usage including GPU training, learning rate schedules, and advanced features, see [PyTorch Usage Guide](docs/README_PYTORCH.md).

## Configuration

DeepOBS uses YAML configuration files for benchmark setup. See [Quick Start Guide](docs/QUICK_START_BENCHMARK.md) for examples.

### Data Directory

DeepOBS automatically downloads most datasets. You can configure the data directory:

```python
from deepobs.pytorch import config
config.set_data_dir('/path/to/data')
```

**Note**: ImageNet requires manual download due to licensing. Place images in `<data_dir>/imagenet/train/` and `<data_dir>/imagenet/val/`.

For advanced configuration options, see [PyTorch Usage Guide](docs/README_PYTORCH.md).

## FAQ

### How do I run benchmarks?

Use the CLI:
```bash
uv run deepobs benchmark my_config.yaml
```

See [Quick Start Guide](docs/QUICK_START_BENCHMARK.md) for configuration examples.

### How do I analyze results?

```bash
uv run deepobs analyze ./results
```

This generates an interactive HTML report with plots and statistics.

### Can I use DeepOBS programmatically?

Yes! See [PyTorch Usage Guide](docs/README_PYTORCH.md) for detailed examples including:
- GPU training (CUDA, MPS)
- Learning rate schedules
- Mixed precision training
- Custom training loops
- Reproducibility
- Checkpointing

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
