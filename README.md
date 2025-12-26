# DeepOBS Pytorch - A Deep Learning Optimizer Benchmark Suite


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

Get started with DeepOBS in 5 minutes:

### 1. Install UV

[UV](https://github.com/astral-sh/uv) is a fast Python package manager:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 2. Clone and Setup

```bash
# Clone the repository
git clone THIS_REPO_URL
cd deepobs

# Sync dependencies (installs everything you need)
uv sync
```

### 3. Run a Benchmark

```bash
# Use an example configuration
uv run deepobs benchmark examples/benchmark_config_adamw_small.yaml

# Or create your own config (see below)
uv run deepobs benchmark my_config.yaml
```

### 4. Analyze Results

```bash
# Generate interactive plots and statistics
uv run deepobs analyze

# Or analyze results from a custom directory
uv run deepobs analyze ./my_results
```

### 5. Using Custom Optimizers

DeepOBS works with any PyTorch optimizer. If you have a custom optimizer package, install it first:

```bash
# Install your own optimizer package
uv pip install /path/to/your/optimizer

# Or install from PyPI
uv pip install your-optimizer-package
```

**Example configuration:**

```yaml
# my_config.yaml
global:
  output_dir: "./results"
  random_seed: 42
  num_epochs: 10

test_problems:
  - name: mnist_mlp
    batch_size: 128

  - name: cifar10_3c3d
    batch_size: 128
    num_epochs: 50

optimizers:
  # Built-in PyTorch optimizer (no optimizer_class needed)
  - name: Adam
    learning_rate: 0.001
    betas: [0.9, 0.999]
    eps: 1.0e-08

  # Custom optimizer - specify full import path
  - name: MyCustomOptimizer
    optimizer_class: my_package.optimizers.MyCustomOptimizer
    learning_rate: 0.01
    momentum: 0.9

# Optional: override settings per problem
overrides:
  mnist_mlp:
    Adam:
      learning_rate: 0.0001
```

**Using pytorch-optimizer package** (already included):

```yaml
optimizers:
  # Ranger optimizer from pytorch-optimizer
  - name: Ranger
    optimizer_class: torch_optimizer.Ranger
    learning_rate: 0.001

  # Lamb optimizer
  - name: Lamb
    optimizer_class: torch_optimizer.Lamb
    learning_rate: 0.001
    weight_decay: 0.01

  # AdaBound optimizer
  - name: AdaBound
    optimizer_class: torch_optimizer.AdaBound
    learning_rate: 0.001
    final_lr: 0.1
```

That's it! DeepOBS will run your benchmarks and generate an interactive HTML report with all results.

See [Quick Start Guide](docs/QUICK_START_BENCHMARK.md) for detailed instructions and more configuration examples.

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

### Text Generation (1 problem)
| Problem | Architecture | Parameters | Description |
|---------|-------------|------------|-------------|
| `ptb_lstm` | 2-Layer LSTM | ~500,000 | Penn Treebank |

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
- **Sequential Models**: ptb_lstm
- **Optimization Benchmarks**: quadratic_deep, two_d_*

---

## Available Datasets

All **9 datasets** are currently implemented:

### Vision Datasets
- **MNIST**: 60,000 train / 10,000 test, 28×28 grayscale, 10 classes (digits)
- **Fashion-MNIST**: 60,000 train / 10,000 test, 28×28 grayscale, 10 classes (fashion items)
- **CIFAR-10**: 50,000 train / 10,000 test, 32×32 RGB, 10 classes
- **CIFAR-100**: 50,000 train / 10,000 test, 32×32 RGB, 100 classes
- **SVHN**: 73,257 train / 26,032 test, 32×32 RGB, 10 classes (street view house numbers)
- **ImageNet**: 1,281,167 train / 50,000 val, variable size→224×224, 1001 classes

### Text Datasets
- **Penn Treebank**: Text generation corpus for language modeling

### Synthetic Datasets
- **Quadratic**: Synthetic Gaussian samples for quadratic optimization
- **TwoD**: 2D noisy samples for optimization benchmarks

---

## Features

- **25 Test Problems**: Realistic deep learning benchmarks across multiple domains
- **9 Architectures**: MLPs, CNNs, ResNets, VGG, Inception, VAE, RNN
- **9 Datasets**: MNIST, Fashion-MNIST, CIFAR-10/100, SVHN, ImageNet, Penn Treebank, and synthetic datasets
- **PyTorch Implementation**: Modern PyTorch-based framework
- **Automated Benchmarking**: Run experiments with minimal code
- **Baseline Comparisons**: Compare against established optimizers
- **Publication-Ready Plots**: Automatic visualization and analysis
- **Per-Example Losses**: Access individual sample losses for advanced optimizer development

## Installation

### Install UV

[UV](https://github.com/astral-sh/uv) is a fast Python package manager. Install it first:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### Install DeepOBS

```bash
# Navigate to repository
git clone ...
cd deepobs

# Create virtual environment
uv sync

uv run deepobs --help
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
- **[QUICK_START_BENCHMARK.md](docs/QUICK_START_BENCHMARK.md)** - CLI benchmarking guide
- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Complete API documentation
- **[examples/](examples/)** - Usage examples and configuration files
- **[KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)** - Known issues and limitations

**Online Documentation**: https://deepobs.readthedocs.io/

### Example Scripts

The `examples/` directory contains complete, runnable examples:

- `basic_usage.py` - Simple end-to-end example
- `custom_optimizer_benchmark.py` - How to benchmark a custom optimizer
- `multiple_test_problems.py` - Running multiple test problems
- `learning_rate_schedule.py` - Using learning rate schedules
- `result_analysis.py` - Analyzing and plotting results


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

## Project Status

- **Current Version**: PyTorch-only (v1.2.0)

This is a complete rewrite of the original DeepOBS project using Pytorch. 

## License

DeepOBS is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

**Original DeepOBS Authors**:
- Frank Schneider
- Lukas Balles
- Philipp Hennig

---

**Last Updated**: 2025-12-16
**Version**: 1.2.0-pytorch
**Framework**: PyTorch >= 1.9.0
