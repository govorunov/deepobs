# DeepOBS - A Deep Learning Optimizer Benchmark Suite

----

> **Note**: The original TensorFlow-only version is no longer maintained and has been superseded by the [AlgoPerf benchmark suite](https://github.com/mlcommons/algorithmic-efficiency).
>
> **NEW**: DeepOBS now supports **PyTorch**! Version 1.2.0 includes a complete PyTorch implementation alongside the original TensorFlow version.

-----

![DeepOBS](docs/deepobs_banner.png "DeepOBS")

[![PyPI version](https://badge.fury.io/py/deepobs.svg)](https://badge.fury.io/py/deepobs)
[![Documentation Status](https://readthedocs.org/projects/deepobs/badge/?version=stable)](https://deepobs.readthedocs.io/en/stable/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-ee4c2c.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-1.4+-ff6f00.svg)


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

## PyTorch Support 🔥

**DeepOBS now supports PyTorch!** Version 1.2.0 includes a complete PyTorch implementation with all 26 test problems.

### Quick Start (PyTorch)

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

### PyTorch Installation

```bash
# Install DeepOBS with PyTorch support
pip install deepobs[pytorch]

# Or install manually
pip install deepobs
pip install torch torchvision
```

### Available Test Problems (PyTorch)

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

### PyTorch Documentation

- **[README_PYTORCH.md](README_PYTORCH.md)** - Complete PyTorch usage guide
- **[docs/MIGRATION_GUIDE.md](docs/MIGRATION_GUIDE.md)** - Migrate from TensorFlow to PyTorch
- **[docs/API_REFERENCE.md](docs/API_REFERENCE.md)** - Full API documentation
- **[docs/KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)** - Known issues and limitations
- **[examples/](examples/)** - Practical runnable examples

---

## Features

- **26 Test Problems**: Realistic deep learning benchmarks across multiple domains
- **9 Architectures**: MLPs, CNNs, ResNets, VGG, Inception, VAE, RNN
- **9 Datasets**: MNIST, CIFAR, ImageNet, and more
- **Dual Framework Support**: Both TensorFlow and PyTorch
- **Automated Benchmarking**: Run experiments with minimal code
- **Baseline Comparisons**: Compare against established optimizers
- **Publication-Ready Plots**: Automatic visualization and analysis

## Installation

### Basic Installation

```bash
pip install deepobs
```

### With Framework Support

```bash
# PyTorch support (recommended)
pip install deepobs[pytorch]

# TensorFlow support
pip install deepobs[tensorflow]

# Both frameworks
pip install deepobs[all]

# Development installation
pip install deepobs[dev]
```

### From Source

```bash
# Clone repository
git clone https://github.com/fsschneider/DeepOBS.git
cd DeepOBS

# Install in development mode
pip install -e .

# With PyTorch
pip install -e ".[pytorch]"
```

## Quick Start

### PyTorch Example

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

### TensorFlow Example

```python
import tensorflow as tf
from deepobs.tensorflow import testproblems

# Create test problem
tp = testproblems.cifar10_3c3d(batch_size=128)
tp.set_up()

# Create optimizer
optimizer = tf.train.MomentumOptimizer(learning_rate=0.1, momentum=0.9)

# Training step
train_op = optimizer.minimize(tp.loss)

# Run training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(100):
        # ... training loop ...
```

## Documentation

- **[README_PYTORCH.md](README_PYTORCH.md)** - PyTorch implementation guide
- **[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md)** - TensorFlow to PyTorch migration
- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete API documentation
- **[EXAMPLES.md](EXAMPLES.md)** - Usage examples and tutorials
- **[KNOWN_ISSUES.md](KNOWN_ISSUES.md)** - Known issues and limitations

**TensorFlow Documentation**: https://deepobs.readthedocs.io/

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

### PyTorch Version
- Python >= 3.6
- PyTorch >= 1.9.0
- torchvision >= 0.10.0
- NumPy >= 1.19.0
- Pandas >= 1.1.0
- Matplotlib >= 3.3.0
- Seaborn >= 0.11.0

### TensorFlow Version
- Python >= 3.6
- TensorFlow >= 1.4.0 (tested with 1.12)
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Project Status

- **TensorFlow Version**: Stable but no longer actively maintained
- **PyTorch Version**: Actively maintained (v1.2.0)
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
- **[TensorFlow Models](https://github.com/tensorflow/models)** - TensorFlow model implementations
