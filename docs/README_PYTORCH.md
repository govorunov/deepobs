# DeepOBS PyTorch - Deep Learning Optimizer Benchmark Suite

**Version**: 1.2.0-pytorch
**Framework**: PyTorch
**Status**: Production-ready

DeepOBS PyTorch is a comprehensive benchmarking framework for deep learning optimizers, providing standardized test problems, automated evaluation, and publication-quality visualizations.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Available Test Problems](#available-test-problems)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Documentation](#documentation)
- [FAQ](#faq)

---

## Features

- **25 Test Problems**: Realistic deep learning benchmarks across multiple domains
- **8 Datasets**: MNIST, Fashion-MNIST, CIFAR-10/100, SVHN, ImageNet, and synthetic datasets
- **9 Architectures**: From simple logistic regression to Inception V3, VAE, and LSTMs
- **Automated Benchmarking**: Complete workflow from data loading to result visualization
- **Baseline Comparisons**: Compare against well-tuned SGD and Adam baselines
- **PyTorch Native**: Leverages PyTorch's modern features and simpler programming model
- **Reproducible**: Controlled random seeds and deterministic training

---

## Installation

### Prerequisites

- Python >= 3.6
- PyTorch >= 1.9.0 (PyTorch >= 2.0 recommended)
- torchvision >= 0.10.0

### Install DeepOBS

```bash
pip install deepobs
```

For the latest PyTorch version from source:

```bash
pip install -e git+https://github.com/fsschneider/DeepOBS.git@develop#egg=DeepOBS
```

### Verify Installation

```python
import deepobs.pytorch as pytorch_deepobs
print(pytorch_deepobs.__version__)
```

---

## Quick Start

Here's a simple example to benchmark an optimizer on MNIST:

```python
import torch
from deepobs.pytorch import testproblems

# Create a test problem
tproblem = testproblems.mnist_mlp(batch_size=128)
tproblem.set_up()

# Create optimizer
optimizer = torch.optim.SGD(
    tproblem.model.parameters(),
    lr=0.01,
    momentum=0.9
)

# Training loop
for epoch in range(10):
    # Training phase
    tproblem.model.train()
    for batch_idx, batch in enumerate(tproblem.dataset.train_loader):
        optimizer.zero_grad()

        # Get loss and accuracy
        loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)

        # Backward and step
        loss.mean().backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, '
                  f'Loss: {loss.mean().item():.4f}, '
                  f'Accuracy: {accuracy:.4f}')

    # Evaluation phase
    tproblem.model.eval()
    test_loss, test_acc = 0.0, 0.0
    with torch.no_grad():
        for batch in tproblem.dataset.test_loader:
            loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
            test_loss += loss.mean().item()
            test_acc += accuracy

    test_loss /= len(tproblem.dataset.test_loader)
    test_acc /= len(tproblem.dataset.test_loader)
    print(f'Epoch {epoch} - Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
```

---

## Available Test Problems

DeepOBS PyTorch provides **26 test problems** across 9 datasets and 9 architectures.

### Summary Table

| Test Problem | Dataset | Architecture | Parameters | Batch Size | Description |
|--------------|---------|--------------|------------|------------|-------------|
| **MNIST (4 problems)** |
| `mnist_logreg` | MNIST | Linear | 7,850 | 128 | Logistic regression baseline |
| `mnist_mlp` | MNIST | 4-layer MLP | 1,134,410 | 128 | Fully-connected network |
| `mnist_2c2d` | MNIST | 2 conv + 2 FC | 2,949,120 | 128 | Basic CNN |
| `mnist_vae` | MNIST | VAE | ~500k | 128 | Variational autoencoder |
| **Fashion-MNIST (4 problems)** |
| `fmnist_logreg` | Fashion-MNIST | Linear | 7,850 | 128 | Fashion items classification |
| `fmnist_mlp` | Fashion-MNIST | 4-layer MLP | 1,134,410 | 128 | Fully-connected network |
| `fmnist_2c2d` | Fashion-MNIST | 2 conv + 2 FC | 2,949,120 | 128 | Basic CNN |
| `fmnist_vae` | Fashion-MNIST | VAE | ~500k | 128 | Generative model |
| **CIFAR-10 (3 problems)** |
| `cifar10_3c3d` | CIFAR-10 | 3 conv + 3 FC | 1,411,850 | 128 | Deeper CNN |
| `cifar10_vgg16` | CIFAR-10 | VGG-16 | 14,987,722 | 128 | Classic deep CNN |
| `cifar10_vgg19` | CIFAR-10 | VGG-19 | 20,040,522 | 128 | Deeper VGG variant |
| **CIFAR-100 (5 problems)** |
| `cifar100_3c3d` | CIFAR-100 | 3 conv + 3 FC | 1,461,700 | 128 | Deeper CNN, 100 classes |
| `cifar100_allcnnc` | CIFAR-100 | All-CNN-C | 1,387,108 | 256 | All-convolutional network |
| `cifar100_vgg16` | CIFAR-100 | VGG-16 | 15,002,212 | 128 | Deep VGG, 100 classes |
| `cifar100_vgg19` | CIFAR-100 | VGG-19 | 20,055,012 | 128 | Deeper VGG, 100 classes |
| `cifar100_wrn404` | CIFAR-100 | WRN-40-4 | 8,952,420 | 128 | Wide ResNet |
| **SVHN (2 problems)** |
| `svhn_3c3d` | SVHN | 3 conv + 3 FC | 1,411,850 | 128 | Street view digits |
| `svhn_wrn164` | SVHN | WRN-16-4 | 2,748,218 | 128 | Wide ResNet |
| **ImageNet (3 problems)** |
| `imagenet_vgg16` | ImageNet | VGG-16 | 138,357,544 | 128 | Large-scale VGG |
| `imagenet_vgg19` | ImageNet | VGG-19 | 143,667,240 | 128 | Large-scale VGG |
| `imagenet_inception_v3` | ImageNet | Inception V3 | ~27M | 128 | Multi-branch architecture |
| **Text Generation (1 problem)** |
| `ptb_lstm` | Penn Treebank | 2-layer LSTM | ~500k | 64 | Character-level RNN |
| **Quadratic (1 problem)** |
| `quadratic_deep` | Synthetic | 100-D quadratic | 100 | 128 | Deep learning eigenspectrum |
| **2D Optimization (3 problems)** |
| `two_d_rosenbrock` | Synthetic | 2-D scalar | 2 | 128 | Classic Rosenbrock function |
| `two_d_beale` | Synthetic | 2-D scalar | 2 | 128 | Beale function |
| `two_d_branin` | Synthetic | 2-D scalar | 2 | 128 | Branin function |

### Problem Categories

**Classification Problems**: mnist_*, fmnist_*, cifar10_*, cifar100_*, svhn_*, imagenet_*
**Generative Models**: mnist_vae, fmnist_vae
**Sequential Models**: ptb_lstm
**Optimization Benchmarks**: quadratic_deep, two_d_*

---

## Usage Examples

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

### GPU Training

DeepOBS supports CUDA and MPS (Apple Silicon) backends for GPU acceleration.

```python
import torch
from deepobs.pytorch import testproblems

# Auto-select best available device (MPS on Apple Silicon, CUDA on NVIDIA GPUs, CPU otherwise)
# DeepOBS automatically selects the best device when no device is specified
tproblem = testproblems.cifar10_3c3d(batch_size=128)
tproblem.set_up()
print(f"Using device: {tproblem.device}")  # Will show 'mps', 'cuda', or 'cpu'

# Or manually specify device
# device = torch.device('cuda')  # NVIDIA GPU
# device = torch.device('mps')   # Apple Silicon GPU
# device = torch.device('cpu')   # CPU only
# tproblem = testproblems.cifar10_3c3d(batch_size=128, device=device)

# Optimizer and training loop
optimizer = torch.optim.Adam(tproblem.model.parameters(), lr=1e-3)

for epoch in range(100):
    tproblem.model.train()
    for batch in tproblem.dataset.train_loader:
        # Data is automatically moved to device by test problem
        optimizer.zero_grad()
        loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
        loss.mean().backward()
        optimizer.step()
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
    for batch in tproblem.dataset.train_loader:
        optimizer.zero_grad()
        loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
        loss.mean().backward()
        optimizer.step()

    # Step scheduler at epoch end
    scheduler.step()

    print(f'Epoch {epoch}, LR: {scheduler.get_last_lr()[0]:.6f}')
```

### Custom Optimizer

```python
import torch
from torch.optim.optimizer import Optimizer

class MyOptimizer(Optimizer):
    """Custom optimizer example."""

    def __init__(self, params, lr=1e-3, momentum=0.9):
        defaults = dict(lr=lr, momentum=momentum)
        super(MyOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                # Custom update rule here
                d_p = p.grad

                # Example: SGD with momentum
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(group['momentum']).add_(d_p)

                p.add_(buf, alpha=-group['lr'])

        return loss

# Use with StandardRunner
from deepobs.pytorch.runners import StandardRunner

hyperparams = [
    {"name": "momentum", "type": float, "default": 0.9}
]

runner = StandardRunner(MyOptimizer, hyperparams)
runner.run(
    testproblem='mnist_mlp',
    batch_size=128,
    num_epochs=10,
    learning_rate=0.01,
    momentum=0.9
)
```

### Accessing Per-Example Losses

DeepOBS returns per-example losses for advanced optimizer development:

```python
tproblem = testproblems.mnist_mlp(batch_size=128)
tproblem.set_up()

optimizer = torch.optim.SGD(tproblem.model.parameters(), lr=0.01)

for batch in tproblem.dataset.train_loader:
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

### Evaluating on Test Set

```python
import torch

tproblem = testproblems.cifar10_3c3d(batch_size=128)
tproblem.set_up()

def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in test_loader:
            loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
            total_loss += loss.mean().item()
            total_accuracy += accuracy
            total_batches += 1

    avg_loss = total_loss / total_batches
    avg_accuracy = total_accuracy / total_batches

    return avg_loss, avg_accuracy

# After training
test_loss, test_acc = evaluate(
    tproblem.model,
    tproblem.dataset.test_loader,
    tproblem.device
)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}')
```

---

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

### Setting Baseline Directory

```python
from deepobs.pytorch import config

# Set custom baseline directory
config.set_baseline_dir('/path/to/baselines')

# Get current baseline directory
baseline_dir = config.get_baseline_dir()
print(f'Baseline directory: {baseline_dir}')
```

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
└── penn_treebank/  (automatically downloaded)
    └── ptb_processed.pt
```

**Note**: ImageNet requires manual download due to licensing. Place training images in `<data_dir>/imagenet/train/` and validation images in `<data_dir>/imagenet/val/`.

---

## Documentation

### Additional Resources

- **API Reference**: See [API_REFERENCE.md](API_REFERENCE.md) for detailed API documentation
- **Examples**: See [examples/](../examples/) directory for complete example scripts
- **Original Paper**: [DeepOBS: A Deep Learning Optimizer Benchmark Suite (ICLR 2019)](https://openreview.net/forum?id=rJg6ssC5Y7)

### Example Scripts

The `examples/` directory contains complete, runnable examples:

- `basic_usage.py` - Simple end-to-end example
- `custom_optimizer_benchmark.py` - How to benchmark a custom optimizer
- `multiple_test_problems.py` - Running multiple test problems
- `learning_rate_schedule.py` - Using learning rate schedules
- `result_analysis.py` - Analyzing and plotting results

---

## FAQ

### Q: How do I use a specific GPU?

```python
import torch

# DeepOBS auto-selects the best available device by default
# Priority: MPS (Apple Silicon) > CUDA > CPU

# For NVIDIA GPUs (CUDA)
device = torch.device('cuda:0')  # Use GPU 0
tproblem = testproblems.mnist_mlp(batch_size=128, device=device)
tproblem.set_up()

# For Apple Silicon (MPS)
device = torch.device('mps')
tproblem = testproblems.mnist_mlp(batch_size=128, device=device)
tproblem.set_up()

# For CPU only
device = torch.device('cpu')
tproblem = testproblems.mnist_mlp(batch_size=128, device=device)
tproblem.set_up()
```

### Q: How do I ensure reproducibility?

```python
import torch
import numpy as np
import random

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # Also seeds MPS
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Q: Can I use mixed precision training?

```python
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

for batch in tproblem.dataset.train_loader:
    optimizer.zero_grad()

    with autocast():
        loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
        loss = loss.mean()

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Q: How do I save and load checkpoints?

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

### Q: What's the difference between weight_decay in optimizer vs regularization?

PyTorch optimizers handle L2 regularization via the `weight_decay` parameter:

```python
# Correct: Use optimizer's weight_decay
optimizer = torch.optim.SGD(
    tproblem.model.parameters(),
    lr=0.01,
    weight_decay=5e-4  # L2 regularization
)

# Note: For Adam, consider using AdamW for proper weight decay
optimizer = torch.optim.AdamW(
    tproblem.model.parameters(),
    lr=1e-3,
    weight_decay=1e-2
)
```

### Q: How do I handle out-of-memory errors?

- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Use gradient checkpointing for very deep models

```python
# Gradient accumulation example
accumulation_steps = 4
optimizer.zero_grad()

for i, batch in enumerate(tproblem.dataset.train_loader):
    loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
    loss = loss.mean() / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Q: Can I access the dataset directly without test problems?

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

---

## Troubleshooting

### Issue: ImportError when importing deepobs.pytorch

**Solution**: Ensure PyTorch is installed:
```bash
pip install torch torchvision
```

### Issue: CUDA out of memory

**Solutions**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Clear cache: `torch.cuda.empty_cache()`

### Issue: ImageNet not found

**Solution**: ImageNet requires manual download. Download from [image-net.org](http://www.image-net.org/) and place in:
```
<data_dir>/imagenet/train/
<data_dir>/imagenet/val/
```

### Issue: Numerical differences between runs

**Possible causes**:
- Different random seeds
- Different initialization (check batch norm momentum conversion)
- Different data augmentation randomness
- Floating point precision differences

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed troubleshooting.

---

## Performance Tips

1. **Use DataLoader workers**: Set `num_workers > 0` for faster data loading
2. **Pin memory**: Enable `pin_memory=True` for GPU training
3. **Benchmark mode**: Set `torch.backends.cudnn.benchmark = True` (if not requiring determinism)
4. **Mixed precision**: Use AMP for faster training on modern GPUs
5. **Profile your code**: Use `torch.profiler` to identify bottlenecks

---

## Citation

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

---

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

---

## License

DeepOBS is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/fsschneider/DeepOBS/issues)
- **Documentation**: [ReadTheDocs](https://deepobs.readthedocs.io/)
- **Paper**: [ICLR 2019](https://openreview.net/forum?id=rJg6ssC5Y7)

---

## Acknowledgments

**Original DeepOBS Authors**:
- Frank Schneider, Lukas Balles, Philipp Hennig
- University of Tübingen
- ICLR 2019

**PyTorch Implementation**:
- Maintained by the DeepOBS community
- Version 1.2.0+

**Successor Project**: For large-scale benchmarking, see [AlgoPerf](https://github.com/mlcommons/algorithmic-efficiency), the MLCommons algorithmic efficiency benchmark.

---

**Last Updated**: 2025-12-14
**Version**: 1.2.0-pytorch
**Framework**: PyTorch >= 1.9.0
