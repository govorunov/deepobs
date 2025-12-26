# DeepOBS PyTorch API Reference

**Version**: 1.2.0-pytorch
**Last Updated**: 2025-12-26

Complete API documentation for DeepOBS PyTorch.

---

## Table of Contents

- [Configuration](#configuration)
- [Base Classes](#base-classes)
- [Test Problems](#test-problems)
- [Datasets](#datasets)
- [Runners](#runners)
- [Utilities](#utilities)

---

## Configuration

Module: `deepobs.pytorch.config`

### Functions

#### `get_data_dir()`

Get the current data directory where datasets are stored.

**Returns**:
- `str`: Path to the data directory.

**Example**:
```python
from deepobs.pytorch import config
data_dir = config.get_data_dir()
print(f'Data directory: {data_dir}')
```

---

#### `set_data_dir(data_dir)`

Set the data directory where datasets will be stored.

**Arguments**:
- `data_dir` (str): Path to the data directory.

**Returns**:
- None

**Example**:
```python
from deepobs.pytorch import config
config.set_data_dir('/path/to/my/data')
```

---

#### `get_baseline_dir()`

Get the current baseline directory where baseline results are stored.

**Returns**:
- `str`: Path to the baseline directory.

**Example**:
```python
from deepobs.pytorch import config
baseline_dir = config.get_baseline_dir()
```

---

#### `set_baseline_dir(baseline_dir)`

Set the baseline directory where baseline results will be stored.

**Arguments**:
- `baseline_dir` (str): Path to the baseline directory.

**Returns**:
- None

**Example**:
```python
from deepobs.pytorch import config
config.set_baseline_dir('/path/to/baselines')
```

---

#### `get_dtype()`

Get the current default data type for tensors.

**Returns**:
- `torch.dtype`: Default data type (e.g., `torch.float32`).

**Example**:
```python
from deepobs.pytorch import config
dtype = config.get_dtype()
```

---

#### `set_dtype(dtype)`

Set the default data type for tensors.

**Arguments**:
- `dtype` (torch.dtype): Data type to use (e.g., `torch.float32`, `torch.float64`).

**Returns**:
- None

**Example**:
```python
import torch
from deepobs.pytorch import config
config.set_dtype(torch.float64)  # Use double precision
```

---

## Base Classes

### DataSet

Module: `deepobs.pytorch.datasets.dataset`

Base class for all DeepOBS datasets.

#### Constructor

```python
DataSet(batch_size)
```

**Arguments**:
- `batch_size` (int): Batch size for training and testing.

#### Attributes

- `batch_size` (int): The batch size.
- `train_loader` (DataLoader): PyTorch DataLoader for training data.
- `test_loader` (DataLoader): PyTorch DataLoader for test data.
- `train_eval_loader` (DataLoader, optional): DataLoader for training set evaluation (no augmentation).

#### Methods

**Abstract methods** (must be implemented by subclasses):

- `_make_train_dataset()`: Create the training dataset.
- `_make_test_dataset()`: Create the test dataset.

**Example**:
```python
from deepobs.pytorch.datasets import MNIST

dataset = MNIST(batch_size=128)
for batch in dataset.train_loader:
    images, labels = batch
    # Training code
```

---

### TestProblem

Module: `deepobs.pytorch.testproblems.testproblem`

Base class for all DeepOBS test problems.

#### Constructor

```python
TestProblem(batch_size, weight_decay=None, device=None)
```

**Arguments**:
- `batch_size` (int): Batch size for training and testing.
- `weight_decay` (float, optional): Weight decay factor. Defaults to `None`.
- `device` (str or torch.device, optional): Device to use. Defaults to MPS (macOS) if available, then CUDA, then CPU.

#### Attributes

- `batch_size` (int): The batch size.
- `device` (torch.device): The device where computations are performed.
- `dataset` (DataSet): The dataset instance (set after `set_up()`).
- `model` (nn.Module): The model instance (set after `set_up()`).

#### Methods

##### `set_up()`

Set up the test problem by creating the dataset and model.

**Returns**:
- None

**Example**:
```python
from deepobs.pytorch import testproblems

tproblem = testproblems.mnist_mlp(batch_size=128)
tproblem.set_up()
```

---

##### `get_batch_loss_and_accuracy(batch, reduction='mean')`

Compute the loss and accuracy for a batch.

**Arguments**:
- `batch` (tuple): Batch of data `(inputs, targets)` from DataLoader.
- `reduction` (str, optional): Loss reduction method - `'mean'` (default) for scalar loss or `'none'` for per-example losses.

**Returns**:
- `loss` (torch.Tensor): Scalar loss (with `reduction='mean'`) or per-example losses with shape `[batch_size]` (with `reduction='none'`).
- `accuracy` (float or None): Accuracy for the batch, or `None` for non-classification tasks.

**Example**:
```python
for batch in tproblem.dataset.train_loader:
    loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
    print(f'Batch loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')
```

---

##### `get_regularization_loss()`

Get the regularization loss (if any).

**Returns**:
- `torch.Tensor`: Regularization loss (scalar). Returns `0.0` if no regularization.

**Note**: Most test problems use optimizer `weight_decay` instead of manual regularization, so this typically returns `0.0`.

**Example**:
```python
reg_loss = tproblem.get_regularization_loss()
total_loss = loss + reg_loss
```

---

## Test Problems

Module: `deepobs.pytorch.testproblems`

All test problems follow the same API. They are functions that return a `TestProblem` instance.

### General Signature

```python
test_problem_name(batch_size, weight_decay=None, device=None)
```

**Arguments**:
- `batch_size` (int): Batch size for training and testing.
- `weight_decay` (float, optional): Weight decay (L2 regularization) factor.
- `device` (str or torch.device, optional): Device to use.

**Returns**:
- `TestProblem`: Instance of the test problem class.

---

### MNIST Test Problems

#### `mnist_logreg(batch_size, weight_decay=None, device=None)`

Logistic regression on MNIST.

**Model**: Single linear layer (784 → 10)
**Parameters**: 7,850
**Default Batch Size**: 128
**Suggested LR**: 0.1

**Example**:
```python
tproblem = testproblems.mnist_logreg(batch_size=128)
tproblem.set_up()
```

---

#### `mnist_mlp(batch_size, weight_decay=None, device=None)`

Multi-layer perceptron on MNIST.

**Model**: 4-layer fully-connected (784 → 1000 → 500 → 100 → 10)
**Parameters**: 1,134,410
**Default Batch Size**: 128
**Suggested LR**: 0.01

---

#### `mnist_2c2d(batch_size, weight_decay=None, device=None)`

2 convolutional + 2 dense layers on MNIST.

**Model**: Conv(32) → MaxPool → Conv(64) → MaxPool → FC(1024) → FC(10)
**Parameters**: 2,949,120
**Default Batch Size**: 128
**Suggested LR**: 0.01

---

#### `mnist_vae(batch_size, weight_decay=None, device=None)`

Variational autoencoder on MNIST.

**Model**: Encoder (3 conv) → Latent(64) → Decoder (3 deconv)
**Parameters**: ~500,000
**Default Batch Size**: 128
**Suggested LR**: 0.001
**Loss**: Reconstruction (BCE) + KL divergence

**Note**: Returns `accuracy = None` (generative model, no classification).

---

### Fashion-MNIST Test Problems

Same architectures as MNIST, different dataset.

- `fmnist_logreg(batch_size, weight_decay=None, device=None)`
- `fmnist_mlp(batch_size, weight_decay=None, device=None)`
- `fmnist_2c2d(batch_size, weight_decay=None, device=None)`
- `fmnist_vae(batch_size, weight_decay=None, device=None)`

---

### CIFAR-10 Test Problems

#### `cifar10_3c3d(batch_size, weight_decay=None, device=None)`

3 convolutional + 3 dense layers on CIFAR-10.

**Model**: Conv(64) → Conv(96) → Conv(128) → FC(512) → FC(256) → FC(10)
**Parameters**: 1,411,850
**Default Batch Size**: 128
**Suggested LR**: 0.001
**Data Augmentation**: Random crop (32x32 from 40x40), horizontal flip

---

#### `cifar10_vgg16(batch_size, weight_decay=None, device=None)`

VGG-16 on CIFAR-10.

**Model**: VGG-16 (13 conv + 3 FC layers)
**Parameters**: 14,987,722
**Default Batch Size**: 128
**Suggested LR**: 0.01
**Regularization**: Dropout 0.5, typically use `weight_decay=5e-4`
**Input**: Resized to 224x224

---

#### `cifar10_vgg19(batch_size, weight_decay=None, device=None)`

VGG-19 on CIFAR-10.

**Model**: VGG-19 (16 conv + 3 FC layers)
**Parameters**: 20,040,522
**Default Batch Size**: 128
**Suggested LR**: 0.01

---

### CIFAR-100 Test Problems

#### `cifar100_3c3d(batch_size, weight_decay=None, device=None)`

3C3D architecture on CIFAR-100 (100 classes).

**Parameters**: 1,461,700

---

#### `cifar100_allcnnc(batch_size, weight_decay=None, device=None)`

All-convolutional network on CIFAR-100.

**Model**: All convolutional (no pooling), strided convolutions for downsampling
**Parameters**: 1,387,108
**Default Batch Size**: 256
**Suggested LR**: 0.01
**Regularization**: Progressive dropout (0.2 → 0.5), use `weight_decay=1e-3`

---

#### `cifar100_vgg16(batch_size, weight_decay=None, device=None)`

VGG-16 on CIFAR-100.

**Parameters**: 15,002,212

---

#### `cifar100_vgg19(batch_size, weight_decay=None, device=None)`

VGG-19 on CIFAR-100.

**Parameters**: 20,055,012

---

#### `cifar100_wrn404(batch_size, weight_decay=None, device=None)`

Wide ResNet 40-4 on CIFAR-100.

**Model**: 40-layer ResNet with widening factor 4
**Parameters**: 8,952,420
**Default Batch Size**: 128
**Suggested LR**: 0.1
**LR Schedule**: Decay at epochs [60, 120, 160] by factor 0.1
**Regularization**: Use `weight_decay=5e-4`

**Example with LR schedule**:
```python
from torch.optim.lr_scheduler import MultiStepLR

tproblem = testproblems.cifar100_wrn404(batch_size=128)
tproblem.set_up()

optimizer = torch.optim.SGD(
    tproblem.model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=5e-4
)

scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)
```

---

### SVHN Test Problems

#### `svhn_3c3d(batch_size, weight_decay=None, device=None)`

3C3D on Street View House Numbers.

**Parameters**: 1,411,850

---

#### `svhn_wrn164(batch_size, weight_decay=None, device=None)`

Wide ResNet 16-4 on SVHN.

**Model**: 16-layer ResNet with widening factor 4
**Parameters**: 2,748,218
**Default Batch Size**: 128
**Suggested LR**: 0.01

---

### ImageNet Test Problems

**Note**: ImageNet requires manual download. See README for setup instructions.

#### `imagenet_vgg16(batch_size, weight_decay=None, device=None)`

VGG-16 on ImageNet.

**Parameters**: 138,357,544
**Default Batch Size**: 128 (reduce if OOM)
**Suggested LR**: 0.01
**Input**: Resized to 224x224
**Classes**: 1000 + 1 background class (1001 total)

---

#### `imagenet_vgg19(batch_size, weight_decay=None, device=None)`

VGG-19 on ImageNet.

**Parameters**: 143,667,240

---

#### `imagenet_inception_v3(batch_size, weight_decay=None, device=None)`

Inception V3 on ImageNet.

**Model**: Multi-branch architecture with factorized convolutions
**Parameters**: ~27,000,000
**Default Batch Size**: 128
**Suggested LR**: 0.045
**Input**: 299x299
**Features**: Auxiliary classifier, batch norm momentum 0.0003

---

### Quadratic Test Problems

#### `quadratic_deep(batch_size, weight_decay=None, device=None)`

100-dimensional quadratic optimization problem.

**Model**: 100 scalar parameters
**Objective**: `0.5 * (θ - x)^T Q (θ - x)` where `Q` has deep learning eigenspectrum
**Eigenvalues**: 90% in [0, 1], 10% in [30, 60]
**Parameters**: 100
**Default Batch Size**: 128
**Returns**: `accuracy = None` (optimization problem)

**Use Case**: Testing optimizer behavior on ill-conditioned quadratics.

---

### 2D Optimization Test Problems

Classic optimization benchmarks with 2 scalar parameters.

#### `two_d_rosenbrock(batch_size, weight_decay=None, device=None)`

Rosenbrock function: `(1 - u)^2 + 100(v - u^2)^2 + noise`

**Parameters**: 2 (u, v)
**Optimum**: (1, 1)
**Characteristics**: Long, narrow valley

---

#### `two_d_beale(batch_size, weight_decay=None, device=None)`

Beale function: `(1.5 - u + uv)^2 + (2.25 - u + uv^2)^2 + (2.625 - u + uv^3)^2 + noise`

**Parameters**: 2 (u, v)
**Optimum**: (3, 0.5)
**Characteristics**: Flat regions, narrow valley

---

#### `two_d_branin(batch_size, weight_decay=None, device=None)`

Branin function: Complex multi-modal function

**Parameters**: 2 (u, v)
**Characteristics**: Multiple local minima

**Example**:
```python
tproblem = testproblems.two_d_rosenbrock(batch_size=128)
tproblem.set_up()

# Access parameters
params = list(tproblem.model.parameters())
u, v = params[0], params[1]
print(f'Current position: u={u.item():.4f}, v={v.item():.4f}')
```

---

## Datasets

Module: `deepobs.pytorch.datasets`

All datasets can be used independently of test problems.

### MNIST

```python
from deepobs.pytorch.datasets import MNIST

dataset = MNIST(batch_size=128)
```

**Properties**:
- Images: 28x28 grayscale
- Training: 60,000 samples
- Test: 10,000 samples
- Classes: 10 (digits 0-9)
- Normalization: [0, 1]

---

### FashionMNIST

```python
from deepobs.pytorch.datasets import FashionMNIST

dataset = FashionMNIST(batch_size=128)
```

**Properties**:
- Images: 28x28 grayscale
- Training: 60,000 samples
- Test: 10,000 samples
- Classes: 10 (fashion items)

---

### CIFAR10

```python
from deepobs.pytorch.datasets import CIFAR10

dataset = CIFAR10(batch_size=128)
```

**Properties**:
- Images: 32x32 RGB
- Training: 50,000 samples
- Test: 10,000 samples
- Classes: 10
- Augmentation: Random crop (32x32 from 40x40), horizontal flip
- Normalization: Mean [0.4914, 0.4822, 0.4465], Std [0.2023, 0.1994, 0.2010]

---

### CIFAR100

```python
from deepobs.pytorch.datasets import CIFAR100

dataset = CIFAR100(batch_size=128)
```

**Properties**:
- Images: 32x32 RGB
- Training: 50,000 samples
- Test: 10,000 samples
- Classes: 100

---

### SVHN

```python
from deepobs.pytorch.datasets import SVHN

dataset = SVHN(batch_size=128)
```

**Properties**:
- Images: 32x32 RGB
- Training: 73,257 samples
- Test: 26,032 samples
- Classes: 10 (digits 0-9)
- Augmentation: Similar to CIFAR

---

### ImageNet

```python
from deepobs.pytorch.datasets import ImageNet

dataset = ImageNet(batch_size=128)
```

**Properties**:
- Images: Variable size → 224x224
- Training: 1,281,167 samples
- Validation: 50,000 samples
- Classes: 1001 (1000 + background)
- **Requires manual download**

---

### Quadratic

```python
from deepobs.pytorch.datasets import Quadratic

dataset = Quadratic(batch_size=128, dim=100)
```

**Properties**:
- Synthetic Gaussian samples
- Dimension: Configurable (default 100)

---

### TwoD

```python
from deepobs.pytorch.datasets import TwoD

dataset = TwoD(batch_size=128)
```

**Properties**:
- 2D noisy samples for optimization benchmarks

---

## Runners

### StandardRunner

Module: `deepobs.pytorch.runners.standard_runner`

Automates the complete training workflow.

#### Constructor

```python
StandardRunner(optimizer_class, hyperparams)
```

**Arguments**:
- `optimizer_class`: PyTorch optimizer class (e.g., `torch.optim.SGD`).
- `hyperparams` (list): List of hyperparameter dictionaries, each with:
  - `"name"` (str): Parameter name
  - `"type"` (type): Parameter type (e.g., `float`, `int`, `bool`)
  - `"default"` (optional): Default value

**Example**:
```python
from deepobs.pytorch.runners import StandardRunner
import torch.optim as optim

optimizer_class = optim.SGD
hyperparams = [
    {"name": "momentum", "type": float, "default": 0.0},
    {"name": "nesterov", "type": bool, "default": False}
]

runner = StandardRunner(optimizer_class, hyperparams)
```

---

#### `run(**kwargs)`

Run the optimizer on a test problem.

**Arguments**:
- `testproblem` (str): Name of test problem (e.g., `'mnist_mlp'`)
- `batch_size` (int): Batch size
- `num_epochs` (int): Number of training epochs
- `learning_rate` (float): Learning rate
- `lr_sched_epochs` (list, optional): Epochs at which to change LR
- `lr_sched_factors` (list, optional): Factors by which to multiply LR
- `random_seed` (int, optional): Random seed (default: 42)
- `data_dir` (str, optional): Data directory
- `output_dir` (str, optional): Output directory for results (default: 'results')
- `train_log_interval` (int, optional): Logging interval (default: 10)
- `print_train_iter` (bool, optional): Print training iterations (default: False)
- `no_logs` (bool, optional): Disable logging (default: False)
- `weight_decay` (float, optional): Weight decay factor
- `**optimizer_hyperparams`: Additional optimizer hyperparameters

**Returns**:
- None (saves results to JSON file)

**Example**:
```python
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

**Output Structure**:
```
<output_dir>/
└── <testproblem>/
    └── <optimizer>/
        └── <run_id>/
            └── results.json
```

**results.json format**:
```json
{
  "train_losses": [...],
  "test_losses": [...],
  "train_accuracies": [...],
  "test_accuracies": [...],
  "runtime": [...],
  "hyperparams": {...},
  "num_epochs": 10,
  ...
}
```

---

## Utilities

### runner_utils

Module: `deepobs.pytorch.runners.runner_utils`

Utility functions for the StandardRunner.

#### `parse_args(args_dict, hyperparams)`

Parse command-line arguments for the runner.

**Arguments**:
- `args_dict` (dict): Dictionary of arguments
- `hyperparams` (list): Hyperparameter specifications

**Returns**:
- `argparse.Namespace`: Parsed arguments

---

## Type Hints

For better IDE support, use type hints:

```python
from typing import Tuple, Optional
import torch
from deepobs.pytorch.testproblems import TestProblem

def train_model(tproblem: TestProblem,
                optimizer: torch.optim.Optimizer,
                num_epochs: int) -> Tuple[float, float]:
    """Train a model and return final loss and accuracy."""
    tproblem.model.train()

    for epoch in range(num_epochs):
        for batch in tproblem.dataset.train_loader:
            optimizer.zero_grad()
            loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
            loss.backward()
            optimizer.step()

    return loss.item(), accuracy
```

---

## Advanced Usage

### Custom Test Problems

Create custom test problems by subclassing `TestProblem`:

```python
from deepobs.pytorch.testproblems import TestProblem
from deepobs.pytorch.datasets import MNIST
import torch.nn as nn
import torch.nn.functional as F

class MyCustomProblem(TestProblem):
    def __init__(self, batch_size, weight_decay=None, device=None):
        super().__init__(batch_size, weight_decay, device)

    def set_up(self):
        # Create dataset
        self.dataset = MNIST(batch_size=self._batch_size)

        # Create model
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
        self.model.to(self.device)

    def _compute_loss(self, outputs, targets, reduction='mean'):
        if targets.dim() == 2:
            targets = targets.argmax(dim=1)
        return F.cross_entropy(outputs, targets, reduction=reduction)

# Use it
tproblem = MyCustomProblem(batch_size=128)
tproblem.set_up()
```

---

### Multi-GPU Training

```python
import torch
from torch.nn.parallel import DataParallel

tproblem = testproblems.cifar100_wrn404(batch_size=256)
tproblem.set_up()

# Wrap model with DataParallel
if torch.cuda.device_count() > 1:
    print(f'Using {torch.cuda.device_count()} GPUs')
    tproblem.model = DataParallel(tproblem.model)

# Training loop unchanged
optimizer = torch.optim.SGD(tproblem.model.parameters(), lr=0.1)
```

---

## See Also

- **README.md**: Main project overview and usage guide
- **BENCHMARK_SUITE_README.md**: Comprehensive benchmark documentation
- **QUICKSTART.md**: Quick start guide
- **examples/**: Complete example scripts

---

**Last Updated**: 2025-12-26
**Version**: 1.2.0-pytorch
