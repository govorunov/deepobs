# DeepOBS PyTorch Test Problems

This directory contains test problem implementations for PyTorch DeepOBS.

## Directory Structure

```
testproblems/
├── __init__.py                # Package exports
├── testproblem.py             # Base class for all test problems
│
├── _logreg.py                 # Logistic regression architecture
├── _mlp.py                    # Multi-layer perceptron architecture
├── _2c2d.py                   # 2 Conv + 2 Dense architecture
├── _3c3d.py                   # 3 Conv + 3 Dense architecture
│
├── mnist_logreg.py            # MNIST + Logistic Regression
├── mnist_mlp.py               # MNIST + MLP
├── mnist_2c2d.py              # MNIST + 2C2D
│
├── fmnist_logreg.py           # Fashion-MNIST + Logistic Regression
├── fmnist_mlp.py              # Fashion-MNIST + MLP
├── fmnist_2c2d.py             # Fashion-MNIST + 2C2D
│
├── cifar10_3c3d.py            # CIFAR-10 + 3C3D
└── cifar100_3c3d.py           # CIFAR-100 + 3C3D
```

## Available Test Problems

### MNIST (3 problems)
- `mnist_logreg` - Logistic regression (784→10)
- `mnist_mlp` - 4-layer MLP (784→1000→500→100→10)
- `mnist_2c2d` - 2 conv + 2 dense layers

### Fashion-MNIST (3 problems)
- `fmnist_logreg` - Logistic regression (784→10)
- `fmnist_mlp` - 4-layer MLP (784→1000→500→100→10)
- `fmnist_2c2d` - 2 conv + 2 dense layers

### CIFAR-10 (1 problem)
- `cifar10_3c3d` - 3 conv + 3 dense layers (with weight decay 0.002)

### CIFAR-100 (1 problem)
- `cifar100_3c3d` - 3 conv + 3 dense layers (with weight decay 0.002)

## Usage Example

```python
import torch
from deepobs.pytorch.testproblems import mnist_mlp

# Create test problem
problem = mnist_mlp(batch_size=128, device='cuda')
problem.set_up()

# Access dataset and model
train_loader = problem.dataset.train_loader
model = problem.model

# Training loop
model.train()
for batch in train_loader:
    # Get loss and accuracy
    loss, accuracy = problem.get_batch_loss_and_accuracy(batch, reduction='mean')

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Architecture Details

### Logistic Regression
- **Layers**: Single linear layer (784→num_outputs)
- **Initialization**: Zero weights and biases
- **Used in**: MNIST, Fashion-MNIST

### MLP (Multi-Layer Perceptron)
- **Layers**: 784→1000→500→100→num_outputs
- **Activations**: ReLU on hidden layers
- **Initialization**: Truncated normal (std=0.03), zero biases
- **Used in**: MNIST, Fashion-MNIST

### 2C2D (2 Convolutional + 2 Dense)
- **Conv Layers**: 1→32→64 (5×5 kernels, same padding)
- **Pooling**: 2×2 max pooling after each conv
- **FC Layers**: 3136→1024→num_outputs
- **Initialization**: Truncated normal (std=0.05), constant biases (0.05)
- **Used in**: MNIST, Fashion-MNIST

### 3C3D (3 Convolutional + 3 Dense)
- **Conv Layers**: 3→64→96→128 (5×5, 3×3, 3×3 kernels)
- **Pooling**: 3×3 max pooling (stride=2) after each conv
- **FC Layers**: 1152→512→256→num_outputs
- **Initialization**: Xavier/Glorot (normal for conv, uniform for FC)
- **Weight Decay**: Default 0.002
- **Used in**: CIFAR-10, CIFAR-100

## Implementation Status

### Phase 3 (Current)
- ✅ 4 architecture modules
- ✅ 8 test problems (MNIST, Fashion-MNIST, CIFAR)
- ✅ Comprehensive test suite

### Future Phases
- Phase 4: Runner implementation
- Phase 5: Additional datasets (SVHN, ImageNet, etc.)
- Phase 6: Advanced architectures (VGG, ResNet, Inception, VAE)
- Phase 7: RNN and specialized problems

## Testing

Run the test suite:
```bash
python -m pytest tests/test_pytorch_architectures.py -v
```

## Notes

- All test problems inherit from `TestProblem` base class
- Loss computation uses `F.cross_entropy` with `reduction='none'` for per-example losses
- Weight decay is handled by the optimizer, not added to the loss manually
- One-hot encoded targets are automatically converted to class indices

## References

- Original TensorFlow implementation: `deepobs/tensorflow/testproblems/`
- Migration guide: `docs/pytorch-migration/`
- Phase 3 completion report: `docs/pytorch-migration/phase3_completion_report.md`
