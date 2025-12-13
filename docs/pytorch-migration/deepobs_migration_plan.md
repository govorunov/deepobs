# DeepOBS TensorFlow → PyTorch Migration Plan

**Generated**: 2025-12-13
**Project**: DeepOBS (Deep Learning Optimizer Benchmark Suite)
**Source**: TensorFlow 1.x → **Target**: PyTorch 2.x

---

## Executive Summary

DeepOBS is a framework for benchmarking deep learning optimizers across 26 standardized test problems. The migration to PyTorch will be a clean reimplementation focusing on modern patterns, not backward compatibility. Core functionality: run optimizers on test problems, collect metrics (loss, accuracy, timing), generate JSON output, and create publication-quality plots.

**Key Insight**: The analyzer component (plotting/analysis) is already framework-agnostic and works with JSON files, so it requires minimal changes. The main conversion effort is in datasets, models, and the training runner.

---

## Current Codebase Structure

```
deepobs/
├── tensorflow/                    # CONVERT TO PYTORCH
│   ├── config.py                  # Simple config (35 lines)
│   ├── datasets/                  # 10 dataset modules
│   │   ├── dataset.py             # Base class (92 lines)
│   │   └── mnist.py, fmnist.py, cifar10.py, cifar100.py,
│   │       svhn.py, imagenet.py, tolstoi.py, quadratic.py, two_d.py
│   ├── testproblems/              # 37 files total
│   │   ├── testproblem.py         # Base class (63 lines)
│   │   ├── _*.py                  # 9 architecture modules (reusable)
│   │   │   ├── _logreg.py, _mlp.py, _2c2d.py, _3c3d.py
│   │   │   ├── _vgg.py, _wrn.py, _inception_v3.py
│   │   │   ├── _vae.py, _quadratic.py
│   │   └── {dataset}_{arch}.py    # 27 concrete test problems
│   └── runners/
│       ├── standard_runner.py     # Main runner (565 lines)
│       └── runner_utils.py        # Utilities (105 lines)
│
└── analyzer/                      # KEEP (MOSTLY UNCHANGED)
    ├── analyze.py                 # Framework-agnostic (191 lines)
    └── analyze_utils.py           # Plotting utilities (999 lines)
```

**Component Counts**:
- 10 datasets (9 real + 1 synthetic)
- 9 reusable architectures
- 27 concrete test problems
- 1 main runner + utilities
- Analysis tools (framework-agnostic)

---

## Architecture Inventory

### 1. Simple Architectures (Easy)
- **Logistic Regression**: Single linear layer (784→10)
- **MLP**: 4 FC layers (784→1000→500→100→10)
- **2C2D**: 2 conv + 2 dense (Conv32→Conv64→FC1024→FC10)

### 2. Convolutional Networks (Medium)
- **3C3D**: 3 conv + 3 dense (Conv64→Conv96→Conv128→FC512→FC256→FC10)
- **VGG16/19**: 13/16 conv + 3 dense (standard VGG architecture)
- **All-CNN-C**: All convolutional (no pooling, strided conv for downsampling)

### 3. Advanced Architectures (Complex)
- **Wide ResNet**: Residual network with batch norm (16-40 layers, width factor 4)
- **Inception V3**: Multi-branch architecture with factorized convolutions
- **VAE**: Encoder-decoder with latent space (conv encoder + deconv decoder)
- **Char-RNN**: 2-layer LSTM for character-level language modeling

### 4. Synthetic Problems (Simple)
- **Quadratic**: Synthetic quadratic optimization landscape
- **2D Functions**: Rosenbrock, Beale, Branin (2D optimization test functions)

---

## Dataset Inventory

1. **MNIST**: 60k train, 10k test (28x28 grayscale)
2. **Fashion-MNIST**: Same shape as MNIST, fashion items
3. **CIFAR-10**: 50k train, 10k test (32x32 RGB, 10 classes)
4. **CIFAR-100**: Same as CIFAR-10 but 100 classes
5. **SVHN**: Street View House Numbers
6. **ImageNet**: Large-scale (requires manual setup)
7. **Tolstoi**: Character-level text (War and Peace)
8. **Quadratic**: Synthetic quadratic problems
9. **Two-D**: 2D optimization test functions

**Key Features**:
- Auto-download and preprocessing
- Data augmentation (random crop, flip, lighting)
- Train/eval/test split handling
- Consistent normalization

---

## Test Problem Combinations (26 Total)

| Dataset | Test Problems |
|---------|--------------|
| MNIST | logreg, mlp, 2c2d, vae |
| Fashion-MNIST | logreg, mlp, 2c2d, vae |
| CIFAR-10 | 3c3d, vgg16, vgg19 |
| CIFAR-100 | 3c3d, allcnnc, vgg16, vgg19, wrn404 |
| SVHN | 3c3d, wrn164 |
| ImageNet | vgg16, vgg19, inception_v3 |
| Tolstoi | char_rnn |
| Synthetic | quadratic_deep, 2d_rosenbrock, 2d_beale, 2d_branin |

---

## Key TensorFlow Patterns to Convert

### 1. Data Pipeline
**TensorFlow**:
```python
# Reinitializable iterator with phase switching
iterator = tf.data.Iterator.from_structure(...)
batch = iterator.get_next()
train_init_op = iterator.make_initializer(train_dataset)
test_init_op = iterator.make_initializer(test_dataset)
phase = tf.Variable("train", trainable=False)
```

**PyTorch**:
```python
# Simple DataLoader switching
train_loader = DataLoader(dataset, batch_size, shuffle=True)
test_loader = DataLoader(dataset, batch_size, shuffle=False)
# Phase handled by model.train() / model.eval()
```

### 2. Model Building
**TensorFlow** (functional):
```python
def _mlp(x, num_outputs):
    x = tf.reshape(x, [-1, 784])
    x = tf.layers.dense(x, 1000, activation=tf.nn.relu)
    x = tf.layers.dense(x, 500, activation=tf.nn.relu)
    return tf.layers.dense(x, num_outputs)
```

**PyTorch** (class-based):
```python
class MLP(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(500, 100)
        self.fc3 = nn.Linear(100, num_outputs)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
```

### 3. Training Loop
**TensorFlow** (session-based):
```python
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(train_init_op)
while True:
    try:
        _, loss_val = sess.run([step, loss])
    except tf.errors.OutOfRangeError:
        break
```

**PyTorch** (eager execution):
```python
model.train()
for batch in train_loader:
    optimizer.zero_grad()
    loss = compute_loss(model(batch))
    loss.backward()
    optimizer.step()
```

### 4. Batch Normalization
**TensorFlow** (manual update collection):
```python
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    step = opt.minimize(loss)
```

**PyTorch** (automatic):
```python
# Handled automatically by model.train() / model.eval()
```

---

## Migration Phases (Ordered by Dependency)

### Phase 1: Core Infrastructure
**Duration**: 1-2 days
**Complexity**: Low
**Files**: 3 files, ~150 lines total

**Tasks**:
1. Create `deepobs/pytorch/` directory structure
2. Implement `config.py` (data/baseline dirs, dtype settings)
3. Implement base `DataSet` class
4. Implement base `TestProblem` class

**Deliverables**:
- `deepobs/pytorch/__init__.py`
- `deepobs/pytorch/config.py`
- `deepobs/pytorch/datasets/dataset.py`
- `deepobs/pytorch/testproblems/testproblem.py`

**Success Criteria**: Base classes can be imported and instantiated

---

### Phase 2: Simple Datasets
**Duration**: 2-3 days
**Complexity**: Low
**Files**: 4 datasets

**Tasks**:
1. Convert MNIST dataset
2. Convert Fashion-MNIST dataset
3. Convert CIFAR-10 dataset
4. Convert CIFAR-100 dataset
5. Add basic unit tests for data loading

**Key Considerations**:
- Use `torchvision.datasets` where available (MNIST, FashionMNIST, CIFAR10/100)
- Implement custom `DataLoader` wrapping for consistent API
- Handle train/eval/test split properly
- Verify preprocessing matches TensorFlow version

**Success Criteria**: Can load batches with correct shapes and preprocessing

---

### Phase 3: Simple Test Problems
**Duration**: 3-4 days
**Complexity**: Medium
**Files**: 3 architectures + 8 test problems

**Tasks**:
1. Implement `_logreg.py` architecture
2. Implement `_mlp.py` architecture
3. Implement `_2c2d.py` architecture
4. Create 8 test problems (mnist/fmnist × logreg/mlp/2c2d)
5. Verify forward pass outputs match shapes
6. Test loss computation and accuracy metrics

**Key Considerations**:
- Per-example loss required (use `reduction='none'`)
- Proper weight initialization (truncated normal, stddev=0.03)
- No regularization for these problems

**Success Criteria**: Can run forward pass and compute loss/accuracy on all 8 problems

---

### Phase 4: Basic Runner
**Duration**: 4-5 days
**Complexity**: Medium-High
**Files**: 2 files, ~400 lines

**Tasks**:
1. Implement `StandardRunner` class
2. Implement `runner_utils.py` helpers
3. Support command-line argument parsing
4. Implement training loop with epoch-based evaluation
5. Add basic metric logging (loss, accuracy)
6. Generate JSON output compatible with existing format
7. Implement learning rate scheduling

**Key Considerations**:
- Must maintain JSON output format for analyzer compatibility
- Support hyperparameter specification
- Epoch-based evaluation (not step-based)
- Learning rate schedule via `torch.optim.lr_scheduler`
- Timing measurements for performance comparison

**Success Criteria**: Can run end-to-end training on simple test problems and produce valid JSON output

---

### Phase 5: Remaining Datasets
**Duration**: 3-4 days
**Complexity**: Medium
**Files**: 5 datasets

**Tasks**:
1. Convert SVHN dataset
2. Convert ImageNet dataset (wrapper for torchvision)
3. Convert Tolstoi dataset (character-level text)
4. Convert Quadratic dataset (synthetic)
5. Convert Two-D dataset (synthetic 2D functions)

**Key Considerations**:
- ImageNet requires manual setup (document clearly)
- Tolstoi needs custom text processing
- Synthetic datasets are simple numpy arrays
- Ensure preprocessing matches TensorFlow version

**Success Criteria**: All datasets load correctly and pass unit tests

---

### Phase 6: Advanced Architectures
**Duration**: 5-7 days
**Complexity**: High
**Files**: 6 architectures + 15 test problems

**Tasks**:
1. Implement `_3c3d.py` architecture
2. Implement `_vgg.py` architecture (VGG16/19)
3. Implement `_wrn.py` architecture (Wide ResNet)
4. Implement `_inception_v3.py` architecture
5. Implement `_vae.py` architecture
6. Implement All-CNN-C architecture
7. Implement `_quadratic.py` architecture
8. Create all remaining test problems (15 total)
9. Test batch normalization behavior carefully
10. Verify VAE reconstruction quality

**Key Considerations**:
- **Batch Norm Momentum**: PyTorch uses opposite convention (0.1 vs 0.9)
- **Weight Decay**: Use optimizer's `weight_decay` parameter
- **VGG/Inception**: Can reference `torchvision.models` for architecture verification
- **ResNet**: Careful with skip connections and pre-activation order
- **VAE**: Reparameterization trick, KL divergence loss
- **Channel Ordering**: NCHW (PyTorch) vs NHWC (TensorFlow)

**Success Criteria**: All 15 test problems run without errors, architectures match TensorFlow output shapes

---

### Phase 7: RNN and Specialized Problems
**Duration**: 2-3 days
**Complexity**: Medium
**Files**: 1 architecture + 4 test problems

**Tasks**:
1. Implement Char-RNN architecture (2-layer LSTM)
2. Create `tolstoi_char_rnn` test problem
3. Implement 2D optimization test problems (Rosenbrock, Beale, Branin)
4. Handle LSTM state persistence across batches
5. Test sequence processing

**Key Considerations**:
- **LSTM State**: Detach between batches to prevent gradients across batches
- **Dropout**: Use `nn.LSTM(..., dropout=0.2)` for inter-layer dropout
- **Batch-First**: Use `batch_first=True` for consistency
- **State Reset**: Reset hidden state at epoch boundaries

**Success Criteria**: Char-RNN trains on Tolstoi dataset with proper state handling

---

## Validation Strategy

### Unit Tests (per phase)
- Dataset loading and shapes
- Model forward pass
- Loss computation (per-example and reduced)
- Accuracy computation
- Weight initialization verification

### Integration Tests
- End-to-end training (1 epoch minimum)
- JSON output format validation
- Learning rate schedule correctness
- Metric tracking accuracy

### Regression Tests
- Compare loss curves with TensorFlow baseline (within 1% tolerance)
- Compare final accuracy (within 0.5% tolerance)
- Performance benchmarking (training speed)

---

## Critical Conversion Gotchas

### 1. Batch Normalization Momentum
- **TF**: `momentum=0.9` means 90% old, 10% new
- **PyTorch**: `momentum=0.1` means 10% old, 90% new
- **Conversion**: `momentum_pt = 1 - momentum_tf`

### 2. Cross-Entropy Loss
- **TF**: `softmax_cross_entropy_with_logits_v2(labels, logits)` (one-hot labels)
- **PyTorch**: `F.cross_entropy(logits, labels)` (class indices, argument order reversed)

### 3. Weight Decay vs L2 Regularization
- **TF**: Manual via `kernel_regularizer`, added to loss
- **PyTorch**: Built into optimizer via `weight_decay` parameter
- **Note**: Not exactly equivalent for Adam (use AdamW for true weight decay)

### 4. Tensor Shape Conventions
- **TF**: Channel-last by default (NHWC for images)
- **PyTorch**: Channel-first by default (NCHW for images)
- Must transpose: `img.permute(0, 3, 1, 2)` when loading data

### 5. Per-Example Losses
- **Requirement**: Must return per-example losses before averaging
- **PyTorch**: Use `reduction='none'` in loss functions
- Example: `F.cross_entropy(logits, y, reduction='none')`

### 6. Dropout and Training Mode
- **TF**: Manual phase variable and `training` argument
- **PyTorch**: Use `model.train()` / `model.eval()` (automatic)

---

## Dependencies

### Current (TensorFlow)
```
tensorflow >= 1.4.0
numpy
pandas
matplotlib
matplotlib2tikz (0.6.18)
seaborn
argparse
```

### New (PyTorch)
```
torch >= 2.0.0
torchvision >= 0.15.0
numpy
pandas
matplotlib
seaborn
argparse
```

**Notes**:
- Remove `matplotlib2tikz` (deprecated, not needed)
- All analysis tools remain compatible
- Consider adding `tqdm` for progress bars

---

## Success Metrics

### Functional Completeness
- [ ] All 10 datasets loading correctly
- [ ] All 9 architectures implemented
- [ ] All 26 test problems working
- [ ] Runner produces valid JSON output
- [ ] Analyzer works with PyTorch output

### Correctness
- [ ] Loss curves match TensorFlow within 1%
- [ ] Final accuracies match TensorFlow within 0.5%
- [ ] Training convergence speed similar

### Code Quality
- [ ] Modern PyTorch patterns (no TF-style code)
- [ ] Clean, readable implementations
- [ ] Proper documentation
- [ ] Unit test coverage > 80%

---

## Estimated Timeline

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Core Infrastructure | 1-2 days | 2 days |
| Phase 2: Simple Datasets | 2-3 days | 5 days |
| Phase 3: Simple Test Problems | 3-4 days | 9 days |
| Phase 4: Basic Runner | 4-5 days | 14 days |
| Phase 5: Remaining Datasets | 3-4 days | 18 days |
| Phase 6: Advanced Architectures | 5-7 days | 25 days |
| Phase 7: RNN and Specialized | 2-3 days | 28 days |
| Testing and Validation | 3-4 days | 32 days |
| Documentation | 2-3 days | 35 days |

**Total Estimated Time**: 5-7 weeks (1 developer, full-time)

**Minimum Viable Product**: Phases 1-4 (2-3 weeks) provides basic functionality for simple test problems

---

## Next Steps (Immediate Actions)

1. **Set up environment**: Create `deepobs/pytorch/` directory structure
2. **Phase 1 Start**: Implement base classes (config, dataset, testproblem)
3. **Proof of concept**: Convert MNIST dataset and MLP architecture
4. **Verify approach**: Run one simple test problem end-to-end
5. **Iterate**: Once PoC works, proceed with remaining phases

---

## Design Principles

1. **Clean Reimplementation**: Don't mirror TensorFlow design, use PyTorch idioms
2. **Simplicity**: Prefer simple, readable code over clever abstractions
3. **Compatibility**: Maintain JSON output format for analyzer compatibility
4. **Modern Patterns**: Use PyTorch 2.x features (no legacy patterns)
5. **Testability**: Write unit tests as you go, not after
6. **Documentation**: Clear docstrings and usage examples

---

## Risk Mitigation

### High Risk Areas
1. **Batch Normalization**: Different momentum conventions, careful testing needed
2. **LSTM State Management**: TensorFlow uses variables, PyTorch uses returned tensors
3. **Regularization**: Weight decay in optimizer vs manual L2 loss
4. **Numerical Differences**: Small differences in initialization/computation may compound

### Mitigation Strategies
1. **Incremental Testing**: Test each component against TensorFlow output
2. **Reference Implementations**: Use `torchvision.models` to verify architectures
3. **Regression Tests**: Compare final metrics with TensorFlow baselines
4. **Documentation**: Document all conversion decisions and gotchas

---

**Document Version**: 1.0
**Last Updated**: 2025-12-13
**Status**: Ready for implementation
