# DeepOBS → PyTorch Conversion Guide

**Project**: DeepOBS (Deep Learning Optimizer Benchmark Suite)
**Source Framework**: TensorFlow 1.x
**Target Framework**: PyTorch
**Created**: 2025-12-13
**Status**: ✅ COMPLETE - All phases finished
**Location**: /Users/yaroslav/Sources/Angol/DeepOBS/planning/

---

## 📁 IMPORTANT: Documentation Organization

**All planning and implementation tracking documents belong in the `planning/` folder:**

### Planning Documents (Keep in planning/)
- `CLAUDE.md` - This conversion guide and planning document
- `IMPLEMENTATION_STATUS.md` - Phase-by-phase progress tracking
- `PHASE*_*.md` - All phase implementation notes and summaries
- `MIGRATION_COMPLETE*.md` - Migration completion reports
- `PROJECT_SUMMARY.md` - Internal project statistics
- `FINAL_PROJECT_REPORT.md` - Final completion report
- `DOCUMENTATION_INDEX.md` - Internal documentation index

### User-Facing Documents (Keep in project root)
- `README.md` - Main project README
- `README_PYTORCH.md` - PyTorch usage guide
- `MIGRATION_GUIDE.md` - User migration guide (TF → PyTorch)
- `API_REFERENCE.md` - API documentation
- `EXAMPLES.md` - Usage examples
- `CHANGELOG_PYTORCH.md` - Version history
- `KNOWN_ISSUES.md` - Known issues and limitations
- `RELEASE_CHECKLIST.md` - Release preparation steps
- `CONTRIBUTORS.md` - Contributors and acknowledgments
- `VERSION` - Version file

**Instructions for Future Work:**
- All new planning documents go in `planning/`
- All new phase summaries go in `planning/`
- All implementation notes go in `planning/`
- Keep project root clean with only user-facing docs

---

## 1. PROJECT OVERVIEW

### Purpose
DeepOBS is a benchmarking framework that automates the evaluation and comparison of deep learning optimizers across multiple test problems and datasets. It provides standardized test problems, baseline comparisons, and publication-quality visualizations.

### Key Information
- **Version**: 1.1.2
- **License**: MIT
- **Authors**: Frank Schneider, Lukas Balles, Philipp Hennig
- **Paper**: ICLR 2019 (https://openreview.net/forum?id=rJg6ssC5Y7)
- **Status**: No longer maintained → Superseded by AlgoPerf (https://github.com/mlcommons/algorithmic-efficiency)
- **Current Backend**: TensorFlow 1.12 (compatible with TF >= 1.4.0)

### Project Goals
1. Automate optimizer benchmarking workflow
2. Provide realistic test problems with modern architectures
3. Enable fair comparison through baseline results
4. Generate publication-ready visualizations
5. Reduce bias in optimizer evaluation

---

## 2. DIRECTORY STRUCTURE

```
DeepOBS/
├── deepobs/                          # Main package
│   ├── tensorflow/                   # TensorFlow backend (TO CONVERT)
│   │   ├── config.py                 # Global configuration
│   │   ├── datasets/                 # 10 dataset modules
│   │   │   ├── dataset.py            # Base class
│   │   │   ├── mnist.py, fmnist.py, cifar10.py, cifar100.py
│   │   │   ├── svhn.py, imagenet.py
│   │   │   ├── tolstoi.py, quadratic.py, two_d.py
│   │   ├── testproblems/             # 37 test problem files
│   │   │   ├── testproblem.py        # Base class
│   │   │   ├── _mlp.py, _2c2d.py, _3c3d.py, _vgg.py, _wrn.py
│   │   │   ├── _inception_v3.py, _vae.py, _logreg.py, _quadratic.py
│   │   │   ├── mnist_*.py, fmnist_*.py, cifar10_*.py, cifar100_*.py
│   │   │   ├── imagenet_*.py, svhn_*.py, tolstoi_char_rnn.py
│   │   │   ├── quadratic_deep.py, two_d_*.py
│   │   └── runners/                  # Training orchestration
│   │       ├── standard_runner.py    # Main runner class (24KB)
│   │       └── runner_utils.py       # Utilities
│   ├── analyzer/                     # Result analysis (MOSTLY FRAMEWORK-AGNOSTIC)
│   │   ├── analyze.py
│   │   └── analyze_utils.py          # 40KB+ of plotting/analysis code
│   └── scripts/                      # Command-line tools
│       ├── deepobs_prepare_data.sh   # Data downloading
│       ├── deepobs_get_baselines.sh  # Baseline downloading
│       ├── deepobs_plot_results.py   # Visualization
│       └── deepobs_estimate_runtime.py
├── docs/                             # Sphinx documentation
├── tests/                            # Unit tests
└── example_momentum_runner.py        # Usage example
```

---

## 3. CORE COMPONENTS TO CONVERT

### 3.1 Datasets (10 Modules)

**Base Class**: `deepobs/tensorflow/datasets/dataset.py`

**TensorFlow-Specific Patterns**:
- `tf.data.Dataset` API for data pipelines
- `tf.data.Iterator` with reinitializable structure
- `from_tensor_slices()`, `shuffle()`, `batch()`, `prefetch()`
- Phase variable (`tf.Variable` for "train"/"test" mode switching)
- `train_init_op`, `test_init_op` for dataset switching

**PyTorch Equivalents**:
- `torch.utils.data.Dataset` + `DataLoader`
- No need for reinitializable iterators (simpler design)
- Phase handling via `model.train()` / `model.eval()`
- Batching/shuffling via `DataLoader` parameters

**Dataset List**:
1. MNIST (`mnist.py`) - 28x28 grayscale, 60k train, 10k test
2. Fashion-MNIST (`fmnist.py`) - 28x28 grayscale fashion items
3. CIFAR-10 (`cifar10.py`) - 32x32 RGB, 10 classes
4. CIFAR-100 (`cifar100.py`) - 32x32 RGB, 100 classes
5. SVHN (`svhn.py`) - Street View House Numbers
6. ImageNet (`imagenet.py`) - Large-scale classification (manual setup)
7. Tolstoi (`tolstoi.py`) - Character-level text (War and Peace)
8. Quadratic (`quadratic.py`) - Synthetic quadratic problems
9. Two-D (`two_d.py`) - 2D optimization test functions

**Key Features to Preserve**:
- Automatic downloading and preprocessing
- Data augmentation (random crop, horizontal flip, lighting)
- Train/eval/test split handling
- Consistent normalization and preprocessing

---

### 3.2 Neural Network Architectures (9 Types)

#### A. Multi-Layer Perceptron (MLP)
**File**: `testproblems/_mlp.py`

**Structure**: 784 → 1000 → 500 → 100 → num_outputs

**TensorFlow Code**:
```python
def _mlp(x, num_outputs):
    x = tf.reshape(x, [-1, 784])
    x = tf.layers.dense(x, 1000, activation=tf.nn.relu,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=3e-2))
    x = tf.layers.dense(x, 500, activation=tf.nn.relu)
    x = tf.layers.dense(x, 100, activation=tf.nn.relu)
    outputs = tf.layers.dense(x, num_outputs, activation=None)
    return outputs
```

**PyTorch Conversion**:
```python
import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        self.fc1 = nn.Linear(784, 1000)
        self.fc2 = nn.Linear(1000, 500)
        self.fc3 = nn.Linear(500, 100)
        self.fc4 = nn.Linear(100, num_outputs)
        # Initialize weights
        for m in [self.fc1, self.fc2, self.fc3, self.fc4]:
            init.trunc_normal_(m.weight, std=0.03)
            init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)
```

**Test Problems**: `mnist_mlp`, `fmnist_mlp`

---

#### B. 2C2D and 3C3D (Convolutional Networks)
**Files**: `testproblems/_2c2d.py`, various 3c3d problems

**Structure**:
- 2C2D: Conv(32) → MaxPool → Conv(64) → MaxPool → FC(1024) → FC(out)
- 3C3D: Conv(64) → Conv(96) → Conv(128) → FC(512) → FC(256) → FC(out)

**Key Details**:
- 5x5 kernels for 2C2D
- 3x3 kernels for 3C3D
- ReLU activations
- Max pooling (2x2, stride 2)
- Truncated normal initialization (stddev=0.05)

**Test Problems**: `mnist_2c2d`, `fmnist_2c2d`, `cifar10_3c3d`, `cifar100_3c3d`, `svhn_3c3d`

---

#### C. VGG Networks (VGG16, VGG19)
**File**: `testproblems/_vgg.py`

**Structure**:
- VGG16: 13 conv layers + 3 FC layers (16 weight layers total)
- VGG19: 16 conv layers + 3 FC layers (19 weight layers total)
- Input resized to 224x224
- All conv filters are 3x3
- Max pooling after each block
- Glorot (Xavier) normal initialization
- L2 regularization on all layers
- Dropout 0.5 on FC layers

**TensorFlow-Specific Issues**:
- Uses `tf.contrib.layers.l2_regularizer(weight_decay)`
- Manual control dependencies for batch norm updates (though VGG doesn't use BN)

**PyTorch Benefits**:
- Can use `torchvision.models.vgg16()` / `vgg19()` as reference
- L2 regularization via optimizer `weight_decay` parameter
- Simpler dropout handling

**Test Problems**: `cifar10_vgg16/19`, `cifar100_vgg16/19`, `imagenet_vgg16/19`

---

#### D. Wide Residual Networks (WRN)
**File**: `testproblems/_wrn.py`

**Key Features**:
- Residual connections with skip paths
- Batch normalization (momentum=0.9)
- BN-ReLU-Conv pattern (pre-activation)
- Configurable depth and widening factor
- L2 regularization on conv kernels
- Identity vs projection shortcuts

**Critical TensorFlow Pattern**:
```python
# Batch norm updates must be added to UPDATE_OPS collection
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    step = opt.minimize(loss)
```

**PyTorch Advantage**:
- No manual batch norm update tracking needed
- Can use `torchvision.models.wide_resnet*()` as reference

**Test Problems**: `svhn_wrn164` (16 layers, width 4), `cifar100_wrn404` (40 layers, width 4)

---

#### E. Inception V3
**File**: `testproblems/_inception_v3.py`

**Key Features**:
- Multi-branch parallel conv paths
- Batch norm momentum: 0.9997 (non-standard)
- 1x7 and 7x1 factorized convolutions
- Mixed inception blocks (different branch structures)
- No bias in conv layers (BN provides bias)

**PyTorch Reference**: Can use `torchvision.models.inception_v3()` as base

**Test Problems**: `imagenet_inception_v3`

---

#### F. Variational Autoencoder (VAE)
**File**: `testproblems/_vae.py`

**Structure**:
- Encoder: 3 conv layers → FC → latent distribution (mean, log_std)
- Decoder: FC → 3 deconv layers → reconstruction
- Reparameterization trick: z = mean + epsilon * exp(log_std)
- Leaky ReLU (alpha=0.3) in encoder
- Dropout 0.2 in both encoder and decoder
- Sigmoid activation on decoder output

**Loss Components**:
- Reconstruction loss: Binary cross-entropy
- KL divergence: Regularization term

**Test Problems**: `mnist_vae`, `fmnist_vae`

---

#### G. All-CNN-C
**File**: `testproblems/cifar100_allcnnc.py`

**Key Features**:
- All convolutional (no max pooling)
- Strided convolutions for downsampling
- Progressive dropout: 0.2 → 0.5
- L2 regularization
- Global average pooling before output

**Test Problems**: `cifar100_allcnnc`

---

#### H. Character-level RNN (LSTM)
**File**: `testproblems/tolstoi_char_rnn.py`

**Structure**:
- Embedding layer (vocab_size=83, embedding_dim=128)
- 2-layer LSTM (128 hidden units)
- Dropout wrapper (input_keep_prob=0.8, output_keep_prob=0.8)
- Dense output layer (vocab_size)
- Sequence length: 50

**TensorFlow-Specific Issues**:
- Uses deprecated `tf.contrib.rnn.LSTMCell`
- Uses `tf.nn.static_rnn` (unrolls sequence)
- State preservation via `tf.Variable` (trainable=False)
- `tf.contrib.seq2seq.sequence_loss` for loss computation

**PyTorch Conversion**:
```python
class CharRNN(nn.Module):
    def __init__(self, vocab_size=83, embedding_dim=128, hidden_size=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                           dropout=0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden
```

**Test Problems**: `tolstoi_char_rnn`

---

#### I. Logistic Regression
**File**: `testproblems/_logreg.py`

**Structure**: Single linear layer (784 → num_outputs)

**Test Problems**: `mnist_logreg`, `fmnist_logreg`

---

#### J. Quadratic and 2D Test Functions
**Files**: `testproblems/_quadratic.py`, various `two_d_*.py`

**Purpose**: Synthetic optimization test functions
- Quadratic: Loss = 0.5 * (θ - x)^T * Q * (θ - x)
- 2D functions: Rosenbrock, Beale, Branin

**Test Problems**: `quadratic_deep`, `two_d_rosenbrock`, `two_d_beale`, `two_d_branin`

---

### 3.3 Test Problems (26 Total)

**Base Class**: `deepobs/tensorflow/testproblems/testproblem.py`

**Key Methods to Implement**:
- `set_up()` - Initialize dataset and build model
- `get_batch_loss_and_accuracy()` - Compute metrics for a batch
- Properties: `losses`, `accuracy`, `regularizer`

**TensorFlow Pattern**:
```python
class TestProblem:
    def set_up(self):
        self.dataset = DataSet(batch_size)
        x, y = self.dataset.batch

        # Build model graph
        logits = self._build_graph(x)

        # Define per-example losses
        self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)

        # Define accuracy
        correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

        # Regularization
        self.regularizer = tf.losses.get_regularization_loss()
```

**PyTorch Pattern**:
```python
class TestProblem:
    def __init__(self, batch_size):
        self.dataset = DataSet()
        self.train_loader = DataLoader(dataset, batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset, batch_size, shuffle=False)
        self.model = self._build_model()

    def get_batch_loss_and_accuracy(self, batch):
        x, y = batch
        logits = self.model(x)
        losses = F.cross_entropy(logits, y, reduction='none')  # Per-example
        accuracy = (logits.argmax(1) == y).float().mean()
        return losses, accuracy
```

**Full Test Problem List**:
1. **MNIST** (4): logreg, mlp, 2c2d, vae
2. **Fashion-MNIST** (4): logreg, mlp, 2c2d, vae
3. **CIFAR-10** (3): 3c3d, vgg16, vgg19
4. **CIFAR-100** (5): 3c3d, allcnnc, vgg16, vgg19, wrn404
5. **SVHN** (2): 3c3d, wrn164
6. **ImageNet** (3): vgg16, vgg19, inception_v3
7. **Tolstoi** (1): char_rnn
8. **Synthetic** (4): quadratic_deep, 2d_rosenbrock, 2d_beale, 2d_branin

---

### 3.4 Standard Runner

**File**: `deepobs/tensorflow/runners/standard_runner.py` (24KB)

**Responsibilities**:
1. Parse command-line arguments or config
2. Initialize test problem
3. Create optimizer with hyperparameters
4. Build TensorFlow computation graph
5. Run training loop with epoch-based evaluation
6. Apply learning rate schedules
7. Log metrics (loss, accuracy, timing)
8. Save results to JSON

**TensorFlow-Specific Patterns**:
```python
# Static graph construction
tf.reset_default_graph()
tf.set_random_seed(seed)
tproblem.set_up()

# Create optimizer
learning_rate_var = tf.Variable(learning_rate, trainable=False)
opt = optimizer_class(learning_rate_var, **hyperparams)

# Training step with batch norm update dependencies
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    step = opt.minimize(loss, global_step=global_step)

# Session-based execution
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training loop
for epoch in range(num_epochs):
    # Evaluate on test set
    sess.run(tproblem.test_init_op)
    # ... compute metrics ...

    # Train
    sess.run(tproblem.train_init_op)
    while True:
        try:
            _, loss_val = sess.run([step, loss])
        except tf.errors.OutOfRangeError:
            break

    # Update learning rate
    if epoch in lr_schedule:
        sess.run(learning_rate_var.assign(lr_schedule[epoch]))
```

**PyTorch Conversion**:
```python
# No graph construction needed
torch.manual_seed(seed)
tproblem = TestProblem(batch_size)
tproblem.model.to(device)

# Create optimizer
opt = optimizer_class(tproblem.model.parameters(), lr=learning_rate, **hyperparams)
scheduler = StepLR(opt, step_size=lr_sched_epochs, gamma=lr_sched_factor)

# Training loop
for epoch in range(num_epochs):
    # Evaluate on test set
    tproblem.model.eval()
    with torch.no_grad():
        for batch in tproblem.test_loader:
            # ... compute metrics ...

    # Train
    tproblem.model.train()
    for batch in tproblem.train_loader:
        opt.zero_grad()
        losses, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
        loss = losses.mean() + tproblem.get_regularization_loss()
        loss.backward()
        opt.step()

    scheduler.step()
```

**Key Simplifications in PyTorch**:
- No session management
- No graph building/reset
- Automatic batch norm mode switching
- Simpler learning rate scheduling
- No manual UPDATE_OPS collection

---

### 3.5 Configuration System

**File**: `deepobs/tensorflow/config.py`

**Settings**:
- `get_data_dir()` / `set_data_dir()` - Where datasets are stored
- `get_baseline_dir()` / `set_baseline_dir()` - Baseline results location
- `get_dtype()` / `set_dtype()` - Float precision (float32/float64)

**PyTorch Equivalent**: Simple Python module with global variables or config class

---

## 4. TENSORFLOW → PYTORCH CONVERSION MAP

### 4.1 Core API Mapping

| TensorFlow 1.x | PyTorch |
|----------------|---------|
| `tf.layers.dense()` | `nn.Linear()` |
| `tf.layers.conv2d()` | `nn.Conv2d()` |
| `tf.layers.conv2d_transpose()` | `nn.ConvTranspose2d()` |
| `tf.layers.max_pooling2d()` | `nn.MaxPool2d()` |
| `tf.layers.batch_normalization()` | `nn.BatchNorm2d()` |
| `tf.layers.dropout()` | `nn.Dropout()` |
| `tf.nn.relu` | `torch.relu()` / `nn.ReLU()` |
| `tf.nn.sigmoid` | `torch.sigmoid()` / `nn.Sigmoid()` |
| `tf.nn.softmax_cross_entropy_with_logits_v2()` | `F.cross_entropy()` |

### 4.2 Data Pipeline Mapping

| TensorFlow 1.x | PyTorch |
|----------------|---------|
| `tf.data.Dataset.from_tensor_slices()` | `torch.utils.data.TensorDataset()` |
| `dataset.shuffle(buffer_size)` | `DataLoader(..., shuffle=True)` |
| `dataset.batch(batch_size)` | `DataLoader(..., batch_size=...)` |
| `dataset.prefetch(buffer_size)` | `DataLoader(..., num_workers=...)` |
| `iterator = dataset.make_initializable_iterator()` | `for batch in DataLoader(...)` |
| `sess.run(train_init_op)` | Switch to `train_loader` |
| `sess.run(test_init_op)` | Switch to `test_loader` |

### 4.3 Weight Initialization

| TensorFlow 1.x | PyTorch |
|----------------|---------|
| `tf.truncated_normal_initializer(stddev)` | `torch.nn.init.trunc_normal_(tensor, std)` |
| `tf.keras.initializers.glorot_normal()` | `torch.nn.init.xavier_normal_(tensor)` |
| `tf.keras.initializers.glorot_uniform()` | `torch.nn.init.xavier_uniform_(tensor)` |
| `tf.initializers.constant(value)` | `torch.nn.init.constant_(tensor, value)` |

### 4.4 Regularization

| TensorFlow 1.x | PyTorch |
|----------------|---------|
| `tf.contrib.layers.l2_regularizer(weight_decay)` | Optimizer `weight_decay` parameter |
| `tf.losses.get_regularization_loss()` | Handled automatically by optimizer |
| `tf.layers.dropout(rate, training)` | `F.dropout(p, training=self.training)` |

### 4.5 RNN Components

| TensorFlow 1.x | PyTorch |
|----------------|---------|
| `tf.contrib.rnn.LSTMCell(hidden_size)` | `nn.LSTM(input_size, hidden_size)` |
| `tf.contrib.rnn.MultiRNNCell(cells)` | `nn.LSTM(..., num_layers=...)` |
| `tf.contrib.rnn.DropoutWrapper(cell)` | `nn.LSTM(..., dropout=...)` |
| `tf.nn.static_rnn(cell, inputs)` | `lstm(x, hidden)` (dynamic) |
| State variables (trainable=False) | Return hidden state from forward |

### 4.6 Session vs Eager Execution

| TensorFlow 1.x | PyTorch |
|----------------|---------|
| `tf.Session()` | Not needed (eager by default) |
| `sess.run(op)` | Just call the function |
| `tf.Variable(value, trainable=False)` | Python variable or `torch.tensor` |
| `tf.control_dependencies([ops])` | Not needed (eager execution) |
| `tf.errors.OutOfRangeError` | Not needed (iterator exhaustion) |

---

## 5. CRITICAL CONVERSION CHALLENGES

### 5.1 Batch Normalization Updates

**TensorFlow Issue**:
```python
# Must manually add UPDATE_OPS to training step
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    step = opt.minimize(loss)
```

**PyTorch Solution**: Handled automatically via `model.train()` / `model.eval()`

---

### 5.2 Phase-Based Behavior (Train/Eval)

**TensorFlow Pattern**:
```python
# Phase variable
self.dataset.phase = tf.Variable("train", trainable=False)

# Conditional operations
dropout_rate = tf.cond(
    tf.equal(self.dataset.phase, "train"),
    lambda: 0.2,
    lambda: 0.0)
```

**PyTorch Solution**:
```python
# Automatic via module training mode
self.dropout = nn.Dropout(0.2)  # Automatically disabled in eval mode
model.train()  # Enable dropout
model.eval()   # Disable dropout
```

---

### 5.3 Per-Example Losses

**Requirement**: Return per-example losses before averaging

**TensorFlow**:
```python
self.losses = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y, logits=logits)  # Shape: (batch_size,)
loss = tf.reduce_mean(self.losses)
```

**PyTorch**:
```python
losses = F.cross_entropy(logits, y, reduction='none')  # Shape: (batch_size,)
loss = losses.mean()
```

**Critical**: Must use `reduction='none'` to preserve per-example losses

---

### 5.4 LSTM State Persistence

**TensorFlow Pattern**:
```python
# State stored in non-trainable variables
state_variables = [tf.Variable(state, trainable=False) for state in initial_states]

# Updated at end of each batch
state_update_ops = [var.assign(new_state) for var, new_state in zip(...)]

# Reset at epoch boundaries
sess.run(state_reset_ops)
```

**PyTorch Solution**:
```python
# State is just a Python variable
hidden = None  # Initialize as None

for batch in train_loader:
    output, hidden = model(batch, hidden)
    # Detach to prevent backprop through time across batches
    hidden = tuple(h.detach() for h in hidden)

# Reset at epoch boundaries
hidden = None
```

---

### 5.5 Learning Rate Schedules

**TensorFlow Pattern**:
```python
learning_rate_var = tf.Variable(learning_rate, trainable=False)
opt = optimizer_class(learning_rate_var, **hyperparams)

# Manual assignment
if epoch in lr_schedule:
    sess.run(learning_rate_var.assign(lr_schedule[epoch]))
```

**PyTorch Solution**:
```python
opt = optimizer_class(model.parameters(), lr=learning_rate, **hyperparams)
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    opt, milestones=lr_sched_epochs, gamma=lr_sched_factor)

# Automatic update
scheduler.step()
```

---

## 6. CONVERSION STRATEGY

### Phase 1: Infrastructure Setup
1. Create `deepobs/pytorch/` directory structure
2. Implement base classes:
   - `pytorch/datasets/dataset.py` (base dataset class)
   - `pytorch/testproblems/testproblem.py` (base test problem class)
   - `pytorch/config.py` (configuration management)

### Phase 2: Simple Datasets
1. Convert MNIST dataset (`mnist.py`)
2. Convert Fashion-MNIST (`fmnist.py`)
3. Convert CIFAR-10 (`cifar10.py`)
4. Convert CIFAR-100 (`cifar100.py`)
5. Test data loading and preprocessing

### Phase 3: Simple Architectures
1. Convert Logistic Regression (`_logreg.py`)
2. Convert MLP (`_mlp.py`)
3. Convert 2C2D (`_2c2d.py`)
4. Create corresponding test problems
5. Verify forward passes match TensorFlow

### Phase 4: Convolutional Networks
1. Convert 3C3D architecture
2. Convert VGG networks (`_vgg.py`)
3. Convert All-CNN-C
4. Create test problems
5. Compare outputs with TensorFlow baseline

### Phase 5: Advanced Architectures
1. Convert Wide ResNet (`_wrn.py`)
2. Convert Inception V3 (`_inception_v3.py`)
3. Convert VAE (`_vae.py`)
4. Handle batch norm carefully
5. Test reconstruction quality (VAE)

### Phase 6: RNN and Specialized Problems
1. Convert Tolstoi dataset
2. Convert Character RNN (`tolstoi_char_rnn.py`)
3. Test state persistence
4. Convert quadratic and 2D problems

### Phase 7: Runner Implementation
1. Implement `StandardRunner` in PyTorch
2. Support hyperparameter specifications
3. Implement learning rate scheduling
4. Add metric logging (loss, accuracy, timing)
5. Save results to JSON (compatible format)

### Phase 8: Analysis Tools
1. Verify analyzer works with PyTorch results
2. Test visualization scripts
3. Update example scripts
4. Create migration guide

### Phase 9: Testing and Validation
1. Unit tests for all datasets
2. Unit tests for all architectures
3. End-to-end tests for test problems
4. Compare results with TensorFlow baselines
5. Performance benchmarking

### Phase 10: Documentation
1. Update docstrings
2. Create PyTorch-specific examples
3. Write migration guide for users
4. Update README with PyTorch instructions

---

## 7. TESTING STRATEGY

### 7.1 Unit Tests

**Dataset Tests** (per dataset):
- Test data loading from files
- Verify preprocessing (normalization, augmentation)
- Check batch shapes and types
- Verify train/test split
- Test iterator behavior

**Architecture Tests** (per model):
- Test forward pass with random input
- Verify output shapes
- Check parameter count
- Test with different batch sizes
- Verify weight initialization

**Test Problem Tests**:
- Test `set_up()` method
- Verify loss computation
- Check accuracy computation
- Test regularization loss
- Verify phase switching

### 7.2 Integration Tests

**Runner Tests**:
- Test full training loop (1 epoch)
- Verify learning rate scheduling
- Test metric logging
- Check JSON output format
- Test with different optimizers

**End-to-End Tests**:
- Run small benchmark problems (MNIST, Fashion-MNIST)
- Compare loss curves with TensorFlow
- Verify final accuracy within tolerance
- Test all 26 test problems minimally

### 7.3 Regression Tests

**Baseline Comparison**:
- Run identical hyperparameters as TensorFlow
- Compare final loss (within 1% tolerance)
- Compare final accuracy (within 0.5% tolerance)
- Verify convergence speed is similar

---

## 8. DEPENDENCY MANAGEMENT

### Current Dependencies (TensorFlow)
```
- tensorflow >= 1.4.0 (tested with 1.12)
- numpy
- pandas
- matplotlib
- matplotlib2tikz (0.6.18)
- seaborn
- argparse
```

### New Dependencies (PyTorch)
```
- torch >= 1.9.0 (recommend 2.0+)
- torchvision >= 0.10.0
- numpy
- pandas
- matplotlib
- seaborn
- argparse
```

**Removed**:
- `matplotlib2tikz` (deprecated, use `tikzplotlib` if needed)

**Compatible**:
- All analysis and plotting tools remain unchanged

---

## 9. PERFORMANCE CONSIDERATIONS

### Memory Management
- PyTorch uses different memory allocation patterns
- Watch for memory leaks in data loaders (`num_workers` setting)
- Use `torch.cuda.empty_cache()` if needed for GPU

### Numerical Precision
- Default float dtype: `torch.float32`
- Match TensorFlow precision for fair comparison
- Consider mixed precision training (PyTorch AMP)

### Reproducibility
- Set seeds: `torch.manual_seed()`, `numpy.random.seed()`
- Set `torch.backends.cudnn.deterministic = True`
- Set `torch.backends.cudnn.benchmark = False`
- Note: May impact performance

---

## 10. QUICK REFERENCE: FILE LOCATIONS

### Key Files to Convert (Priority Order)

**Tier 1 - Foundation**:
1. `/deepobs/tensorflow/config.py` → `/deepobs/pytorch/config.py`
2. `/deepobs/tensorflow/datasets/dataset.py` → `/deepobs/pytorch/datasets/dataset.py`
3. `/deepobs/tensorflow/testproblems/testproblem.py` → `/deepobs/pytorch/testproblems/testproblem.py`

**Tier 2 - Simple Datasets**:
4. `/deepobs/tensorflow/datasets/mnist.py` → `/deepobs/pytorch/datasets/mnist.py`
5. `/deepobs/tensorflow/datasets/fmnist.py` → `/deepobs/pytorch/datasets/fmnist.py`

**Tier 3 - Simple Architectures**:
6. `/deepobs/tensorflow/testproblems/_logreg.py` → `/deepobs/pytorch/testproblems/_logreg.py`
7. `/deepobs/tensorflow/testproblems/_mlp.py` → `/deepobs/pytorch/testproblems/_mlp.py`

**Tier 4 - Runner**:
8. `/deepobs/tensorflow/runners/standard_runner.py` → `/deepobs/pytorch/runners/standard_runner.py`

**Tier 5 - All Remaining**:
- 6 more datasets
- 7 more architectures
- 26 test problem variants

---

## 11. GOTCHAS AND PITFALLS

### 1. Batch Norm Momentum
- TensorFlow momentum: 0.9 means 90% old, 10% new
- PyTorch momentum: 0.1 means 10% old, 90% new
- **Conversion**: `momentum_pytorch = 1 - momentum_tensorflow`

### 2. Cross-Entropy Softmax
- TensorFlow: `softmax_cross_entropy_with_logits_v2(labels, logits)`
- PyTorch: `F.cross_entropy(logits, labels)` (note: argument order reversed)
- PyTorch expects class indices (not one-hot) by default

### 3. Weight Decay vs L2 Regularization
- TensorFlow: Manual regularization loss via `kernel_regularizer`
- PyTorch: Built into optimizer via `weight_decay` parameter
- **Not exactly equivalent** for Adam optimizer (AdamW vs Adam)

### 4. Tensor Shapes
- TensorFlow: Channel-last by default (NHWC)
- PyTorch: Channel-first by default (NCHW)
- Must transpose image data: `img.permute(0, 3, 1, 2)`

### 5. RNN Batch Dimension
- TensorFlow: Often time-major (sequence_length, batch, features)
- PyTorch: Often batch-first (batch, sequence_length, features)
- Use `batch_first=True` in PyTorch LSTM for consistency

### 6. Dropout Behavior
- Both frameworks handle train/eval mode differently
- PyTorch: Must call `model.train()` / `model.eval()`
- TensorFlow: Must use `training` argument or phase variable

### 7. Variable Initialization
- TensorFlow: Graph construction time
- PyTorch: Module `__init__` time
- Ensure initialization happens at the right time for reproducibility

---

## 12. USEFUL PYTORCH PATTERNS

### Model Definition Template
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, num_outputs):
        super().__init__()
        # Define layers
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.fc = nn.Linear(64, num_outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        return self.fc(x)
```

### Dataset Template
```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data_file, transform=None):
        self.data = self._load_data(data_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# Usage
dataset = MyDataset(data_file)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

### Training Loop Template
```python
model.train()
for epoch in range(num_epochs):
    for batch_idx, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print(f'Epoch {epoch}, Batch {batch_idx}, Loss {loss.item():.4f}')
```

### Evaluation Loop Template
```python
model.eval()
total_loss = 0
correct = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        outputs = model(x)
        loss = criterion(outputs, y)
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += (pred == y).sum().item()

avg_loss = total_loss / len(test_loader)
accuracy = correct / len(test_loader.dataset)
```

---

## 13. NEXT STEPS

### Immediate Actions
1. Set up PyTorch development environment
2. Create `deepobs/pytorch/` directory structure
3. Implement base `DataSet` class
4. Implement base `TestProblem` class
5. Convert MNIST dataset as proof-of-concept

### Short-Term Goals
1. Convert all 10 datasets
2. Convert 3 simple architectures (logreg, mlp, 2c2d)
3. Implement basic `StandardRunner`
4. Create 3 test problems and verify they work

### Medium-Term Goals
1. Convert all 9 architectures
2. Create all 26 test problems
3. Full runner implementation with all features
4. Unit tests for all components

### Long-Term Goals
1. Performance comparison with TensorFlow version
2. Documentation and migration guide
3. Community testing and feedback
4. Publication and release

---

## 14. RESOURCES

### PyTorch Documentation
- Official Docs: https://pytorch.org/docs/stable/index.html
- Tutorials: https://pytorch.org/tutorials/
- Migration Guide: https://pytorch.org/tutorials/beginner/former_torchies/tensor_tutorial.html

### Reference Implementations
- TorchVision Models: https://github.com/pytorch/vision/tree/main/torchvision/models
- PyTorch Examples: https://github.com/pytorch/examples

### DeepOBS Original
- Paper: https://openreview.net/forum?id=rJg6ssC5Y7
- GitHub: https://github.com/fsschneider/DeepOBS
- Docs: https://deepobs.readthedocs.io/

### Successor Project
- AlgoPerf: https://github.com/mlcommons/algorithmic-efficiency

---

## 15. CONVERSION CHECKLIST

- [ ] Set up PyTorch environment and dependencies
- [ ] Create `deepobs/pytorch/` directory structure
- [ ] Implement base `DataSet` class
- [ ] Implement base `TestProblem` class
- [ ] Implement `config.py`
- [ ] Convert MNIST dataset
- [ ] Convert Fashion-MNIST dataset
- [ ] Convert CIFAR-10 dataset
- [ ] Convert CIFAR-100 dataset
- [ ] Convert SVHN dataset
- [ ] Convert ImageNet dataset
- [ ] Convert Tolstoi dataset
- [ ] Convert Quadratic dataset
- [ ] Convert Two-D dataset
- [ ] Implement Logistic Regression architecture
- [ ] Implement MLP architecture
- [ ] Implement 2C2D architecture
- [ ] Implement 3C3D architecture
- [ ] Implement VGG architecture
- [ ] Implement Wide ResNet architecture
- [ ] Implement Inception V3 architecture
- [ ] Implement VAE architecture
- [ ] Implement All-CNN-C architecture
- [ ] Implement Character RNN architecture
- [ ] Create all 26 test problems
- [ ] Implement `StandardRunner`
- [ ] Implement runner utilities
- [ ] Write unit tests for datasets
- [ ] Write unit tests for architectures
- [ ] Write unit tests for test problems
- [ ] Write integration tests for runner
- [ ] Compare results with TensorFlow baselines
- [ ] Update documentation
- [ ] Create migration guide
- [ ] Update README
- [ ] Create example scripts
- [ ] Performance benchmarking
- [ ] Release PyTorch version

---

**Last Updated**: 2025-12-13
**Document Version**: 1.0
**Status**: Ready for conversion
