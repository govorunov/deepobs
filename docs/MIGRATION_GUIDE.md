# DeepOBS Migration Guide: TensorFlow → PyTorch

**Version**: 1.2.0-pytorch
**Target Audience**: Users migrating from DeepOBS TensorFlow to PyTorch
**Last Updated**: 2025-12-14

---

## Table of Contents

- [Overview](#overview)
- [Quick Comparison](#quick-comparison)
- [API Changes](#api-changes)
- [Key Differences](#key-differences)
- [Migration Checklist](#migration-checklist)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)
- [Known Issues](#known-issues)

---

## Overview

The PyTorch version of DeepOBS maintains API compatibility with the TensorFlow version while leveraging PyTorch's modern features. Most code can be migrated by changing imports and adapting to PyTorch's eager execution model.

### Why Migrate?

- **Simpler Code**: No session management or graph construction
- **Eager Execution**: Easier debugging with standard Python debugging tools
- **Modern Features**: Access to PyTorch's latest optimizations and features
- **Better Performance**: Potential speed improvements on modern hardware
- **Active Development**: PyTorch has a large, active community

### Compatibility

The PyTorch version:
- ✅ Maintains the same test problems (all 26)
- ✅ Uses identical dataset preprocessing
- ✅ Preserves model architectures and initializations
- ✅ Produces comparable numerical results
- ✅ Supports the same StandardRunner API

---

## Quick Comparison

### Side-by-Side Example

#### TensorFlow Version

```python
import tensorflow as tf
from deepobs.tensorflow import testproblems

# Reset graph
tf.reset_default_graph()

# Create test problem
tproblem = testproblems.mnist_mlp(batch_size=128)
tproblem.set_up()

# Create optimizer
opt = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)

# Build training op
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = opt.minimize(tproblem.losses.mean())

# Run training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(10):
        # Training
        sess.run(tproblem.train_init_op)
        while True:
            try:
                _, loss_val = sess.run([train_op, tproblem.losses])
            except tf.errors.OutOfRangeError:
                break

        # Evaluation
        sess.run(tproblem.test_init_op)
        test_loss = []
        while True:
            try:
                loss_val = sess.run(tproblem.losses)
                test_loss.append(loss_val.mean())
            except tf.errors.OutOfRangeError:
                break
        print(f'Epoch {epoch}, Test Loss: {np.mean(test_loss):.4f}')
```

#### PyTorch Version

```python
import torch
from deepobs.pytorch import testproblems

# Create test problem
tproblem = testproblems.mnist_mlp(batch_size=128)
tproblem.set_up()

# Create optimizer
optimizer = torch.optim.SGD(
    tproblem.model.parameters(),
    lr=0.01,
    momentum=0.9
)

# Run training
for epoch in range(10):
    # Training
    tproblem.model.train()
    for batch in tproblem.dataset.train_loader:
        optimizer.zero_grad()
        loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
        loss.mean().backward()
        optimizer.step()

    # Evaluation
    tproblem.model.eval()
    test_losses = []
    with torch.no_grad():
        for batch in tproblem.dataset.test_loader:
            loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
            test_losses.append(loss.mean().item())
    print(f'Epoch {epoch}, Test Loss: {np.mean(test_losses):.4f}')
```

**Key Differences**:
- ❌ No `tf.Session()` in PyTorch
- ❌ No graph construction or reset
- ❌ No `train_init_op` / `test_init_op`
- ✅ Direct Python for loops over data
- ✅ Automatic batch norm handling via `train()` / `eval()`
- ✅ Simpler, more Pythonic code

---

## API Changes

### Import Statements

```python
# TensorFlow
from deepobs.tensorflow import testproblems
from deepobs.tensorflow import config
from deepobs.tensorflow.runners import StandardRunner

# PyTorch
from deepobs.pytorch import testproblems
from deepobs.pytorch import config
from deepobs.pytorch.runners import StandardRunner
```

### Test Problem Creation

```python
# Both versions (identical API)
tproblem = testproblems.mnist_mlp(batch_size=128)
tproblem.set_up()
```

### Accessing Data

```python
# TensorFlow
sess.run(tproblem.train_init_op)  # Initialize training iterator
batch = sess.run(tproblem.batch)  # Get batch

# PyTorch
for batch in tproblem.dataset.train_loader:  # Direct iteration
    images, labels = batch
```

### Computing Loss

```python
# TensorFlow
losses = tproblem.losses  # Tensor in graph
loss = tf.reduce_mean(losses)  # Graph operation
loss_val = sess.run(loss)  # Execute to get value

# PyTorch
loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)  # Immediate execution
loss = loss.mean()  # Eager computation
loss_val = loss.item()  # Get Python scalar
```

### Optimizer Creation

```python
# TensorFlow
opt = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9)
train_op = opt.minimize(loss)

# PyTorch
optimizer = torch.optim.SGD(
    tproblem.model.parameters(),
    lr=0.01,
    momentum=0.9
)
```

### Training Step

```python
# TensorFlow
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = opt.minimize(loss)
sess.run(train_op)

# PyTorch
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### Model Evaluation

```python
# TensorFlow
sess.run(tproblem.test_init_op)
# Dropout/BN controlled by phase variable or training flag

# PyTorch
tproblem.model.eval()  # Automatic dropout/BN adjustment
with torch.no_grad():  # Disable gradient computation
    # Evaluation code
```

---

## Key Differences

### 1. Execution Model

| Feature | TensorFlow 1.x | PyTorch |
|---------|----------------|---------|
| **Execution** | Static graph | Eager (dynamic) |
| **Debugging** | Difficult (symbolic) | Easy (standard Python) |
| **Control Flow** | tf.cond, tf.while_loop | Python if, for, while |
| **Sessions** | Required | Not needed |
| **Graph Building** | Separate phase | Inline with execution |

### 2. Batch Normalization

#### TensorFlow Approach

```python
# Must manually add UPDATE_OPS to training step
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
```

#### PyTorch Approach

```python
# Automatic via model mode
model.train()  # BN updates enabled
model.eval()   # BN updates disabled
```

**CRITICAL**: Batch norm momentum definitions are inverse!

```python
# TensorFlow: momentum=0.9 means 90% old, 10% new
tf.layers.batch_normalization(..., momentum=0.9)

# PyTorch: momentum=0.1 means 10% old, 90% new
nn.BatchNorm2d(..., momentum=0.1)

# Conversion formula
pytorch_momentum = 1.0 - tensorflow_momentum
```

### 3. Data Pipeline

| Aspect | TensorFlow | PyTorch |
|--------|------------|---------|
| **API** | tf.data.Dataset | torch.utils.data.DataLoader |
| **Iteration** | Iterator with init ops | Direct Python iteration |
| **Shuffling** | dataset.shuffle() | DataLoader(shuffle=True) |
| **Batching** | dataset.batch() | DataLoader(batch_size=...) |
| **Prefetching** | dataset.prefetch() | DataLoader(num_workers=...) |
| **End of Epoch** | OutOfRangeError | Iterator exhaustion |

### 4. Loss Functions

#### TensorFlow

```python
# Per-example losses (required for DeepOBS)
losses = tf.nn.softmax_cross_entropy_with_logits_v2(
    labels=y,
    logits=logits
)
# Note: labels come first

# Mean loss
loss = tf.reduce_mean(losses)
```

#### PyTorch

```python
# Per-example losses (required for DeepOBS)
losses = F.cross_entropy(
    logits,
    y,
    reduction='none'
)
# Note: logits come first, reduction='none' for per-example

# Mean loss
loss = losses.mean()
```

### 5. Weight Initialization

| Method | TensorFlow | PyTorch |
|--------|------------|---------|
| **Truncated Normal** | tf.truncated_normal_initializer(stddev=0.03) | torch.nn.init.trunc_normal_(tensor, std=0.03) |
| **Xavier Normal** | tf.keras.initializers.glorot_normal() | torch.nn.init.xavier_normal_(tensor) |
| **Xavier Uniform** | tf.keras.initializers.glorot_uniform() | torch.nn.init.xavier_uniform_(tensor) |
| **Constant** | tf.initializers.constant(value) | torch.nn.init.constant_(tensor, value) |

### 6. Regularization

#### TensorFlow: Manual Regularization

```python
# Define regularizer
regularizer = tf.contrib.layers.l2_regularizer(weight_decay)

# Apply to layers
conv = tf.layers.conv2d(..., kernel_regularizer=regularizer)

# Collect and add to loss
reg_loss = tf.losses.get_regularization_loss()
total_loss = loss + reg_loss
```

#### PyTorch: Optimizer Weight Decay

```python
# Built into optimizer
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    weight_decay=1e-4  # L2 regularization
)

# Note: For Adam, use AdamW for proper weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-2
)
```

**IMPORTANT**: For Adam optimizer, `weight_decay` in standard `Adam` is not equivalent to L2 regularization. Use `AdamW` for proper weight decay.

### 7. RNN State Management

#### TensorFlow: State Variables

```python
# Create state variables (non-trainable)
state_vars = [tf.Variable(init_state, trainable=False) for init_state in initial_states]

# Update state after forward pass
state_update_ops = [var.assign(new_state) for var, new_state in zip(state_vars, new_states)]

# Reset at epoch boundaries
sess.run([var.assign(init_state) for var, init_state in zip(state_vars, initial_states)])
```

#### PyTorch: Simple Hidden State

```python
# State is just a Python variable
hidden = None  # Initialize

# Forward pass
output, hidden = lstm(input, hidden)

# Detach to prevent backprop through time across batches
hidden = tuple(h.detach() for h in hidden)

# Reset at epoch boundaries
hidden = None  # Simply reset to None
```

### 8. Learning Rate Scheduling

#### TensorFlow: Manual Assignment

```python
# Create learning rate variable
learning_rate_var = tf.Variable(learning_rate, trainable=False)
optimizer = tf.train.SGD(learning_rate_var)

# Manually update
if epoch in lr_schedule:
    sess.run(learning_rate_var.assign(lr_schedule[epoch]))
```

#### PyTorch: Built-in Schedulers

```python
# Create scheduler
from torch.optim.lr_scheduler import MultiStepLR

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
scheduler = MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.1)

# Step at epoch end
for epoch in range(num_epochs):
    # Training...
    scheduler.step()
```

---

## Migration Checklist

### Phase 1: Environment Setup

- [ ] Install PyTorch: `pip install torch torchvision`
- [ ] Install DeepOBS PyTorch version
- [ ] Verify imports work: `import deepobs.pytorch`
- [ ] Set up data directory: `config.set_data_dir('/path/to/data')`

### Phase 2: Code Migration

- [ ] Change imports from `deepobs.tensorflow` to `deepobs.pytorch`
- [ ] Replace TensorFlow optimizers with PyTorch equivalents
- [ ] Remove session management code
- [ ] Remove graph construction/reset code
- [ ] Replace dataset iterators with DataLoader iteration
- [ ] Update batch norm momentum values (if applicable)
- [ ] Replace manual UPDATE_OPS with `model.train()` / `model.eval()`
- [ ] Update loss computation to use `get_batch_loss_and_accuracy()`
- [ ] Replace `sess.run()` calls with direct execution
- [ ] Update regularization from manual to optimizer weight_decay

### Phase 3: Training Loop

- [ ] Remove `train_init_op` / `test_init_op` calls
- [ ] Replace TensorFlow's `OutOfRangeError` handling with standard iteration
- [ ] Add `optimizer.zero_grad()` before forward pass
- [ ] Replace `sess.run(train_op)` with `loss.backward(); optimizer.step()`
- [ ] Add `with torch.no_grad():` for evaluation
- [ ] Update learning rate scheduling to use PyTorch schedulers
- [ ] Verify model mode switching (`train()` / `eval()`)

### Phase 4: Testing

- [ ] Run one epoch and verify loss computation
- [ ] Compare initial loss with TensorFlow version (should be close)
- [ ] Verify gradient flow (losses should decrease)
- [ ] Check batch norm behavior (train vs eval mode)
- [ ] Test on GPU if available
- [ ] Compare final accuracy with TensorFlow baselines (within tolerance)
- [ ] Verify reproducibility with fixed seeds

### Phase 5: Optimization

- [ ] Enable GPU if available
- [ ] Add DataLoader num_workers for faster loading
- [ ] Consider mixed precision training (AMP)
- [ ] Profile for bottlenecks
- [ ] Optimize memory usage if needed

---

## Common Patterns

### Pattern 1: Basic Training Loop

#### TensorFlow

```python
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        sess.run(tproblem.train_init_op)
        while True:
            try:
                _, loss = sess.run([train_op, loss_tensor])
            except tf.errors.OutOfRangeError:
                break
```

#### PyTorch

```python
for epoch in range(num_epochs):
    tproblem.model.train()
    for batch in tproblem.dataset.train_loader:
        optimizer.zero_grad()
        loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
        loss.mean().backward()
        optimizer.step()
```

### Pattern 2: Evaluation

#### TensorFlow

```python
sess.run(tproblem.test_init_op)
test_losses = []
while True:
    try:
        loss = sess.run(tproblem.losses)
        test_losses.append(loss.mean())
    except tf.errors.OutOfRangeError:
        break
avg_loss = np.mean(test_losses)
```

#### PyTorch

```python
tproblem.model.eval()
test_losses = []
with torch.no_grad():
    for batch in tproblem.dataset.test_loader:
        loss, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
        test_losses.append(loss.mean().item())
avg_loss = np.mean(test_losses)
```

### Pattern 3: Learning Rate Decay

#### TensorFlow

```python
lr_var = tf.Variable(0.1, trainable=False)
if epoch == 60:
    sess.run(lr_var.assign(0.01))
```

#### PyTorch

```python
scheduler = MultiStepLR(optimizer, milestones=[60], gamma=0.1)
scheduler.step()  # Call at epoch end
```

### Pattern 4: Custom Training Metrics

#### TensorFlow

```python
correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
acc_val = sess.run(accuracy)
```

#### PyTorch

```python
predictions = logits.argmax(dim=1)
correct = (predictions == labels).float()
accuracy = correct.mean()
acc_val = accuracy.item()
```

### Pattern 5: Gradient Clipping

#### TensorFlow

```python
optimizer = tf.train.AdamOptimizer(learning_rate)
gradients, variables = zip(*optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, max_norm)
train_op = optimizer.apply_gradients(zip(gradients, variables))
```

#### PyTorch

```python
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer.zero_grad()
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
optimizer.step()
```

---

## Troubleshooting

### Issue: Different numerical results

**Possible Causes**:
1. Different random seeds
2. Batch norm momentum not converted
3. Different weight initialization
4. Different data augmentation randomness

**Solutions**:
```python
# Ensure identical seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Check batch norm momentum
# TF momentum=0.9 → PyTorch momentum=0.1

# Verify initialization matches TensorFlow
# Check dataset preprocessing
```

### Issue: CUDA out of memory

**Solutions**:
```python
# Reduce batch size
tproblem = testproblems.mnist_mlp(batch_size=64)  # Instead of 128

# Clear cache
torch.cuda.empty_cache()

# Use gradient accumulation
accumulation_steps = 2
for i, batch in enumerate(train_loader):
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue: Model not learning

**Checklist**:
- [ ] Is `optimizer.zero_grad()` called before forward pass?
- [ ] Is `loss.backward()` called?
- [ ] Is `optimizer.step()` called?
- [ ] Is model in training mode (`model.train()`)?
- [ ] Is learning rate reasonable?
- [ ] Are gradients flowing? (check with `loss.requires_grad`)

### Issue: Batch normalization not working

**Solutions**:
```python
# Ensure model mode is set
model.train()  # For training
model.eval()   # For evaluation

# Check momentum conversion
# PyTorch momentum = 1 - TensorFlow momentum

# Verify BN layers exist
for name, module in model.named_modules():
    if isinstance(module, torch.nn.BatchNorm2d):
        print(f'{name}: momentum={module.momentum}')
```

### Issue: Different loss values than TensorFlow

**Checks**:
```python
# Verify label format (class indices vs one-hot)
# TensorFlow often uses one-hot, PyTorch uses class indices
labels = labels.argmax(dim=1)  # Convert one-hot to indices if needed

# Check reduction
losses = F.cross_entropy(logits, labels, reduction='none')  # Per-example
loss = losses.mean()  # Final loss

# Verify regularization
# TensorFlow: manual reg loss
# PyTorch: optimizer weight_decay
```

---

## Known Issues

### 1. ImageNet Dataset

**Issue**: ImageNet requires manual download and organization.

**Solution**: Download ImageNet from [image-net.org](http://www.image-net.org/) and organize as:
```
<data_dir>/imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

### 2. Batch Norm Momentum

**Issue**: TensorFlow and PyTorch define momentum inversely.

**Solution**: Always convert:
```python
pytorch_momentum = 1.0 - tensorflow_momentum
```

### 3. Adam Weight Decay

**Issue**: Standard `torch.optim.Adam` with `weight_decay` is not equivalent to L2 regularization.

**Solution**: Use `AdamW` for proper weight decay:
```python
# Instead of
optimizer = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-2)

# Use
optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-2)
```

### 4. Data Format

**Issue**: TensorFlow uses NHWC (batch, height, width, channels), PyTorch uses NCHW (batch, channels, height, width).

**Solution**: DeepOBS PyTorch handles this automatically. If loading TensorFlow checkpoints, transpose:
```python
# TensorFlow format: [H, W, C_in, C_out]
# PyTorch format: [C_out, C_in, H, W]
pytorch_weights = tf_weights.permute(3, 2, 0, 1)
```

### 5. RNN Cell vs Module

**Issue**: TensorFlow uses `LSTMCell` (single timestep), PyTorch `LSTM` processes full sequences.

**Solution**: Use `batch_first=True` and let PyTorch handle sequence:
```python
# PyTorch LSTM processes entire sequence
lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True)
output, (h_n, c_n) = lstm(input_sequence, (h_0, c_0))
```

---

## Performance Comparison

Expected performance differences:

| Aspect | TensorFlow 1.x | PyTorch | Notes |
|--------|----------------|---------|-------|
| **Training Speed** | Baseline | 0.9-1.1x | Similar, depends on hardware |
| **Memory Usage** | Baseline | 0.9-1.0x | PyTorch often more efficient |
| **Debugging** | Difficult | Easy | Major PyTorch advantage |
| **Startup Time** | Slow (graph build) | Fast (eager) | PyTorch 2-5x faster |
| **Code Complexity** | High | Low | PyTorch ~30% less code |

---

## Additional Resources

- **PyTorch Documentation**: https://pytorch.org/docs/
- **PyTorch Tutorials**: https://pytorch.org/tutorials/
- **DeepOBS Paper**: https://openreview.net/forum?id=rJg6ssC5Y7
- **API Reference**: See [API_REFERENCE.md](API_REFERENCE.md)
- **Examples**: See [examples/](examples/) directory

---

## Migration Support

If you encounter issues not covered in this guide:

1. Check the [examples/](examples/) directory for working code
2. Review [API_REFERENCE.md](API_REFERENCE.md) for detailed API documentation
3. Open an issue on [GitHub](https://github.com/fsschneider/DeepOBS/issues)
4. Check PyTorch migration resources

---

**Last Updated**: 2025-12-14
**PyTorch Version**: >= 1.9.0
**DeepOBS Version**: 1.2.0-pytorch
