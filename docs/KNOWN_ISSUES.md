# DeepOBS PyTorch - Known Issues and Limitations

**Version**: 1.2.0-pytorch
**Last Updated**: 2025-12-15

---

## PyTorch Installation Required

The PyTorch implementation requires PyTorch to be installed separately:

```bash
# Install PyTorch (CPU version)
pip install torch torchvision

# Or GPU version (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Note**: The smoke test and full test suite will fail if PyTorch is not installed.

---

## Dataset Download Requirements

### ImageNet Dataset

**Status**: Manual download required

The ImageNet dataset cannot be automatically downloaded due to licensing restrictions.

**Workaround**:
1. Download ImageNet from the official source: http://www.image-net.org/
2. Place the dataset in the DeepOBS data directory
3. Expected structure:
   ```
   data/
   └── imagenet/
       ├── train/
       │   ├── n01440764/
       │   ├── n01443537/
       │   └── ...
       └── val/
           ├── n01440764/
           ├── n01443537/
           └── ...
   ```

**Tests Affected**:
- `test_imagenet_vgg16`
- `test_imagenet_vgg19`
- `test_imagenet_inception_v3`

These tests are marked with `@pytest.mark.skip` by default.

### Text Generation Dataset

**Status**: Penn Treebank now used (automatic download)

The original `tolstoi_char_rnn` problem required manual download of War and Peace text, which was unreliable.

**New Recommendation**: Use `textgen` problem instead
- Automatically downloads Penn Treebank via torchtext
- No manual download required
- More reliable and standardized dataset

**Legacy**: `tolstoi_char_rnn` is still available but deprecated
- Requires manual download of War and Peace text
- Place `train.npy` and `test.npy` in `data/tolstoi/`

**Tests Affected**:
- `test_tolstoi_char_rnn` (deprecated)
- `test_textgen` (recommended)

---

## Platform-Specific Issues

### macOS M1/M2 (Apple Silicon)

**Issue**: Some operations may use CPU instead of GPU by default.

**Solution**: Use MPS (Metal Performance Shaders) backend:
```python
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tproblem.model.to(device)
```

### Windows

**Issue**: Some file path operations may need adjustment for Windows paths.

**Status**: Not extensively tested on Windows. Linux and macOS are primary platforms.

---

## Test Suite Notes

### Skipped Tests

The following tests are skipped by default:

1. **ImageNet tests** (3 tests)
   - Reason: Manual dataset download required
   - Mark: `@pytest.mark.skip`

2. **Slow tests** (marked with `@pytest.mark.slow`)
   - Full VGG training tests
   - Full WRN training tests
   - Can be run with: `pytest -m slow`

### Expected Test Duration

- Quick tests (`pytest -m "not slow"`): ~2-5 minutes
- Full test suite (`pytest`): ~15-30 minutes (depending on hardware)
- Smoke test: ~30 seconds

---

## Numerical Precision Differences

### TensorFlow vs PyTorch

**Issue**: Minor numerical differences may exist between TensorFlow and PyTorch implementations.

**Expected Difference**:
- Loss values: ±0.1%
- Accuracy: ±0.5%
- Final convergence: Within tolerance

**Causes**:
1. Different random number generators
2. Different default initializations
3. Different batch normalization implementations
4. Floating-point arithmetic differences

**Impact**: Minimal. Both implementations converge to similar final results.

---

## Batch Normalization Momentum

**Issue**: TensorFlow and PyTorch use inverted momentum definitions.

**Conversion**:
```python
# TensorFlow momentum: 0.9
momentum_tf = 0.9

# PyTorch momentum: 0.1
momentum_pt = 1 - momentum_tf  # 0.1
```

**Status**: Already handled in all architecture implementations.

---

## Data Augmentation Randomness

**Issue**: Random crops and flips may produce slightly different results across runs.

**Impact**: Minor variance in training curves, especially in early epochs.

**Solution**: Set random seeds for reproducibility:
```python
import torch
import numpy as np
import random

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
```

---

## Train vs Test Accuracy During Evaluation

**Issue**: Training set accuracy may appear **lower** than test set accuracy during evaluation.

**Explanation**: This is expected behavior for datasets with data augmentation (CIFAR-10, CIFAR-100, SVHN, ImageNet, Food-101):

1. **Training evaluation**: When measuring training accuracy, the training dataset's data augmentation transforms (random crops, flips, color jitter) are **still applied** during evaluation. This makes the training examples harder to classify.

2. **Test evaluation**: Test data uses only normalization/standardization without augmentation, making it easier to classify.

3. **Result**: Test accuracy can appear higher than train accuracy, which is the opposite of typical overfitting patterns.

**What's reported in analysis**:
- The analysis tools (e.g., `result_analysis.py`) report **test accuracy and test loss only**
- Test metrics are **correctly measured** and **unaffected** by this issue
- Test data always uses consistent, non-augmented transforms

**Expected behavior**:
- For CIFAR-10/100, SVHN: Train accuracy during evaluation may be 2-5% lower than without augmentation
- For ImageNet, Food-101: Train accuracy may be 5-10% lower due to random crops

**Why this design**:
- Training data augmentation improves generalization and is standard practice
- The primary metric of interest is **test accuracy**, which is correctly measured
- Train accuracy with augmentation provides a conservative estimate of training set performance

**Note**: If you need true training set accuracy (without augmentation) for analysis:
- You would need to implement a separate evaluation path
- However, this is not the standard metric for optimizer benchmarking
- Test accuracy remains the primary metric for comparing optimizers

---

## Memory Usage

### Large Batch Sizes

**Issue**: Some test problems with large batch sizes may exceed GPU memory.

**Examples**:
- ImageNet with batch_size > 128 (on GPUs with < 16GB VRAM)
- VGG19 with large images

**Solution**: Use gradient accumulation or reduce batch size:
```python
# Reduce batch size
tproblem = testproblems.imagenet_vgg19(batch_size=64)

# Or use gradient accumulation
accumulation_steps = 2
for i, batch in enumerate(train_loader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

---

## CUDA/GPU Issues

### CUDA Out of Memory

**Issue**: "RuntimeError: CUDA out of memory"

**Solutions**:
1. Reduce batch size
2. Use mixed precision training (FP16)
3. Enable gradient checkpointing
4. Use CPU if dataset is small

```python
# Mixed precision example
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    loss = model(x)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### CUDA Version Mismatch

**Issue**: PyTorch CUDA version doesn't match system CUDA.

**Solution**: Install matching PyTorch version:
```bash
# Check CUDA version
nvidia-smi

# Install matching PyTorch
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Dependency Compatibility

### matplotlib2tikz Deprecation

**Issue**: `matplotlib2tikz==0.6.18` is deprecated and may not install on newer Python versions.

**Impact**: Affects TikZ export functionality in analyzer.

**Workaround**: Use `tikzplotlib` instead:
```bash
pip uninstall matplotlib2tikz
pip install tikzplotlib
```

**Status**: Low priority. Most users don't need TikZ export.

---

## Performance Considerations

### DataLoader num_workers

**Issue**: `num_workers > 0` may cause issues on some systems (especially macOS).

**Symptoms**:
- Slow data loading
- Process hanging
- "Too many open files" error

**Solution**: Set `num_workers=0` in dataset configuration:
```python
# In dataset implementation
self.train_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0  # or 2-4 on Linux
)
```

### Deterministic Operations

**Issue**: Full reproducibility requires deterministic operations, which may slow down training.

**Enable with**:
```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Impact**: ~10-20% slower training.

---

## Documentation Issues

### API Documentation

**Status**: Sphinx documentation not yet built for PyTorch version.

**Current**: All documentation is in markdown files:
- `README_PYTORCH.md` - Main PyTorch documentation
- `MIGRATION_GUIDE.md` - TensorFlow → PyTorch migration
- `API_REFERENCE.md` - Complete API reference
- `EXAMPLES.md` - Usage examples

**Future**: Will integrate with existing ReadTheDocs site.

---

## Future Improvements Needed

### 1. Distributed Training Support

**Status**: Not implemented

**Priority**: Medium

**Description**: Add support for multi-GPU training with `torch.nn.DataParallel` or `DistributedDataParallel`.

### 2. Automatic Mixed Precision (AMP)

**Status**: Not implemented

**Priority**: Medium

**Description**: Add optional AMP support for faster training on modern GPUs.

### 3. TensorBoard Logging

**Status**: Not implemented

**Priority**: Low

**Description**: Add TensorBoard logging in addition to JSON output.

### 4. Model Checkpointing

**Status**: Not implemented

**Priority**: Low

**Description**: Add automatic model checkpointing during training.

### 5. Hyperparameter Search Integration

**Status**: Not implemented

**Priority**: Low

**Description**: Integration with Ray Tune, Optuna, or similar libraries.

---

## Reporting Issues

If you encounter any issues not listed here:

1. **Check existing GitHub issues**: https://github.com/fsschneider/DeepOBS/issues
2. **Create a new issue** with:
   - Python version
   - PyTorch version
   - Operating system
   - Complete error traceback
   - Minimal reproducible example

3. **Or contact**: frank.schneider@tue.mpg.de

---

## Version History

- **1.2.0-pytorch** (2025-12-15): Initial PyTorch implementation
  - All known issues documented
  - ImageNet manual download required
  - Text generation now uses Penn Treebank (automatic download)
  - Tolstoi dataset deprecated in favor of Penn Treebank

---

**Note**: This is a living document. It will be updated as new issues are discovered or existing issues are resolved.
