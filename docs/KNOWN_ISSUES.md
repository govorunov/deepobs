# DeepOBS PyTorch - Known Issues and Limitations

**Version**: 1.2.0-pytorch
**Last Updated**: 2025-12-15

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
