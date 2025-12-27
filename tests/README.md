# DeepOBS PyTorch - Testing Guide

This directory contains comprehensive tests for the DeepOBS PyTorch implementation.

## Test Structure

```
tests/
├── test_utils.py              # Testing utilities and helpers
├── test_datasets.py           # Tests for all 8 datasets
├── test_architectures.py      # Tests for all 8 architectures
├── test_testproblems.py       # Tests for all 25 test problems
├── test_runner.py             # Tests for StandardRunner
├── test_config.py             # Tests for configuration system
├── integration/               # Integration tests
│   ├── test_end_to_end.py     # Full training runs
│   └── test_all_problems.py   # Smoke test for all 25 problems
└── README.md                  # This file
```

## Quick Start

### Run All Tests

```bash
# From project root
pytest tests/

# With verbose output
pytest tests/ -v

# With coverage report
pytest tests/ --cov=deepobs.pytorch
```

### Run Specific Test Files

```bash
# Dataset tests only
pytest tests/test_datasets.py

# Architecture tests only
pytest tests/test_architectures.py

# Test problem tests only
pytest tests/test_testproblems.py

# Integration tests only
pytest tests/integration/
```

### Run Quick Smoke Test

For rapid validation without full test suite:

```bash
python tests/smoke_test.py
```

This runs basic import, instantiation, and training tests on simple problems.

## Test Categories

### 1. Dataset Tests (`test_datasets.py`)

Tests for all 8 datasets:
- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- SVHN
- ImageNet (requires manual setup - skipped by default)
- Quadratic
- Two-D

**What's tested:**
- Dataset instantiation
- Data loading (train and test)
- Batch shapes and types
- Reproducibility with fixed seeds
- DataLoader integration

### 2. Architecture Tests (`test_architectures.py`)

Tests for all 8 architecture types:
- Logistic Regression
- Multi-Layer Perceptron (MLP)
- 2C2D (2 conv + 2 dense)
- 3C3D (3 conv + 3 dense)
- VGG (VGG16, VGG19)
- Wide Residual Network (WRN)
- Inception V3
- Variational Autoencoder (VAE)
- All-CNN-C

**What's tested:**
- Model instantiation
- Forward pass with correct output shapes
- Backward pass and gradient flow
- Parameter counting
- Train/eval mode switching
- Different batch sizes
- GPU compatibility (if available)

### 3. Test Problem Tests (`test_testproblems.py`)

Tests for all 25 test problems:

**MNIST (4):** logreg, mlp, 2c2d, vae
**Fashion-MNIST (4):** logreg, mlp, 2c2d, vae
**CIFAR-10 (3):** 3c3d, vgg16, vgg19
**CIFAR-100 (5):** 3c3d, allcnnc, vgg16, vgg19, wrn404
**SVHN (2):** 3c3d, wrn164
**ImageNet (3):** vgg16, vgg19, inception_v3 (requires manual setup)
**Synthetic (4):** quadratic_deep, two_d_rosenbrock, two_d_beale, two_d_branin

**What's tested:**
- Problem instantiation
- Model and data loader availability
- Forward pass
- Backward pass with gradients
- Loss and accuracy computation
- Regularization
- Train/eval mode switching
- Reproducibility

### 4. Runner Tests (`test_runner.py`)

Tests for the StandardRunner:
- Runner initialization
- Training for single/multiple epochs
- Different optimizers (SGD, Adam)
- Learning rate scheduling
- Output file creation
- Metric logging (loss, accuracy, timing)
- JSON output format
- Reproducibility

### 5. Config Tests (`test_config.py`)

Tests for configuration system:
- Get/set data directory
- Get/set baseline directory
- Get/set dtype (float32/float64)
- Path handling (absolute/relative)
- Config persistence

### 6. Integration Tests (`integration/`)

#### End-to-End Tests (`test_end_to_end.py`)

Full training runs on small problems:
- MNIST logistic regression (3 epochs)
- MNIST MLP (3 epochs)
- Fashion-MNIST 2C2D (2 epochs)
- Verifies loss decreases and accuracy increases
- Tests different optimizers
- Tests learning rate scheduling

#### All Problems Test (`test_all_problems.py`)

Quick smoke test of all 25 problems:
- Instantiates each problem
- Runs 1 forward + backward pass
- Verifies no errors
- Reports PASS/FAILED for each problem
- Generates coverage summary

## Test Markers

Tests are marked with pytest markers for selective execution:

- `@pytest.mark.slow` - Long-running tests (multi-epoch training)
- `@pytest.mark.skip` - Tests requiring manual setup (ImageNet)

### Run Only Fast Tests

```bash
pytest tests/ -m "not slow"
```

### Run Slow Tests

```bash
pytest tests/ -m "slow"
```

### Include Skipped Tests

```bash
pytest tests/ --runxfail
```

## Data Requirements

### Auto-Download Datasets

These datasets are automatically downloaded on first use:
- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100
- SVHN
- Quadratic (synthetic)
- Two-D (synthetic)

### Manual Setup Required

This dataset requires manual download and setup:
- **ImageNet**: Download from http://image-net.org and place in data directory

Tests for this dataset are skipped by default.

## Expected Test Results

### Passing Tests

All tests should pass if:
1. Dependencies are installed correctly
2. Auto-download datasets are accessible (internet connection)
3. PyTorch implementation is correct

### Skipped Tests

Some tests may be skipped due to:
- Missing manual datasets (ImageNet)
- CUDA not available (GPU tests)
- Marked as slow (unless explicitly run)

### Test Coverage Goals

- **Dataset tests**: 100% (all 8 datasets)
- **Architecture tests**: 100% (all 8 types)
- **Test problem tests**: 100% (all 25 problems)
- **Runner tests**: 80%+ (core functionality)
- **Integration tests**: Representative subset

## Troubleshooting

### Tests Fail with "Data not available"

**Solution**: Tests are skipped automatically if data is not available. This is expected for ImageNet. For other datasets, ensure internet connection is available for auto-download.

### Tests Fail with "CUDA out of memory"

**Solution**: Reduce batch size or run CPU-only tests:
```bash
CUDA_VISIBLE_DEVICES="" pytest tests/
```

### Tests are Very Slow

**Solution**: Run only fast tests or specific test files:
```bash
pytest tests/ -m "not slow"
pytest tests/test_datasets.py  # Just dataset tests
```

### Import Errors

**Solution**: Ensure DeepOBS is installed in development mode:
```bash
pip install -e .
```

## Continuous Integration

For CI/CD pipelines, recommended test commands:

```bash
# Fast tests only (for quick feedback)
uv run pytest tests/ -m "not slow" -v

# Full test suite (for comprehensive validation)
uv run pytest tests/ -v --cov=deepobs.pytorch

# Smoke test only (for rapid validation)
uv run python tests/smoke_test.py
```

## Adding New Tests

When adding new functionality:

1. **Add unit tests** to appropriate test file
2. **Add integration test** if it's a new test problem
3. **Update test_utils.py** if new helpers are needed
4. **Mark slow tests** with `@pytest.mark.slow`
5. **Skip manual tests** with `@pytest.mark.skip`

## Test Utilities

The `test_utils.py` module provides helpful functions:

- `get_dummy_batch()` - Generate random test data
- `assert_shape()` - Assert tensor shapes
- `assert_decreasing()` - Verify values decrease
- `assert_increasing()` - Verify values increase
- `count_parameters()` - Count model parameters
- `set_seed()` - Set random seeds for reproducibility
- `check_gpu_available()` - Check CUDA availability

## Performance Benchmarks

Approximate test execution times (on modern CPU):

- **Smoke test**: ~10 seconds
- **Fast tests only**: ~2-5 minutes
- **Full test suite**: ~10-20 minutes
- **Integration tests**: ~5-10 minutes

Times vary based on:
- Hardware (CPU/GPU)
- Data download (first run)
- Number of test problems available

## Questions or Issues?

If tests fail unexpectedly:

1. Check you're using compatible PyTorch version (>= 1.9.0)
2. Verify all dependencies are installed
3. Try running smoke test first: `uv run python tests/smoke_test.py`
4. Check individual test files for more details
5. Review test output for specific error messages

For more information, see the main project README.
