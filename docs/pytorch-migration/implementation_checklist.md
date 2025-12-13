# DeepOBS PyTorch Implementation Checklist

**Purpose**: Complete list of all files to create during migration

---

## Phase 1: Core Infrastructure (6 files)

### Directory Structure
```
deepobs/pytorch/
├── __init__.py
├── config.py
├── datasets/
│   ├── __init__.py
│   └── dataset.py
├── testproblems/
│   ├── __init__.py
│   └── testproblem.py
└── runners/
    └── __init__.py
```

### Files to Create

- [ ] `deepobs/pytorch/__init__.py`
  - **Size**: ~30 lines
  - **Purpose**: Package initialization, expose config functions
  - **Reference**: `/tmp/phase1_detailed_plan.md` section 4

- [ ] `deepobs/pytorch/config.py`
  - **Size**: ~80 lines
  - **Purpose**: Global configuration (data dir, dtype)
  - **Reference**: `/tmp/phase1_detailed_plan.md` section 2.1
  - **TF Original**: `deepobs/tensorflow/config.py`

- [ ] `deepobs/pytorch/datasets/__init__.py`
  - **Size**: ~5 lines
  - **Purpose**: Export DataSet base class
  - **Reference**: `/tmp/phase1_detailed_plan.md` section 5

- [ ] `deepobs/pytorch/datasets/dataset.py`
  - **Size**: ~120 lines
  - **Purpose**: Base class for all datasets
  - **Reference**: `/tmp/phase1_detailed_plan.md` section 2.2
  - **TF Original**: `deepobs/tensorflow/datasets/dataset.py`

- [ ] `deepobs/pytorch/testproblems/__init__.py`
  - **Size**: ~5 lines
  - **Purpose**: Export TestProblem base class
  - **Reference**: `/tmp/phase1_detailed_plan.md` section 6

- [ ] `deepobs/pytorch/testproblems/testproblem.py`
  - **Size**: ~150 lines
  - **Purpose**: Base class for all test problems
  - **Reference**: `/tmp/phase1_detailed_plan.md` section 2.3
  - **TF Original**: `deepobs/tensorflow/testproblems/testproblem.py`

- [ ] `deepobs/pytorch/runners/__init__.py`
  - **Size**: ~5 lines
  - **Purpose**: Package initialization (populated in Phase 4)

---

## Phase 2: Simple Datasets (4 files)

- [x] `deepobs/pytorch/datasets/mnist.py` ✅
  - **Size**: ~100 lines
  - **Purpose**: MNIST dataset using torchvision
  - **Reference**: `/tmp/phase2_detailed_plan.md` section 3.1
  - **TF Original**: `deepobs/tensorflow/datasets/mnist.py`

- [x] `deepobs/pytorch/datasets/fmnist.py` ✅
  - **Size**: ~100 lines
  - **Purpose**: Fashion-MNIST dataset
  - **Reference**: `/tmp/phase2_detailed_plan.md` section 3.2
  - **TF Original**: `deepobs/tensorflow/datasets/fmnist.py`

- [x] `deepobs/pytorch/datasets/cifar10.py` ✅
  - **Size**: ~160 lines
  - **Purpose**: CIFAR-10 with data augmentation
  - **Reference**: `/tmp/phase2_detailed_plan.md` section 3.3
  - **TF Original**: `deepobs/tensorflow/datasets/cifar10.py`

- [x] `deepobs/pytorch/datasets/cifar100.py` ✅
  - **Size**: ~160 lines
  - **Purpose**: CIFAR-100 with data augmentation
  - **Reference**: `/tmp/phase2_detailed_plan.md` section 3.4
  - **TF Original**: `deepobs/tensorflow/datasets/cifar100.py`

---

## Phase 3: Simple Architectures (14 files)

### Architecture Modules (4 files)

- [x] `deepobs/pytorch/testproblems/_logreg.py` ✅
  - **Size**: ~40 lines
  - **Purpose**: Logistic regression model
  - **Reference**: `/tmp/phase3_detailed_plan.md` section 1
  - **TF Original**: `deepobs/tensorflow/testproblems/_logreg.py`

- [x] `deepobs/pytorch/testproblems/_mlp.py` ✅
  - **Size**: ~60 lines
  - **Purpose**: 4-layer MLP (784→1000→500→100→out)
  - **Reference**: `/tmp/phase3_detailed_plan.md` section 2
  - **TF Original**: `deepobs/tensorflow/testproblems/_mlp.py`

- [x] `deepobs/pytorch/testproblems/_2c2d.py` ✅
  - **Size**: ~80 lines
  - **Purpose**: 2 conv + 2 dense layers
  - **Reference**: `/tmp/phase3_detailed_plan.md` section 3
  - **TF Original**: `deepobs/tensorflow/testproblems/_2c2d.py`

- [x] `deepobs/pytorch/testproblems/_3c3d.py` ✅
  - **Size**: ~100 lines
  - **Purpose**: 3 conv + 3 dense layers
  - **Reference**: `/tmp/phase3_detailed_plan.md` section 4
  - **TF Original**: `deepobs/tensorflow/testproblems/_3c3d.py`

### Test Problems (10 files)

**MNIST (3 files)**:
- [x] `deepobs/pytorch/testproblems/mnist_logreg.py` ✅
- [x] `deepobs/pytorch/testproblems/mnist_mlp.py` ✅
- [x] `deepobs/pytorch/testproblems/mnist_2c2d.py` ✅

**Fashion-MNIST (3 files)**:
- [x] `deepobs/pytorch/testproblems/fmnist_logreg.py` ✅
- [x] `deepobs/pytorch/testproblems/fmnist_mlp.py` ✅
- [x] `deepobs/pytorch/testproblems/fmnist_2c2d.py` ✅

**CIFAR-10 (1 file)**:
- [x] `deepobs/pytorch/testproblems/cifar10_3c3d.py` ✅

**CIFAR-100 (1 file)**:
- [x] `deepobs/pytorch/testproblems/cifar100_3c3d.py` ✅

**SVHN (2 files)**:
- [ ] `deepobs/pytorch/testproblems/svhn_3c3d.py` (SVHN dataset not yet implemented)

Each test problem file is ~50-80 lines.

---

## Phase 4: Basic Runner (2 files)

- [x] `deepobs/pytorch/runners/runner_utils.py` ✅
  - **Size**: ~150 lines (completed)
  - **Purpose**: Utility functions (mostly copied from TF)
  - **Functions**: `float2str()`, `make_run_name()`, `make_lr_schedule()`
  - **Reference**: `/tmp/phase4_detailed_plan.md` section 2.1
  - **TF Original**: `deepobs/tensorflow/runners/runner_utils.py`

- [x] `deepobs/pytorch/runners/standard_runner.py` ✅
  - **Size**: ~480 lines (completed)
  - **Purpose**: Main training orchestration
  - **Key Methods**: `run()`, `_run()`
  - **Reference**: `/tmp/phase4_detailed_plan.md` section 2.2-2.3
  - **TF Original**: `deepobs/tensorflow/runners/standard_runner.py`

---

## Phase 5: Remaining Datasets (5 files)

- [ ] `deepobs/pytorch/datasets/svhn.py`
  - **Size**: ~150 lines
  - **Purpose**: SVHN dataset (binary format or torchvision)
  - **Reference**: `/tmp/phase5_detailed_plan.md` section 1
  - **TF Original**: `deepobs/tensorflow/datasets/svhn.py`

- [ ] `deepobs/pytorch/datasets/imagenet.py`
  - **Size**: ~100 lines
  - **Purpose**: ImageNet (requires manual setup)
  - **Reference**: `/tmp/phase5_detailed_plan.md` section 2
  - **TF Original**: `deepobs/tensorflow/datasets/imagenet.py`

- [ ] `deepobs/pytorch/datasets/tolstoi.py`
  - **Size**: ~120 lines
  - **Purpose**: Character-level text dataset
  - **Reference**: `/tmp/phase5_detailed_plan.md` section 3
  - **TF Original**: `deepobs/tensorflow/datasets/tolstoi.py`

- [ ] `deepobs/pytorch/datasets/quadratic.py`
  - **Size**: ~80 lines
  - **Purpose**: Synthetic quadratic dataset
  - **Reference**: `/tmp/phase5_detailed_plan.md` section 4
  - **TF Original**: `deepobs/tensorflow/datasets/quadratic.py`

- [ ] `deepobs/pytorch/datasets/two_d.py`
  - **Size**: ~60 lines
  - **Purpose**: 2D optimization test data
  - **Reference**: `/tmp/phase5_detailed_plan.md` section 5
  - **TF Original**: `deepobs/tensorflow/datasets/two_d.py`

---

## Phase 6: Advanced Architectures (20 files)

### Architecture Modules (5 files)

- [ ] `deepobs/pytorch/testproblems/_vgg.py`
  - **Size**: ~200 lines
  - **Purpose**: VGG16/VGG19 architecture
  - **Reference**: `/tmp/phase6_detailed_plan.md` section 2
  - **TF Original**: `deepobs/tensorflow/testproblems/_vgg.py`

- [ ] `deepobs/pytorch/testproblems/_wrn.py`
  - **Size**: ~250 lines
  - **Purpose**: Wide ResNet with batch norm
  - **Reference**: `/tmp/phase6_detailed_plan.md` section 3
  - **TF Original**: `deepobs/tensorflow/testproblems/_wrn.py`

- [ ] `deepobs/pytorch/testproblems/_inception_v3.py`
  - **Size**: ~400+ lines
  - **Purpose**: Inception V3 (complex multi-branch)
  - **Reference**: `/tmp/phase6_detailed_plan.md` section 4
  - **TF Original**: `deepobs/tensorflow/testproblems/_inception_v3.py`

- [ ] `deepobs/pytorch/testproblems/_vae.py`
  - **Size**: ~200 lines
  - **Purpose**: Variational Autoencoder
  - **Reference**: `/tmp/phase6_detailed_plan.md` section 5
  - **TF Original**: `deepobs/tensorflow/testproblems/_vae.py`

- [ ] `deepobs/pytorch/testproblems/_allcnnc.py`
  - **Size**: ~100 lines
  - **Purpose**: All-CNN-C (no pooling)
  - **Reference**: `/tmp/phase6_detailed_plan.md` section 6
  - **TF Original**: `deepobs/tensorflow/testproblems/cifar100_allcnnc.py`

### Test Problems (15 files)

**VGG (6 files)**:
- [ ] `deepobs/pytorch/testproblems/cifar10_vgg16.py`
- [ ] `deepobs/pytorch/testproblems/cifar10_vgg19.py`
- [ ] `deepobs/pytorch/testproblems/cifar100_vgg16.py`
- [ ] `deepobs/pytorch/testproblems/cifar100_vgg19.py`
- [ ] `deepobs/pytorch/testproblems/imagenet_vgg16.py`
- [ ] `deepobs/pytorch/testproblems/imagenet_vgg19.py`

**Wide ResNet (2 files)**:
- [ ] `deepobs/pytorch/testproblems/cifar100_wrn404.py`
- [ ] `deepobs/pytorch/testproblems/svhn_wrn164.py`

**Inception V3 (1 file)**:
- [ ] `deepobs/pytorch/testproblems/imagenet_inception_v3.py`

**VAE (2 files)**:
- [ ] `deepobs/pytorch/testproblems/mnist_vae.py`
- [ ] `deepobs/pytorch/testproblems/fmnist_vae.py`

**All-CNN-C (1 file)**:
- [ ] `deepobs/pytorch/testproblems/cifar100_allcnnc.py`

3 files (architecture modules) + 12 test problem files = 15 files

---

## Phase 7: RNN and Specialized (6 files)

### Architecture Modules (2 files)

- [ ] `deepobs/pytorch/testproblems/tolstoi_char_rnn.py`
  - **Size**: ~200 lines
  - **Purpose**: Character LSTM with state management
  - **Reference**: `/tmp/phase7_detailed_plan.md` section 2
  - **TF Original**: `deepobs/tensorflow/testproblems/tolstoi_char_rnn.py`

- [ ] `deepobs/pytorch/testproblems/_quadratic.py`
  - **Size**: ~100 lines
  - **Purpose**: Base quadratic model (no neural network)
  - **Reference**: `/tmp/phase7_detailed_plan.md` section 3.3
  - **TF Original**: `deepobs/tensorflow/testproblems/_quadratic.py`

### Test Problems (4 files)

**Quadratic (1 file)**:
- [ ] `deepobs/pytorch/testproblems/quadratic_deep.py`
  - **Size**: ~150 lines
  - **Purpose**: 100D quadratic with deep learning eigenspectrum
  - **Reference**: `/tmp/phase7_detailed_plan.md` section 3.4
  - **TF Original**: `deepobs/tensorflow/testproblems/quadratic_deep.py`

**2D Functions (3 files)**:
- [ ] `deepobs/pytorch/testproblems/two_d_rosenbrock.py`
  - **Size**: ~80 lines
  - **Purpose**: Rosenbrock optimization function
  - **Reference**: `/tmp/phase7_detailed_plan.md` section 4.4
  - **TF Original**: `deepobs/tensorflow/testproblems/two_d_rosenbrock.py`

- [ ] `deepobs/pytorch/testproblems/two_d_beale.py`
  - **Size**: ~80 lines
  - **Purpose**: Beale optimization function
  - **Reference**: `/tmp/phase7_detailed_plan.md` section 4.5
  - **TF Original**: `deepobs/tensorflow/testproblems/two_d_beale.py`

- [ ] `deepobs/pytorch/testproblems/two_d_branin.py`
  - **Size**: ~90 lines
  - **Purpose**: Branin optimization function
  - **Reference**: `/tmp/phase7_detailed_plan.md` section 4.6
  - **TF Original**: `deepobs/tensorflow/testproblems/two_d_branin.py`

---

## Summary Statistics

| Phase | Architecture Modules | Test Problems | Dataset Modules | Other | Total Files |
|-------|---------------------|---------------|-----------------|-------|-------------|
| Phase 1 | 0 | 0 | 1 | 6 | **7** |
| Phase 2 | 0 | 0 | 4 | 0 | **4** |
| Phase 3 | 4 | 10 | 0 | 0 | **14** |
| Phase 4 | 0 | 0 | 0 | 2 | **2** |
| Phase 5 | 0 | 0 | 5 | 0 | **5** |
| Phase 6 | 5 | 15 | 0 | 0 | **20** |
| Phase 7 | 2 | 4 | 0 | 0 | **6** |
| **TOTAL** | **11** | **29** | **10** | **8** | **58** |

---

## Quick Reference

### By Component Type

**Base Classes (3)**:
- `config.py`
- `datasets/dataset.py`
- `testproblems/testproblem.py`

**Datasets (10)**:
- MNIST, Fashion-MNIST, CIFAR-10, CIFAR-100
- SVHN, ImageNet, Tolstoi
- Quadratic, Two-D

**Architecture Modules (11)**:
- Simple: logreg, mlp, 2c2d, 3c3d
- Advanced: vgg, wrn, inception_v3, vae, allcnnc
- Specialized: char_rnn, quadratic_base

**Test Problems (29)**:
- MNIST: 4 (logreg, mlp, 2c2d, vae)
- Fashion-MNIST: 4 (logreg, mlp, 2c2d, vae)
- CIFAR-10: 3 (3c3d, vgg16, vgg19)
- CIFAR-100: 5 (3c3d, allcnnc, vgg16, vgg19, wrn404)
- SVHN: 2 (3c3d, wrn164)
- ImageNet: 3 (vgg16, vgg19, inception_v3)
- Tolstoi: 1 (char_rnn)
- Synthetic: 4 (quadratic_deep, 3× 2D functions)

**Runner (2)**:
- `runner_utils.py`
- `standard_runner.py`

**Package Init (5)**:
- `pytorch/__init__.py`
- `pytorch/datasets/__init__.py`
- `pytorch/testproblems/__init__.py`
- `pytorch/runners/__init__.py`

---

## File Size Estimates

| Category | Total Lines | Estimated Total |
|----------|-------------|-----------------|
| Phase 1 | ~500 | 500 |
| Phase 2 | ~500 | 500 |
| Phase 3 | ~1,200 | 1,200 |
| Phase 4 | ~500 | 500 |
| Phase 5 | ~500 | 500 |
| Phase 6 | ~2,500 | 2,500 |
| Phase 7 | ~800 | 800 |
| **TOTAL** | **~6,500 lines** | **6,500** |

---

## Validation Steps

After creating each file:

1. **Syntax Check**:
   ```bash
   python -m py_compile deepobs/pytorch/path/to/file.py
   ```

2. **Import Test**:
   ```python
   from deepobs.pytorch.path.to.module import ClassName
   ```

3. **Basic Functionality**:
   - Run example from detailed plan
   - Verify output shapes
   - Check parameter counts

4. **Update Checklist**: Mark file as complete ✅

---

**Last Updated**: 2025-12-13
**Total Files**: 58
**Estimated Lines**: ~6,500
