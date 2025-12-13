# DeepOBS Migration Resumption Guide

**Purpose**: Quick reference to resume the TensorFlow → PyTorch migration

---

## 🎯 Where We Are

**Completed**: ✅ Analysis and Planning (100%)
**In Progress**: Implementation (0%)
**Blocked By**: Rate limits on implementation agents

---

## 🚀 How to Resume

### Option 1: Continue with New Claude Session

1. **Share context**:
   - This file: `/tmp/resumption_guide.md`
   - Status file: `/tmp/deepobs_migration_status.md`
   - Main guide: `/Users/yaroslav/Sources/Angol/DeepOBS/CLAUDE.md`

2. **Start with Phase 1**:
   ```
   "Implement Phase 1 of the DeepOBS PyTorch migration.
   Read the detailed plan at /tmp/phase1_detailed_plan.md and implement:
   - config.py
   - datasets/dataset.py
   - testproblems/testproblem.py"
   ```

3. **Proceed sequentially**:
   - Phase 1 → Phase 2 → Phase 3 → Phase 4 (MVP)
   - Then Phase 5 → Phase 6 → Phase 7 (complete)

---

### Option 2: Manual Implementation

Follow detailed plans in order:

#### Step 1: Phase 1 (4-6 hours)
```bash
cd /Users/yaroslav/Sources/Angol/DeepOBS/deepobs
mkdir -p pytorch/{datasets,testproblems,runners}
touch pytorch/__init__.py
touch pytorch/datasets/__init__.py
touch pytorch/testproblems/__init__.py
touch pytorch/runners/__init__.py
```

Then implement:
1. `pytorch/config.py` - Use plan section 2.1
2. `pytorch/datasets/dataset.py` - Use plan section 2.2
3. `pytorch/testproblems/testproblem.py` - Use plan section 2.3

**Reference**: `/tmp/phase1_detailed_plan.md`

#### Step 2: Phase 2 (8-12 hours)
Implement datasets:
1. `pytorch/datasets/mnist.py`
2. `pytorch/datasets/fmnist.py`
3. `pytorch/datasets/cifar10.py`
4. `pytorch/datasets/cifar100.py`

**Reference**: `/tmp/phase2_detailed_plan.md`

#### Step 3: Phase 3 (12-16 hours)
Implement architectures and test problems.

**Reference**: `/tmp/phase3_detailed_plan.md`

#### Step 4: Phase 4 (8-12 hours)
Implement runner.

**Reference**: `/tmp/phase4_detailed_plan.md`

---

## 📚 Essential Reading Order

1. **Start here**: `/tmp/deepobs_migration_status.md` (this shows current status)
2. **Overview**: `/tmp/deepobs_migration_plan.md` (general strategy)
3. **Details**: `/tmp/phase1_detailed_plan.md` (start implementation here)
4. **Reference**: `/Users/yaroslav/Sources/Angol/DeepOBS/CLAUDE.md` (comprehensive guide)

---

## 🔧 Key Implementation Patterns

### Config Module Pattern
```python
# deepobs/pytorch/config.py
import torch

_DATA_DIR = "data_deepobs"
_DTYPE = torch.float32

def get_data_dir():
    return _DATA_DIR

def set_data_dir(data_dir):
    global _DATA_DIR
    _DATA_DIR = data_dir
```

### DataSet Base Class Pattern
```python
# deepobs/pytorch/datasets/dataset.py
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

class DataSet(ABC):
    def __init__(self, batch_size):
        self._batch_size = batch_size
        self.train_loader = self._make_train_loader()
        self.test_loader = self._make_test_loader()

    @abstractmethod
    def _make_train_dataset(self):
        raise NotImplementedError
```

### TestProblem Base Class Pattern
```python
# deepobs/pytorch/testproblems/testproblem.py
from abc import ABC, abstractmethod
import torch.nn as nn

class TestProblem(ABC):
    def __init__(self, batch_size, weight_decay=None):
        self._batch_size = batch_size
        self._weight_decay = weight_decay
        self.model = None
        self.dataset = None

    @abstractmethod
    def set_up(self):
        raise NotImplementedError

    def get_batch_loss_and_accuracy(self, batch):
        # Implement loss and accuracy computation
        pass
```

---

## ✅ Validation Checklist

After each phase, verify:

### Phase 1
- [ ] Can import `from deepobs.pytorch import config`
- [ ] Can set/get data directory
- [ ] DataSet base class is abstract
- [ ] TestProblem base class is abstract

### Phase 2
- [ ] MNIST loads with `torchvision.datasets.MNIST`
- [ ] Batch shape is `(batch_size, 1, 28, 28)` for MNIST
- [ ] Batch shape is `(batch_size, 3, 32, 32)` for CIFAR
- [ ] Labels are class indices (not one-hot)

### Phase 3
- [ ] All architecture modules forward pass works
- [ ] Parameter counts match TensorFlow
- [ ] Test problems can be instantiated
- [ ] Loss computation returns per-example losses

### Phase 4
- [ ] Runner can parse command-line arguments
- [ ] Training loop completes 1 epoch without errors
- [ ] JSON output matches expected format
- [ ] Learning rate scheduling works

---

## 🎓 Critical Knowledge

### Must Remember

1. **Batch Norm Momentum**: PyTorch uses `1 - tf_momentum`
   - TF 0.9 → PyTorch 0.1
   - Inception V3: TF 0.9997 → PyTorch 0.0003

2. **Weight Decay**: Use optimizer parameter, not manual L2
   ```python
   optimizer = torch.optim.SGD(params, lr=0.1, weight_decay=5e-4)
   ```

3. **Per-Example Losses**: Always use `reduction='none'`
   ```python
   losses = F.cross_entropy(logits, labels, reduction='none')
   ```

4. **Channel Order**: PyTorch uses NCHW (channels first)

5. **No Session**: PyTorch is eager, no session.run() needed

---

## 🐛 Common Issues & Solutions

### Issue: Import errors
**Solution**: Make sure all `__init__.py` files exist

### Issue: Batch norm not working
**Solution**: Call `model.train()` before training, `model.eval()` before eval

### Issue: Loss shape wrong
**Solution**: Check `reduction='none'` for per-example losses

### Issue: Weight initialization different
**Solution**: Use exact same initializer as TensorFlow
- Truncated normal: `torch.nn.init.trunc_normal_()`
- Xavier: `torch.nn.init.xavier_normal_()`

### Issue: LSTM state memory leak
**Solution**: Detach state after each batch
```python
hidden = tuple(h.detach() for h in hidden)
```

---

## 📞 Help & Resources

### Documentation
- PyTorch Docs: https://pytorch.org/docs/stable/
- TorchVision: https://pytorch.org/vision/stable/
- Migration Guide: https://pytorch.org/tutorials/beginner/former_torchies/

### Files
- Plans: `/tmp/phase*.md`
- Status: `/tmp/deepobs_migration_status.md`
- Guide: `/Users/yaroslav/Sources/Angol/DeepOBS/CLAUDE.md`

### Original
- TensorFlow Code: `/Users/yaroslav/Sources/Angol/DeepOBS/deepobs/tensorflow/`
- Paper: https://openreview.net/forum?id=rJg6ssC5Y7

---

## ⏱️ Time Estimates

| Phase | Estimated Time | Can Parallelize? |
|-------|----------------|------------------|
| Phase 1 | 4-6 hours | No (foundation) |
| Phase 2 | 8-12 hours | Partially |
| Phase 3 | 12-16 hours | Yes (per test problem) |
| Phase 4 | 8-12 hours | No |
| Phase 5 | 10-14 hours | Yes (per dataset) |
| Phase 6 | 20-30 hours | Yes (per architecture) |
| Phase 7 | 12-16 hours | Partially |

**MVP** (Phases 1-4): ~32-46 hours
**Complete**: ~74-106 hours

---

## 🎯 Success Criteria

### Minimum Viable Product (Phases 1-4)
- ✅ Can run MNIST/CIFAR-10 test problems
- ✅ Runner produces valid JSON output
- ✅ Analyzer can plot results
- ✅ Results match TensorFlow within 1%

### Complete Migration (Phases 1-7)
- ✅ All 26 test problems implemented
- ✅ All datasets working
- ✅ All architectures validated
- ✅ Documentation updated
- ✅ Tests passing

---

**Last Updated**: 2025-12-13
**Status**: Ready to resume implementation
