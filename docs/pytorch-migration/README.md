# DeepOBS PyTorch Migration

This folder contains documentation and planning for converting DeepOBS from TensorFlow to PyTorch.

## Current Status

**Planning**: ✅ Complete
**Implementation**: 🚧 In Progress (starting Phase 1)

## Available Documentation

- **`deepobs_migration_plan.md`** - Overall migration strategy and architecture analysis
- **`deepobs_migration_status.md`** - Current status and progress tracking
- **`implementation_checklist.md`** - Complete list of 58 files to create
- **`resumption_guide.md`** - Quick-start guide for resuming work

## Phase Plans

Detailed implementation plans for each phase are provided in the main guide (`CLAUDE.md` in project root).

## Quick Start

### Phase 1: Core Infrastructure (START HERE)

Create the base classes that everything else depends on:

```bash
cd /Users/yaroslav/Sources/Angol/DeepOBS/deepobs
mkdir -p pytorch/{datasets,testproblems,runners}
```

Implement these files in order:
1. `pytorch/config.py` - Configuration (data dirs, dtype)
2. `pytorch/datasets/dataset.py` - Base DataSet class
3. `pytorch/testproblems/testproblem.py` - Base TestProblem class
4. Package `__init__.py` files

See `CLAUDE.md` sections on Phase 1 for detailed specifications.

## Migration Principles

1. **No backward compatibility** - Clean PyTorch implementation
2. **Modern patterns** - Use DataLoader, nn.Module, no sessions
3. **Simple code** - Don't over-engineer, match functionality only
4. **Test each phase** - Verify before moving to next phase

## File Structure

Target PyTorch structure:
```
deepobs/pytorch/
├── __init__.py
├── config.py
├── datasets/
│   ├── __init__.py
│   ├── dataset.py (base class)
│   ├── mnist.py
│   ├── cifar10.py
│   └── ... (10 datasets total)
├── testproblems/
│   ├── __init__.py
│   ├── testproblem.py (base class)
│   ├── _mlp.py (architecture)
│   ├── mnist_mlp.py (test problem)
│   └── ... (29 test problems + 11 architectures)
└── runners/
    ├── __init__.py
    ├── runner_utils.py
    └── standard_runner.py
```

## References

- Original TensorFlow: `deepobs/tensorflow/`
- Main conversion guide: `CLAUDE.md` (project root)
- PyTorch docs: https://pytorch.org/docs/stable/

## Progress Tracking

Track progress by checking off items in `implementation_checklist.md`.

---

**Last Updated**: 2025-12-13
