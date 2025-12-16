# DeepOBS PyTorch Examples

This directory contains complete, runnable examples demonstrating various features of DeepOBS PyTorch.

## Examples Overview

1. **basic_usage.py** - Simple end-to-end example
   - Basic training loop
   - Loss and accuracy computation
   - Model evaluation

2. **custom_optimizer_benchmark.py** - Benchmarking custom optimizers
   - Implementing a custom optimizer
   - Using StandardRunner
   - Saving and analyzing results

3. **multiple_test_problems.py** - Running multiple test problems
   - Batch benchmarking
   - Comparing across problems
   - Aggregating results

4. **learning_rate_schedule.py** - Learning rate scheduling
   - MultiStepLR scheduling
   - Cosine annealing
   - Custom schedules

5. **result_analysis.py** - Analyzing and plotting results
   - Loading results from JSON
   - Creating plots
   - Statistical analysis

## Running Examples

Each example is self-contained and can be run directly:

```bash
cd examples
python basic_usage.py
```

## Requirements

All examples require:
- PyTorch >= 1.9.0
- DeepOBS PyTorch version
- matplotlib (for visualization examples)
- numpy

Install with:
```bash
pip install torch torchvision matplotlib numpy pandas seaborn
```

## Example Output

Examples save results to:
```
./results/
├── <testproblem>/
│   └── <optimizer>/
│       └── <run_id>/
│           └── results.json
└── plots/
    └── *.png
```

## Tips

- Start with `basic_usage.py` to understand the fundamentals
- Modify hyperparameters in examples to experiment
- Use GPU if available for faster training (set `device='cuda'`)
- Check comments in each script for detailed explanations

## See Also

- [README_PYTORCH.md](../README_PYTORCH.md) - Main usage guide
- [API_REFERENCE.md](../API_REFERENCE.md) - Complete API documentation
- [MIGRATION_GUIDE.md](../MIGRATION_GUIDE.md) - Migrating from TensorFlow
