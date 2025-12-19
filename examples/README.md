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

## Configuration Files

The following YAML configuration files are provided for CLI benchmarking:

- **benchmark_config.yaml** - Full benchmark configuration example
- **benchmark_config_quick.yaml** - Quick test configuration
- **benchmark_config_adamw_small.yaml** - AdamW optimizer example
- **benchmark_config_angol_small.yaml** - Custom optimizer example

Use with the DeepOBS CLI:
```bash
uv run deepobs benchmark benchmark_config_quick.yaml
```

See [QUICK_START_BENCHMARK.md](../docs/QUICK_START_BENCHMARK.md) for more details on CLI usage.

## See Also

- [README_PYTORCH.md](../docs/README_PYTORCH.md) - Main usage guide
- [QUICK_START_BENCHMARK.md](../docs/QUICK_START_BENCHMARK.md) - CLI benchmarking guide
- [API_REFERENCE.md](../docs/API_REFERENCE.md) - Complete API documentation
