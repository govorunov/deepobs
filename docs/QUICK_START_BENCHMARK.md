# Quick Start: Configuration-Driven Benchmark Suite

## What's New

You now have a complete configuration-driven benchmark system for DeepOBS!

## Files Created

1. **`run_benchmark.py`** - Main benchmark runner script
2. **`benchmark_config.yaml`** - Full configuration example
3. **`benchmark_config_quick.yaml`** - Quick test configuration
4. **`BENCHMARK_SUITE_README.md`** - Complete documentation

## Quick Test (2 minutes)

Run a minimal benchmark to verify everything works:

```bash
uv run python run_benchmark.py benchmark_config_quick.yaml
```

This runs:
- 2 test problems (mnist_logreg, mnist_mlp)
- 2 optimizers (SGD, Adam)
- 3-5 epochs each
- **Total: 4 benchmark runs**

## View Results

After the benchmark completes:

```bash
uv run python examples/result_analysis.py
```

This generates:
- Learning curves
- Comparison charts
- Performance profiles
- Statistical summaries

Plots saved to: `./results/plots/`

## Full Benchmark

Edit `benchmark_config.yaml` and run:

```bash
uv run python run_benchmark.py benchmark_config.yaml
```

## Key Features

### 1. YAML Configuration
```yaml
test_problems:
  - name: mnist_mlp
    num_epochs: 10

optimizers:
  - name: SGD
    learning_rate: 0.01
    hyperparams:
      momentum: 0.9
```

### 2. Problem-Specific Overrides
```yaml
overrides:
  mnist_logreg:
    SGD:
      learning_rate: 0.1  # Higher LR for simple problems
```

### 3. Learning Rate Schedules
```yaml
optimizers:
  - name: SGD
    learning_rate: 0.1
    lr_schedule:
      epochs: [30, 60, 90]
      factors: [0.1, 0.1, 0.1]
```

### 4. Dry Run
```bash
uv run python run_benchmark.py my_config.yaml --dry-run
```

## Example Workflow

```bash
# 1. Create/edit configuration
nano my_benchmark.yaml

# 2. Test with dry run
uv run python run_benchmark.py my_benchmark.yaml --dry-run

# 3. Run actual benchmark
uv run python run_benchmark.py my_benchmark.yaml

# 4. Analyze results
uv run python examples/result_analysis.py

# 5. View plots
open results/plots/
```

## Compatible with result_analysis.py

Results are automatically saved in the format expected by the existing `result_analysis.py` script. No changes needed!

## Next Steps

1. **Read** `BENCHMARK_SUITE_README.md` for full documentation
2. **Edit** `benchmark_config.yaml` to select your problems
3. **Run** your benchmark suite
4. **Analyze** with `result_analysis.py`

## Need Help?

- Full docs: `BENCHMARK_SUITE_README.md`
- DeepOBS API: `docs/API_REFERENCE.md`
- Example configs: `benchmark_config*.yaml`
