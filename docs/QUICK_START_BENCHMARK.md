# Quick Start: DeepOBS Benchmark Suite

## Overview

DeepOBS provides a streamlined command-line interface for running comprehensive optimizer benchmarks.

## Quick Test (2 minutes)

Run a minimal benchmark to verify everything works:

```bash
# Run benchmark with quick configuration
uv run deepobs benchmark benchmark_config_quick.yaml

# Or just use the default configuration
uv run deepobs benchmark
```

This runs:
- 2 test problems (mnist_logreg, mnist_mlp)
- 2 optimizers (SGD, Adam)
- 3-5 epochs each
- **Total: 4 benchmark runs**

## View Results

After the benchmark completes:

```bash
# Analyze and generate interactive HTML report
uv run deepobs analyze

# Or analyze results from a custom directory
uv run deepobs analyze ./results
```

This generates:
- Interactive learning curves
- Comparison charts
- Performance profiles
- Statistical summaries

Results saved to: `./results/benchmark_report.html`

## Full Benchmark

Edit `benchmark_config.yaml` and run:

```bash
uv run deepobs benchmark benchmark_config.yaml
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

### 4. Custom Optimizers
```yaml
optimizers:
  - name: MyCustomOptimizer
    optimizer_class: my_package.optimizers.MyCustomOptimizer
    learning_rate: 0.01
    custom_param: 0.9
```

### 5. Dry Run
```bash
uv run deepobs benchmark my_config.yaml --dry-run
```

## Example Workflow

```bash
# 1. Create/edit configuration
nano my_benchmark.yaml

# 2. Test with dry run
uv run deepobs benchmark my_benchmark.yaml --dry-run

# 3. Run actual benchmark
uv run deepobs benchmark my_benchmark.yaml

# 4. Analyze results
uv run deepobs analyze

# 5. View interactive report
open results/benchmark_report.html
```

## CLI Commands

### Benchmark Command

```bash
# Run with default config (benchmark_config.yaml)
uv run deepobs benchmark

# Run with custom config
uv run deepobs benchmark my_config.yaml

# Dry run to preview what will be executed
uv run deepobs benchmark my_config.yaml --dry-run

# Get help
uv run deepobs benchmark --help
```

### Analyze Command

```bash
# Analyze default results directory (./results)
uv run deepobs analyze

# Analyze custom results directory
uv run deepobs analyze ./my_results

# Get help
uv run deepobs analyze --help
```

## Next Steps

1. **Read** `BENCHMARK_SUITE_README.md` for full documentation
2. **Edit** `benchmark_config.yaml` to select your problems
3. **Run** your benchmark suite with `uv run deepobs benchmark`
4. **Analyze** results with `uv run deepobs analyze`

## Need Help?

- Full docs: `BENCHMARK_SUITE_README.md`
- DeepOBS API: `docs/API_REFERENCE.md`
- Example configs: `benchmark_config*.yaml`
