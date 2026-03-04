# DeepOBS PyTorch Examples

This directory contains runnable examples demonstrating DeepOBS PyTorch.

## Recommended workflow

The primary way to use DeepOBS is through the CLI and YAML config files:

```bash
# 1. Run a benchmark
uv run deepobs benchmark examples/benchmark_config_adamw_small.yaml

# 2. Analyze and generate an interactive HTML report
uv run deepobs analyze
```

The Python example scripts show the **programmatic API**, which is useful
when you need custom training logic not covered by the CLI.

## Python examples

| Script | What it shows |
|--------|---------------|
| **basic_usage.py** | Minimal training loop using the Python API |
| **custom_optimizer_benchmark.py** | Implementing a custom optimizer; benchmarking via direct loop and StandardRunner |
| **multiple_test_problems.py** | Running the same optimizer on several test problems programmatically |
| **learning_rate_schedule.py** | Using PyTorch's scheduler API with DeepOBS test problems |
| **result_analysis.py** | Loading and summarising JSON result files produced by the benchmark |

Run any example with:

```bash
uv run python examples/basic_usage.py
```

## Configuration files

YAML configs for the CLI benchmark command:

| File | Description |
|------|-------------|
| **benchmark_config_quick.yaml** | Quick smoke-test (2 problems, 5 epochs) |
| **benchmark_config_adamw_small.yaml** | AdamW on 22 problems (small–medium models) |
| **benchmark_config_angol_small.yaml** | Custom optimizer example |

```bash
# Dry-run to preview what will execute
uv run deepobs benchmark examples/benchmark_config_quick.yaml --dry-run

# Full run
uv run deepobs benchmark examples/benchmark_config_quick.yaml
```

## See also

- [README.md](../README.md) — main documentation and CLI reference
- [docs/README_PYTORCH.md](../docs/README_PYTORCH.md) — detailed PyTorch usage guide
- [docs/API_REFERENCE.md](../docs/API_REFERENCE.md) — API documentation
