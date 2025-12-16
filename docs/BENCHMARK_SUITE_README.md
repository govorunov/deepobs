# DeepOBS Configuration-Driven Benchmark Suite

A flexible, configuration-based system for running comprehensive optimizer benchmarks across multiple DeepOBS test problems.

## Features

- **YAML Configuration**: Define benchmarks using simple YAML files
- **Multiple Optimizers**: Compare SGD, Adam, AdamW, RMSprop, and more
- **Multiple Problems**: Run across all 26 DeepOBS test problems
- **Problem-Specific Overrides**: Fine-tune learning rates and hyperparameters per problem
- **Learning Rate Schedules**: Support for step-based LR decay
- **Compatible Output**: Results work seamlessly with `result_analysis.py`
- **Dry Run Mode**: Preview what will be executed without running

## Quick Start

### 1. Install Dependencies

First, ensure you have PyYAML installed:

```bash
# Using UV (recommended)
uv pip install pyyaml

# Or using pip
pip install pyyaml
```

### 2. Run a Quick Benchmark

Use the provided quick configuration for testing:

```bash
# Run quick benchmark (2 problems × 2 optimizers = 4 runs)
uv run python run_benchmark.py benchmark_config_quick.yaml
```

This will complete in a few minutes and create results in `./results/`.

### 3. Analyze Results

Generate plots and statistics:

```bash
uv run python examples/result_analysis.py
```

## Configuration File Format

### Basic Structure

```yaml
# Global settings (apply to all benchmarks)
global:
  output_dir: "./results"
  random_seed: 42
  num_epochs: 10
  batch_size: 128
  train_log_interval: 100
  print_train_iter: false
  data_dir: null  # null = use default

# Test problems to run
test_problems:
  - name: mnist_mlp
    num_epochs: 10    # Override global setting
    batch_size: 128

  - name: cifar10_3c3d
    num_epochs: 20

# Optimizers to benchmark
optimizers:
  - name: SGD
    learning_rate: 0.01
    hyperparams:
      momentum: 0.9
      nesterov: false

  - name: Adam
    learning_rate: 0.001
    hyperparams:
      betas: [0.9, 0.999]
      eps: 1.0e-08

# Problem-specific overrides
overrides:
  mnist_logreg:
    SGD:
      learning_rate: 0.1  # Use higher LR for logreg

# Advanced options
advanced:
  parallel: false
  continue_existing: false
  verbose: true
```

### Global Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `output_dir` | string | `"./results"` | Directory for saving results |
| `random_seed` | int | `42` | Random seed for reproducibility |
| `num_epochs` | int | `10` | Default number of training epochs |
| `batch_size` | int | `128` | Default batch size |
| `train_log_interval` | int | `100` | Log every N batches |
| `print_train_iter` | bool | `false` | Print training iterations |
| `data_dir` | string | `null` | Data directory (null = default) |

### Test Problems

Each test problem can specify:

```yaml
test_problems:
  - name: mnist_mlp          # Required: problem name
    num_epochs: 10           # Optional: override global
    batch_size: 128          # Optional: override global
```

Available test problems:
- **MNIST**: `mnist_logreg`, `mnist_mlp`, `mnist_2c2d`, `mnist_vae`
- **Fashion-MNIST**: `fmnist_logreg`, `fmnist_mlp`, `fmnist_2c2d`, `fmnist_vae`
- **CIFAR-10**: `cifar10_3c3d`, `cifar10_vgg16`, `cifar10_vgg19`
- **CIFAR-100**: `cifar100_3c3d`, `cifar100_allcnnc`, `cifar100_vgg16`, `cifar100_vgg19`, `cifar100_wrn404`
- **SVHN**: `svhn_3c3d`, `svhn_wrn164`
- **ImageNet**: `imagenet_vgg16`, `imagenet_vgg19`, `imagenet_inception_v3`
- **Text**: `tolstoi_char_rnn`
- **Synthetic**: `quadratic_deep`, `two_d_rosenbrock`, `two_d_beale`, `two_d_branin`

### Optimizers

Each optimizer configuration:

```yaml
optimizers:
  - name: SGD                # Required: display name
    optimizer_class: SGD     # Optional: if different from name
    learning_rate: 0.01      # Required: base learning rate
    hyperparams:             # Optimizer-specific parameters
      momentum: 0.9
      nesterov: false
    lr_schedule:             # Optional: LR decay schedule
      epochs: [30, 60, 90]
      factors: [0.1, 0.1, 0.1]
```

Supported optimizers:
- `SGD` - Stochastic Gradient Descent
- `Adam` - Adam optimizer
- `AdamW` - Adam with weight decay
- `RMSprop` - RMSprop
- `Adagrad` - Adagrad
- `Adadelta` - Adadelta
- `Adamax` - Adamax
- `ASGD` - Averaged SGD

### Learning Rate Schedules

Define step-based learning rate decay:

```yaml
optimizers:
  - name: SGD
    learning_rate: 0.1
    hyperparams:
      momentum: 0.9
    lr_schedule:
      epochs: [50, 100]      # Decay at epochs 50 and 100
      factors: [0.1, 0.01]   # Multiply by 0.1, then 0.01
```

This creates the schedule:
- Epochs 0-49: LR = 0.1
- Epochs 50-99: LR = 0.1 × 0.1 = 0.01
- Epochs 100+: LR = 0.1 × 0.01 = 0.001

### Problem-Specific Overrides

Fine-tune settings for specific problem/optimizer combinations:

```yaml
overrides:
  mnist_logreg:              # Problem name
    SGD:                     # Optimizer name
      learning_rate: 0.1     # Override learning rate
      hyperparams:           # Override hyperparameters
        momentum: 0.0

  cifar10_3c3d:
    Adam:
      learning_rate: 0.0001
      lr_schedule:
        epochs: [40, 80]
        factors: [0.1, 0.1]
```

## Usage Examples

### Run Full Benchmark

```bash
# Run complete benchmark suite
uv run python run_benchmark.py benchmark_config.yaml
```

### Dry Run

Preview what will be executed without running:

```bash
uv run python run_benchmark.py benchmark_config.yaml --dry-run
```

### Custom Configuration

```bash
# Use your own config file
uv run python run_benchmark.py my_custom_config.yaml
```

### Analyze Results

After benchmarks complete:

```bash
# Generate plots and statistics
uv run python examples/result_analysis.py
```

This creates:
- Learning curves for each test problem
- Comparison bar charts
- Performance profiles
- Convergence analysis
- Statistical summaries

Results are saved to `./results/plots/`.

## Example Configurations

### Minimal (Fast Testing)

```yaml
# benchmark_config_quick.yaml
global:
  output_dir: "./results"
  random_seed: 42
  num_epochs: 5

test_problems:
  - name: mnist_logreg
  - name: mnist_mlp

optimizers:
  - name: SGD
    learning_rate: 0.01
    hyperparams:
      momentum: 0.9

  - name: Adam
    learning_rate: 0.001
    hyperparams:
      betas: [0.9, 0.999]
      eps: 1.0e-08
```

### Comprehensive (Full Benchmark)

```yaml
global:
  output_dir: "./results"
  random_seed: 42
  num_epochs: 10
  batch_size: 128

test_problems:
  # MNIST suite
  - name: mnist_logreg
  - name: mnist_mlp
  - name: mnist_2c2d
  - name: mnist_vae

  # Fashion-MNIST suite
  - name: fmnist_logreg
  - name: fmnist_mlp
  - name: fmnist_2c2d
  - name: fmnist_vae

  # CIFAR-10 suite
  - name: cifar10_3c3d
    num_epochs: 20
  - name: cifar10_vgg16
    num_epochs: 50

optimizers:
  - name: SGD
    learning_rate: 0.01
    hyperparams:
      momentum: 0.9

  - name: SGD_Nesterov
    optimizer_class: SGD
    learning_rate: 0.01
    hyperparams:
      momentum: 0.9
      nesterov: true

  - name: Adam
    learning_rate: 0.001
    hyperparams:
      betas: [0.9, 0.999]
      eps: 1.0e-08

  - name: AdamW
    learning_rate: 0.001
    hyperparams:
      betas: [0.9, 0.999]
      eps: 1.0e-08
      weight_decay: 0.01

  - name: RMSprop
    learning_rate: 0.001
    hyperparams:
      alpha: 0.99
      eps: 1.0e-08
```

### With Learning Rate Schedule

```yaml
optimizers:
  - name: SGD_Scheduled
    optimizer_class: SGD
    learning_rate: 0.1
    hyperparams:
      momentum: 0.9
    lr_schedule:
      epochs: [30, 60, 90]
      factors: [0.1, 0.1, 0.1]
```

## Output Structure

Results are organized for compatibility with `result_analysis.py`:

```
results/
├── mnist_mlp/
│   ├── SGD/
│   │   └── num_epochs__10__batch_size__128__lr__1e-02__momentum__9e-01/
│   │       └── random_seed__42__2025-12-16-14-30-00.json
│   └── Adam/
│       └── num_epochs__10__batch_size__128__lr__1e-03__betas__[0.9, 0.999]/
│           └── random_seed__42__2025-12-16-14-35-00.json
└── cifar10_3c3d/
    └── SGD/
        └── ...
```

Each JSON file contains:
- Training losses (per epoch)
- Test losses (per epoch)
- Test accuracies (per epoch)
- Minibatch training losses
- All hyperparameters and settings
- Metadata (optimizer name, problem name, etc.)

## Tips and Best Practices

### Start Small

Begin with a quick configuration to verify everything works:

```bash
uv run python run_benchmark.py benchmark_config_quick.yaml
```

### Use Overrides Wisely

Different problems often need different learning rates:

```yaml
overrides:
  mnist_logreg:      # Simpler problem - higher LR
    SGD:
      learning_rate: 0.1

  cifar10_vgg16:     # Harder problem - lower LR
    SGD:
      learning_rate: 0.001
```

### Monitor Progress

The script prints detailed progress:
- Current problem/optimizer combination
- Hyperparameter settings
- Training progress
- Success/failure status

### Reproducibility

Always set `random_seed` for reproducible results:

```yaml
global:
  random_seed: 42
```

### Resource Management

For large benchmarks:
- Start with fewer epochs to verify configuration
- Run on GPU if available (automatically detected)
- Consider running overnight for comprehensive benchmarks

## Troubleshooting

### Import Error: No module named 'yaml'

Install PyYAML:
```bash
uv pip install pyyaml
```

### Out of Memory (CUDA)

Reduce batch size in config:
```yaml
global:
  batch_size: 64  # or 32
```

### Configuration Not Found

Ensure the config file path is correct:
```bash
ls benchmark_config.yaml
```

### Results Not Showing in result_analysis.py

Verify results directory structure:
```bash
ls -R results/
```

Results should be in `results/[problem]/[optimizer]/[run_id]/results.json`.

## Integration with Existing Tools

### result_analysis.py

The benchmark suite generates results in the exact format expected:

```bash
# Run benchmarks
uv run python run_benchmark.py benchmark_config.yaml

# Analyze and visualize
uv run python examples/result_analysis.py
```

### Custom Analysis

Load results programmatically:

```python
import json
import glob

# Find all result files
pattern = 'results/**/results.json'
for filepath in glob.glob(pattern, recursive=True):
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"Problem: {data['testproblem']}")
    print(f"Optimizer: {data['optimizer']}")
    print(f"Final accuracy: {data['test_accuracies'][-1]:.4f}")
```

## Contributing

To add support for custom optimizers:

1. Import your optimizer class
2. Add to `OPTIMIZER_CLASSES` dict in `run_benchmark.py`
3. Use in configuration file

Example:

```python
# In run_benchmark.py
from my_optimizers import MyCustomOptimizer

OPTIMIZER_CLASSES = {
    # ... existing optimizers ...
    'MyCustom': MyCustomOptimizer,
}
```

```yaml
# In config file
optimizers:
  - name: MyCustom
    learning_rate: 0.01
    hyperparams:
      my_param: 0.9
```

## See Also

- [README.md](README.md) - Main DeepOBS documentation
- [README_PYTORCH.md](README_PYTORCH.md) - PyTorch usage guide
- [examples/result_analysis.py](examples/result_analysis.py) - Result analysis script
- [examples/multiple_test_problems.py](examples/multiple_test_problems.py) - Alternative approach

## Support

For issues or questions:
- Check [GitHub Issues](https://github.com/fsschneider/DeepOBS/issues)
- Review [API_REFERENCE.md](docs/API_REFERENCE.md)
- Read [KNOWN_ISSUES.md](docs/KNOWN_ISSUES.md)
