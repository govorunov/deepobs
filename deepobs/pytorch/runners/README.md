# DeepOBS PyTorch Runners

This module provides the `StandardRunner` class for executing optimizer benchmarks on DeepOBS test problems.

## Overview

The `StandardRunner` handles the complete training workflow:
1. Parsing command-line arguments
2. Setting up the test problem (dataset + model)
3. Creating and configuring the optimizer
4. Training loop with periodic evaluation
5. Learning rate scheduling
6. Metric logging and saving results to JSON

## Basic Usage

```python
import torch.optim as optim
from deepobs.pytorch.runners import StandardRunner

# Define optimizer and hyperparameters
optimizer_class = optim.SGD
hyperparams = [
    {"name": "momentum", "type": float, "default": 0.0},
    {"name": "nesterov", "type": bool, "default": False}
]

# Create runner
runner = StandardRunner(optimizer_class, hyperparams)

# Run training (all arguments can be passed programmatically)
runner._run(
    testproblem="mnist_mlp",
    batch_size=128,
    num_epochs=10,
    learning_rate=0.01,
    lr_sched_epochs=None,
    lr_sched_factors=None,
    random_seed=42,
    data_dir=None,
    output_dir="results",
    train_log_interval=10,
    print_train_iter=False,
    no_logs=False,
    momentum=0.9,
    nesterov=False
)
```

## Command-Line Interface

The `run()` method provides automatic command-line argument parsing. Any argument not provided programmatically will be requested from the command line.

```python
# Create a script (e.g., run_sgd.py)
import torch.optim as optim
from deepobs.pytorch.runners import StandardRunner

optimizer_class = optim.SGD
hyperparams = [
    {"name": "momentum", "type": float, "default": 0.0},
    {"name": "nesterov", "type": bool, "default": False}
]

runner = StandardRunner(optimizer_class, hyperparams)
runner.run()  # Will parse all arguments from command line
```

Then run from command line:

```bash
python run_sgd.py mnist_mlp --batch_size 128 --num_epochs 10 --learning_rate 0.01 --momentum 0.9
```

## Learning Rate Schedules

You can specify learning rate schedules that change the learning rate at specific epochs:

```python
# Start with LR=0.1, then multiply by 0.1 at epoch 50 and 0.01 at epoch 100
runner._run(
    testproblem="cifar10_3c3d",
    batch_size=128,
    num_epochs=150,
    learning_rate=0.1,
    lr_sched_epochs=[50, 100],
    lr_sched_factors=[0.1, 0.01],  # LR becomes 0.01 at epoch 50, 0.001 at epoch 100
    random_seed=42,
    # ... other arguments
)
```

## Output Format

Results are saved to JSON files in the following structure:

```
results/
└── mnist_mlp/
    └── SGD/
        └── num_epochs__10__batch_size__128__lr__1e-02__momentum__9e-01/
            └── random_seed__42__2025-12-13-16-00-00.json
```

The JSON file contains:

```json
{
  "train_losses": [2.3, 1.8, 1.5, ...],
  "test_losses": [2.4, 1.9, 1.6, ...],
  "minibatch_train_losses": [2.5, 2.4, 2.3, ...],
  "train_accuracies": [0.15, 0.35, 0.50, ...],
  "test_accuracies": [0.14, 0.33, 0.48, ...],
  "optimizer": "SGD",
  "testproblem": "mnist_mlp",
  "batch_size": 128,
  "num_epochs": 10,
  "learning_rate": 0.01,
  "lr_sched_epochs": null,
  "lr_sched_factors": null,
  "random_seed": 42,
  "train_log_interval": 10,
  "weight_decay": null,
  "hyperparams": {
    "momentum": 0.9,
    "nesterov": false
  }
}
```

## Available Test Problems

Current test problems (Phase 3 complete):
- `mnist_logreg` - Logistic regression on MNIST
- `mnist_mlp` - Multi-layer perceptron on MNIST
- `cifar10_3c3d` - 3-layer CNN on CIFAR-10

More test problems will be added in future phases.

## Hyperparameter Specification

When creating a `StandardRunner`, you must specify the optimizer's hyperparameters (excluding learning rate, which is handled separately):

```python
hyperparams = [
    {
        "name": "momentum",        # Exact parameter name for optimizer
        "type": float,             # Type (int, float, bool)
        "default": 0.0             # Optional default value
    },
    {
        "name": "nesterov",
        "type": bool,
        "default": False
    }
]
```

## Utility Functions

The `runner_utils` module provides helper functions:

### `float2str(x)`
Converts a float to a compact string representation:
```python
>>> runner_utils.float2str(0.001)
'1e-03'
```

### `make_lr_schedule(lr_base, lr_sched_epochs, lr_sched_factors)`
Creates a learning rate schedule dictionary:
```python
>>> runner_utils.make_lr_schedule(0.3, [50, 100], [0.1, 0.01])
{0: 0.3, 50: 0.03, 100: 0.003}
```

### `make_run_name(...)`
Generates descriptive folder and file names for outputs:
```python
>>> folder, filename = runner_utils.make_run_name(
...     weight_decay=0.001,
...     batch_size=128,
...     num_epochs=10,
...     learning_rate=0.01,
...     lr_sched_epochs=None,
...     lr_sched_factors=None,
...     random_seed=42,
...     momentum=0.9
... )
>>> folder
'num_epochs__10__batch_size__128__weight_decay__1e-03__momentum__9e-01__lr__1e-02'
>>> filename
'random_seed__42__2025-12-13-16-00-00'
```

## Device Management

The runner automatically uses CUDA if available, otherwise falls back to CPU. The device is determined by the test problem's initialization.

## Reproducibility

For reproducible results, the runner sets random seeds for:
- PyTorch (`torch.manual_seed`)
- NumPy (`np.random.seed`)
- CUDA (`torch.cuda.manual_seed_all`)

It also sets:
```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Note: This may impact performance but ensures reproducibility.

## Differences from TensorFlow Version

The PyTorch runner is simpler than the TensorFlow version:

1. **No Session Management**: PyTorch uses eager execution, so no `tf.Session()` needed
2. **Automatic Batch Norm Updates**: No need for `UPDATE_OPS` collection
3. **Simpler Phase Switching**: `model.train()` and `model.eval()` handle train/test mode
4. **Manual LR Scheduling**: Updates learning rate via `param_groups` for exact TF compatibility
5. **No TensorBoard Logging**: Currently only JSON output (TensorBoard support may be added later)

## Example: Comparing Optimizers

```python
import torch.optim as optim
from deepobs.pytorch.runners import StandardRunner

# Test SGD
sgd_runner = StandardRunner(
    optim.SGD,
    [{"name": "momentum", "type": float, "default": 0.0}]
)
sgd_runner._run(
    testproblem="mnist_mlp",
    batch_size=128,
    num_epochs=10,
    learning_rate=0.01,
    momentum=0.9,
    random_seed=42,
    output_dir="results"
)

# Test Adam
adam_runner = StandardRunner(
    optim.Adam,
    [
        {"name": "betas", "type": tuple, "default": (0.9, 0.999)},
        {"name": "eps", "type": float, "default": 1e-8}
    ]
)
adam_runner._run(
    testproblem="mnist_mlp",
    batch_size=128,
    num_epochs=10,
    learning_rate=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    random_seed=42,
    output_dir="results"
)

# Results can be analyzed with: deepobs analyze results/
```

## See Also

- Test problems: `deepobs.pytorch.testproblems`
- Datasets: `deepobs.pytorch.datasets`
- Analysis tools: `deepobs analyze` command (generates interactive HTML reports)
