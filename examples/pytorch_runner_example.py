"""Example script demonstrating the PyTorch StandardRunner.

This script shows how to use the StandardRunner to train optimizers on
DeepOBS test problems. You can run it from the command line or modify it
for programmatic use.

Usage (command line):
    python pytorch_runner_example.py mnist_logreg --batch_size 128 --num_epochs 5 --learning_rate 0.01

Usage (programmatic):
    Uncomment the programmatic_run() call at the bottom.
"""

import sys
import os

# Add parent directory to path to import deepobs
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.optim as optim
from deepobs.pytorch.runners import StandardRunner


def command_line_run():
    """Run with command-line argument parsing."""
    # Define the optimizer and its hyperparameters
    optimizer_class = optim.SGD
    hyperparams = [
        {"name": "momentum", "type": float, "default": 0.0},
        {"name": "nesterov", "type": bool, "default": False}
    ]

    # Create runner
    runner = StandardRunner(optimizer_class, hyperparams)

    # Run with command-line arguments
    # Try: python pytorch_runner_example.py mnist_logreg --batch_size 128 --num_epochs 5 --learning_rate 0.01
    runner.run()


def programmatic_run():
    """Run with all arguments specified programmatically."""
    # Define the optimizer and its hyperparameters
    optimizer_class = optim.SGD
    hyperparams = [
        {"name": "momentum", "type": float, "default": 0.0},
        {"name": "nesterov", "type": bool, "default": False}
    ]

    # Create runner
    runner = StandardRunner(optimizer_class, hyperparams)

    # Run with all arguments specified
    runner._run(
        testproblem="mnist_logreg",
        weight_decay=None,
        batch_size=128,
        num_epochs=5,
        learning_rate=0.01,
        lr_sched_epochs=None,
        lr_sched_factors=None,
        random_seed=42,
        data_dir=None,
        output_dir="results",
        train_log_interval=10,
        print_train_iter=True,  # Print training progress
        no_logs=False,
        momentum=0.9,
        nesterov=False
    )


def learning_rate_schedule_example():
    """Example with learning rate scheduling."""
    optimizer_class = optim.SGD
    hyperparams = [{"name": "momentum", "type": float, "default": 0.0}]

    runner = StandardRunner(optimizer_class, hyperparams)

    # Start with LR=0.1, multiply by 0.1 at epoch 3
    runner._run(
        testproblem="mnist_mlp",
        batch_size=128,
        num_epochs=5,
        learning_rate=0.1,
        lr_sched_epochs=[3],
        lr_sched_factors=[0.1],  # LR becomes 0.01 at epoch 3
        random_seed=42,
        output_dir="results",
        train_log_interval=10,
        print_train_iter=True,
        no_logs=False,
        momentum=0.9
    )


def adam_optimizer_example():
    """Example using Adam optimizer."""
    optimizer_class = optim.Adam
    hyperparams = [
        {"name": "betas", "type": tuple, "default": (0.9, 0.999)},
        {"name": "eps", "type": float, "default": 1e-8}
    ]

    runner = StandardRunner(optimizer_class, hyperparams)

    runner._run(
        testproblem="mnist_mlp",
        batch_size=128,
        num_epochs=5,
        learning_rate=0.001,
        random_seed=42,
        output_dir="results",
        train_log_interval=10,
        print_train_iter=True,
        no_logs=False,
        betas=(0.9, 0.999),
        eps=1e-8
    )


if __name__ == "__main__":
    # Default: run with command-line arguments
    command_line_run()

    # Uncomment one of these to run programmatically:
    # programmatic_run()
    # learning_rate_schedule_example()
    # adam_optimizer_example()
