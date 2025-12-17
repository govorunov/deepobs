#!/usr/bin/env python
"""
Configuration-driven benchmark suite for DeepOBS PyTorch.

This script reads a YAML configuration file and runs a comprehensive benchmark
of multiple optimizers across multiple test problems using StandardRunner.

Results are saved in a format compatible with result_analysis.py.

Usage:
    python run_benchmark.py [config_file]
    python run_benchmark.py benchmark_config.yaml

If no config file is specified, uses 'benchmark_config.yaml' by default.
"""

import argparse
import importlib
import os
import sys
import yaml
import torch.optim as optim
from datetime import datetime
from deepobs.pytorch.runners import StandardRunner


def load_config(config_file):
    """Load benchmark configuration from YAML file.

    Args:
        config_file: Path to YAML configuration file

    Returns:
        Dictionary containing configuration
    """
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config


def get_optimizer_class(optimizer_name, optimizer_class_path=None):
    """Get optimizer class by dynamic import.

    Args:
        optimizer_name: Name of the optimizer (used as fallback for torch.optim)
        optimizer_class_path: Optional full import path (e.g., 'my_pkg.optimizers.MyOptimizer')

    Returns:
        Optimizer class

    Raises:
        ImportError: If the optimizer class cannot be imported
        AttributeError: If the class is not found in the module
    """
    if optimizer_class_path:
        # Use custom import path from config
        try:
            module_path, class_name = optimizer_class_path.rsplit('.', 1)
            module = importlib.import_module(module_path)
            optimizer_class = getattr(module, class_name)
            return optimizer_class
        except (ValueError, ImportError, AttributeError) as e:
            raise ImportError(
                f"Failed to import optimizer from '{optimizer_class_path}': {e}"
            ) from e
    else:
        # Try to import from torch.optim using optimizer_name
        try:
            optimizer_class = getattr(optim, optimizer_name)
            return optimizer_class
        except AttributeError as e:
            raise ImportError(
                f"Optimizer '{optimizer_name}' not found in torch.optim. "
                f"Specify 'optimizer_class' in config to use a custom optimizer."
            ) from e


def create_hyperparams_spec(optimizer_class, optimizer_params):
    """Create hyperparameters specification for StandardRunner.

    Args:
        optimizer_class: PyTorch optimizer class
        optimizer_params: Dictionary of optimizer parameters (excluding lr, lr_schedule, name, optimizer_class)

    Returns:
        List of hyperparameter specifications for StandardRunner
    """
    hyperparams_spec = []

    for param_name, param_value in optimizer_params.items():
        # Determine type from value
        if isinstance(param_value, bool):
            param_type = bool
        elif isinstance(param_value, int):
            param_type = int
        elif isinstance(param_value, float):
            param_type = float
        elif isinstance(param_value, (list, tuple)):
            param_type = type(param_value)
        else:
            param_type = type(param_value)

        hyperparams_spec.append({
            "name": param_name,
            "type": param_type,
            "default": param_value
        })

    return hyperparams_spec


def get_problem_settings(problem_config, global_config):
    """Get settings for a test problem, merging with global defaults.

    Args:
        problem_config: Problem-specific configuration
        global_config: Global configuration

    Returns:
        Dictionary of settings for the problem
    """
    settings = {
        'num_epochs': global_config.get('num_epochs', 10),
        'batch_size': global_config.get('batch_size', 128),
    }

    # Override with problem-specific settings
    if 'num_epochs' in problem_config:
        settings['num_epochs'] = problem_config['num_epochs']
    if 'batch_size' in problem_config:
        settings['batch_size'] = problem_config['batch_size']

    return settings


def get_optimizer_settings(optimizer_config, problem_name, overrides):
    """Get settings for an optimizer, applying problem-specific overrides.

    Args:
        optimizer_config: Optimizer configuration
        problem_name: Name of the test problem
        overrides: Problem-specific overrides from config

    Returns:
        Dictionary with 'learning_rate', 'lr_sched_epochs', 'lr_sched_factors', and 'params'
    """
    # Known special parameters that should not be passed to optimizer
    SPECIAL_PARAMS = {'name', 'optimizer_class', 'learning_rate', 'lr_schedule'}

    # Start with base optimizer config - copy all params
    all_params = dict(optimizer_config)

    # Extract learning rate
    learning_rate = all_params.pop('learning_rate')

    # Extract learning rate schedule if present
    lr_schedule = all_params.pop('lr_schedule', None)
    lr_sched_epochs = None
    lr_sched_factors = None
    if lr_schedule:
        lr_sched_epochs = lr_schedule.get('epochs')
        lr_sched_factors = lr_schedule.get('factors')

    # Remove other special params
    all_params.pop('name', None)
    all_params.pop('optimizer_class', None)

    # Apply problem-specific overrides
    if problem_name in overrides:
        optimizer_name = optimizer_config['name']
        if optimizer_name in overrides[problem_name]:
            override = overrides[problem_name][optimizer_name]

            # Override learning rate if specified
            if 'learning_rate' in override:
                learning_rate = override['learning_rate']

            # Override lr_schedule if specified
            if 'lr_schedule' in override:
                lr_sched = override['lr_schedule']
                lr_sched_epochs = lr_sched.get('epochs')
                lr_sched_factors = lr_sched.get('factors')

            # Override other parameters (excluding special ones)
            for key, value in override.items():
                if key not in SPECIAL_PARAMS:
                    all_params[key] = value

    return {
        'learning_rate': learning_rate,
        'lr_sched_epochs': lr_sched_epochs,
        'lr_sched_factors': lr_sched_factors,
        'params': all_params  # All other optimizer parameters
    }


def run_single_benchmark(problem_name, problem_settings, optimizer_config,
                         global_config, overrides):
    """Run a single benchmark (one optimizer on one problem).

    Args:
        problem_name: Name of the test problem
        problem_settings: Settings for the problem
        optimizer_config: Configuration for the optimizer
        global_config: Global configuration
        overrides: Problem-specific overrides

    Returns:
        True if successful, False otherwise
    """
    optimizer_name = optimizer_config['name']

    print(f"\n{'='*80}")
    print(f"Running: {optimizer_name} on {problem_name}")
    print(f"{'='*80}")

    try:
        # Get optimizer class
        optimizer_class_path = optimizer_config.get('optimizer_class')
        optimizer_class = get_optimizer_class(optimizer_name, optimizer_class_path)

        # Get optimizer settings with overrides
        optimizer_settings = get_optimizer_settings(
            optimizer_config, problem_name, overrides
        )

        # Create hyperparameters specification
        hyperparams_spec = create_hyperparams_spec(
            optimizer_class,
            optimizer_settings['params']
        )

        # Create StandardRunner
        runner = StandardRunner(optimizer_class, hyperparams_spec)

        # Prepare run parameters
        run_params = {
            'testproblem': problem_name,
            'batch_size': problem_settings['batch_size'],
            'num_epochs': problem_settings['num_epochs'],
            'learning_rate': optimizer_settings['learning_rate'],
            'lr_sched_epochs': optimizer_settings['lr_sched_epochs'],
            'lr_sched_factors': optimizer_settings['lr_sched_factors'],
            'random_seed': global_config.get('random_seed', 42),
            'data_dir': global_config.get('data_dir'),
            'output_dir': global_config.get('output_dir', './results'),
            'train_log_interval': global_config.get('train_log_interval', 100),
            'print_train_iter': global_config.get('print_train_iter', False),
            'no_logs': False,
            'weight_decay': None,
        }

        # Add optimizer parameters as **kwargs
        run_params.update(optimizer_settings['params'])

        # Print configuration
        print(f"Test Problem: {problem_name}")
        print(f"Optimizer: {optimizer_name}")
        print(f"Learning Rate: {optimizer_settings['learning_rate']}")
        print(f"Batch Size: {problem_settings['batch_size']}")
        print(f"Epochs: {problem_settings['num_epochs']}")
        if optimizer_settings['lr_sched_epochs']:
            print(f"LR Schedule: epochs={optimizer_settings['lr_sched_epochs']}, "
                  f"factors={optimizer_settings['lr_sched_factors']}")
        if optimizer_settings['params']:
            print(f"Optimizer Parameters: {optimizer_settings['params']}")
        print()

        # Run the benchmark
        runner._run(**run_params)

        print(f"\n{'='*80}")
        print(f"✓ Completed: {optimizer_name} on {problem_name}")
        print(f"{'='*80}")

        return True

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"✗ Failed: {optimizer_name} on {problem_name}")
        print(f"Error: {e}")
        print(f"{'='*80}")
        return False


def print_summary(results):
    """Print summary of benchmark results.

    Args:
        results: List of (problem, optimizer, success, output_dir) tuples
    """
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    total = len(results)
    successful = sum(1 for _, _, success, _ in results if success)
    failed = total - successful

    print(f"\nTotal benchmarks: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed benchmarks:")
        for problem, optimizer, success, _ in results:
            if not success:
                print(f"  - {optimizer} on {problem}")

    print("\n" + "="*80)
    print(f"Results saved to: {results[0][3] if results else './results'}")
    print("Run 'python examples/result_analysis.py' to analyze and visualize results")
    print("="*80)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run DeepOBS benchmark suite from configuration file"
    )
    parser.add_argument(
        'config',
        nargs='?',
        default='benchmark_config.yaml',
        help='Path to YAML configuration file (default: benchmark_config.yaml)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be run without actually running'
    )

    args = parser.parse_args()

    # Load configuration
    print("="*80)
    print("DeepOBS Configuration-Driven Benchmark Suite")
    print("="*80)
    print(f"\nLoading configuration from: {args.config}")

    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        return 1

    # Extract configuration sections
    global_config = config.get('global', {})
    test_problems = config.get('test_problems', [])
    optimizers = config.get('optimizers', [])
    overrides = config.get('overrides', {})

    # Validate configuration
    if not test_problems:
        print("Error: No test problems specified in configuration")
        return 1

    if not optimizers:
        print("Error: No optimizers specified in configuration")
        return 1

    # Print configuration summary
    print(f"\nTest Problems: {len(test_problems)}")
    for problem in test_problems:
        print(f"  - {problem['name']}")

    print(f"\nOptimizers: {len(optimizers)}")
    for optimizer in optimizers:
        print(f"  - {optimizer['name']}")

    print(f"\nTotal benchmarks to run: {len(test_problems) * len(optimizers)}")
    print(f"Output directory: {global_config.get('output_dir', './results')}")
    print(f"Random seed: {global_config.get('random_seed', 42)}")

    if args.dry_run:
        print("\nDry run mode - no benchmarks will be executed")
        return 0

    # Confirm with user
    print("\nStarting benchmarks...")
    start_time = datetime.now()

    # Run benchmarks
    results = []

    for problem_config in test_problems:
        problem_name = problem_config['name']
        problem_settings = get_problem_settings(problem_config, global_config)

        for optimizer_config in optimizers:
            success = run_single_benchmark(
                problem_name,
                problem_settings,
                optimizer_config,
                global_config,
                overrides
            )

            results.append((
                problem_name,
                optimizer_config['name'],
                success,
                global_config.get('output_dir', './results')
            ))

    # Print summary
    end_time = datetime.now()
    duration = end_time - start_time

    print_summary(results)
    print(f"\nTotal time: {duration}")

    # Return exit code based on failures
    failures = sum(1 for _, _, success, _ in results if not success)
    return 1 if failures > 0 else 0


if __name__ == '__main__':
    sys.exit(main())
