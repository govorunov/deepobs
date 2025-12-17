#!/usr/bin/env python
"""
Quick result analysis script for DeepOBS benchmark results.
Adapted to work with the actual file structure from StandardRunner.
"""

import json
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_results(results_dir):
    """Load all result JSON files from a directory.

    Args:
        results_dir: Directory containing result files

    Returns:
        Dictionary mapping (testproblem, optimizer) to results
    """
    results = {}

    # Find all JSON files (excluding results.json since actual files have different names)
    pattern = os.path.join(results_dir, '**', '*.json')
    result_files = glob.glob(pattern, recursive=True)

    print(f"Found {len(result_files)} result files")

    for filepath in result_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Extract testproblem and optimizer from the JSON data itself
            if 'testproblem' in data and 'optimizer' in data:
                testproblem = data['testproblem']
                optimizer = data['optimizer']

                # Use the most recent result for each combination
                # (or you could average multiple runs)
                results[(testproblem, optimizer)] = data

        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return results


def plot_learning_curves(results, output_dir='./results/plots'):
    """Plot learning curves for all test problems and optimizers."""
    os.makedirs(output_dir, exist_ok=True)

    # Group by test problem
    by_testproblem = defaultdict(dict)
    for (testproblem, optimizer), data in results.items():
        by_testproblem[testproblem][optimizer] = data

    # Create plots for each test problem
    for testproblem, optimizer_results in by_testproblem.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot test loss
        ax = axes[0]
        for optimizer, data in optimizer_results.items():
            if 'test_losses' in data:
                epochs = range(len(data['test_losses']))
                ax.plot(epochs, data['test_losses'], label=optimizer, linewidth=2, marker='o')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Test Loss', fontsize=12)
        ax.set_title(f'{testproblem} - Test Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # Plot test accuracy
        ax = axes[1]
        for optimizer, data in optimizer_results.items():
            if 'test_accuracies' in data:
                epochs = range(len(data['test_accuracies']))
                accuracies = [acc * 100 for acc in data['test_accuracies']]  # Convert to percentage
                ax.plot(epochs, accuracies, label=optimizer, linewidth=2, marker='o')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title(f'{testproblem} - Test Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'{testproblem}_learning_curves.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def plot_comparison_bar(results, output_dir='./results/plots'):
    """Create bar plot comparing optimizers across test problems."""
    os.makedirs(output_dir, exist_ok=True)

    # Collect data
    testproblems = sorted(set(tp for tp, _ in results.keys()))
    optimizers = sorted(set(opt for _, opt in results.keys()))

    # Create subplots for accuracy and loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Final Accuracy comparison
    ax = axes[0]
    x = np.arange(len(testproblems))
    width = 0.8 / len(optimizers)

    for j, optimizer in enumerate(optimizers):
        accuracies = []
        for testproblem in testproblems:
            if (testproblem, optimizer) in results:
                res = results[(testproblem, optimizer)]
                if 'test_accuracies' in res:
                    accuracies.append(res['test_accuracies'][-1] * 100)  # Convert to percentage
                else:
                    accuracies.append(0)
            else:
                accuracies.append(0)

        offset = (j - len(optimizers) / 2) * width + width / 2
        ax.bar(x + offset, accuracies, width, label=optimizer)

    ax.set_xlabel('Test Problem', fontsize=12)
    ax.set_ylabel('Final Test Accuracy (%)', fontsize=12)
    ax.set_title('Optimizer Comparison - Final Test Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(testproblems, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Final Loss comparison
    ax = axes[1]
    for j, optimizer in enumerate(optimizers):
        losses = []
        for testproblem in testproblems:
            if (testproblem, optimizer) in results:
                res = results[(testproblem, optimizer)]
                if 'test_losses' in res:
                    losses.append(res['test_losses'][-1])
                else:
                    losses.append(0)
            else:
                losses.append(0)

        offset = (j - len(optimizers) / 2) * width + width / 2
        ax.bar(x + offset, losses, width, label=optimizer)

    ax.set_xlabel('Test Problem', fontsize=12)
    ax.set_ylabel('Final Test Loss', fontsize=12)
    ax.set_title('Optimizer Comparison - Final Test Loss', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(testproblems, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'optimizer_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def print_detailed_summary(results):
    """Print detailed summary of all results."""
    print("\n" + "=" * 80)
    print("DETAILED RESULTS SUMMARY")
    print("=" * 80)

    # Group by test problem
    by_testproblem = defaultdict(dict)
    for (testproblem, optimizer), data in results.items():
        by_testproblem[testproblem][optimizer] = data

    for testproblem in sorted(by_testproblem.keys()):
        print(f"\n{testproblem.upper()}")
        print("-" * 80)
        print(f"{'Optimizer':<15} {'Final Acc (%)':<15} {'Final Loss':<15} {'Epochs':<10} {'LR':<10}")
        print("-" * 80)

        for optimizer in sorted(by_testproblem[testproblem].keys()):
            data = by_testproblem[testproblem][optimizer]
            final_acc = data['test_accuracies'][-1] * 100 if 'test_accuracies' in data else 0
            final_loss = data['test_losses'][-1] if 'test_losses' in data else 0
            num_epochs = data.get('num_epochs', 0)
            lr = data.get('learning_rate', 0)

            print(f"{optimizer:<15} {final_acc:<15.2f} {final_loss:<15.4f} {num_epochs:<10} {lr:<10.4f}")

        print()


def print_statistics(results):
    """Print statistics across all results."""
    stats = defaultdict(lambda: defaultdict(list))

    for (testproblem, optimizer), data in results.items():
        if 'test_accuracies' in data:
            final_acc = data['test_accuracies'][-1] * 100
            stats[optimizer]['accuracies'].append(final_acc)
            stats[optimizer]['testproblems'].append(testproblem)

        if 'test_losses' in data:
            final_loss = data['test_losses'][-1]
            stats[optimizer]['losses'].append(final_loss)

    print("\n" + "=" * 80)
    print("OPTIMIZER PERFORMANCE SUMMARY (Across All Test Problems)")
    print("=" * 80)
    print(f"\n{'Optimizer':<15} {'Avg Acc (%)':<15} {'Std Acc':<15} {'Avg Loss':<15} {'Problems':<10}")
    print("-" * 80)

    for optimizer in sorted(stats.keys()):
        data = stats[optimizer]
        if data['accuracies']:
            mean_acc = np.mean(data['accuracies'])
            std_acc = np.std(data['accuracies'])
            mean_loss = np.mean(data['losses']) if data['losses'] else 0
            num_problems = len(data['accuracies'])

            print(f"{optimizer:<15} {mean_acc:<15.2f} {std_acc:<15.2f} {mean_loss:<15.4f} {num_problems:<10}")

    print("-" * 80)


def main():
    """Main function for result analysis."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze DeepOBS benchmark results and generate plots.'
    )
    parser.add_argument(
        'results_dir',
        nargs='?',
        default='./results',
        help='Path to the directory containing result files (default: ./results)'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("DeepOBS Benchmark Results Analysis")
    print("=" * 80)

    # Load results
    results_dir = args.results_dir
    print(f"\nLoading results from: {results_dir}")

    results = load_results(results_dir)

    if not results:
        print("\nNo results found!")
        return

    print(f"\nLoaded {len(results)} unique (test_problem, optimizer) combinations")

    # Print summaries
    print_detailed_summary(results)
    print_statistics(results)

    # Create plots
    print("\n" + "=" * 80)
    print("Creating plots...")
    print("=" * 80 + "\n")

    try:
        plot_learning_curves(results)
        plot_comparison_bar(results)

        print("\n" + "=" * 80)
        print("✓ All plots created successfully!")
        print(f"Check ./results/plots/ directory")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error creating plots: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
