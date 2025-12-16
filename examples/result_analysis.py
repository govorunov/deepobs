"""
Result analysis and plotting example for DeepOBS PyTorch.

This script demonstrates:
- Loading results from JSON files
- Creating publication-quality plots
- Statistical analysis of results
- Comparing multiple optimizers

Run: python result_analysis.py
"""

import json
import os
import glob
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

    # Find all results.json files
    pattern = os.path.join(results_dir, '**', 'results.json')
    result_files = glob.glob(pattern, recursive=True)

    print(f"Found {len(result_files)} result files")

    for filepath in result_files:
        # Parse path to extract testproblem and optimizer
        parts = filepath.split(os.sep)

        # Assuming structure: results_dir/testproblem/optimizer/run_id/results.json
        if len(parts) >= 4:
            testproblem = parts[-4]
            optimizer = parts[-3]

            with open(filepath, 'r') as f:
                data = json.load(f)

            results[(testproblem, optimizer)] = data

    return results


def plot_learning_curves(results, output_dir='./results/plots'):
    """Plot learning curves for all test problems and optimizers.

    Args:
        results: Dictionary of results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Group by test problem
    by_testproblem = defaultdict(dict)
    for (testproblem, optimizer), data in results.items():
        by_testproblem[testproblem][optimizer] = data

    # Create plots for each test problem
    for testproblem, optimizer_results in by_testproblem.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Plot test loss
        ax = axes[0]
        for optimizer, data in optimizer_results.items():
            if 'test_losses' in data:
                ax.plot(data['test_losses'], label=optimizer, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Test Loss', fontsize=12)
        ax.set_title(f'{testproblem} - Test Loss', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot test accuracy
        ax = axes[1]
        for optimizer, data in optimizer_results.items():
            if 'test_accuracies' in data:
                ax.plot(data['test_accuracies'], label=optimizer, linewidth=2)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title(f'{testproblem} - Test Accuracy', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_file = os.path.join(output_dir, f'{testproblem}_learning_curves.png')
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
        plt.close()


def plot_comparison_bar(results, metric='final_accuracy', output_dir='./results/plots'):
    """Create bar plot comparing optimizers across test problems.

    Args:
        results: Dictionary of results
        metric: Metric to compare ('final_accuracy' or 'final_loss')
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect data
    testproblems = sorted(set(tp for tp, _ in results.keys()))
    optimizers = sorted(set(opt for _, opt in results.keys()))

    data_matrix = np.zeros((len(testproblems), len(optimizers)))

    for i, testproblem in enumerate(testproblems):
        for j, optimizer in enumerate(optimizers):
            if (testproblem, optimizer) in results:
                res = results[(testproblem, optimizer)]
                if metric == 'final_accuracy' and 'test_accuracies' in res:
                    data_matrix[i, j] = res['test_accuracies'][-1]
                elif metric == 'final_loss' and 'test_losses' in res:
                    data_matrix[i, j] = res['test_losses'][-1]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(testproblems))
    width = 0.8 / len(optimizers)

    for j, optimizer in enumerate(optimizers):
        offset = (j - len(optimizers) / 2) * width + width / 2
        ax.bar(x + offset, data_matrix[:, j], width, label=optimizer)

    ax.set_xlabel('Test Problem', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f'Optimizer Comparison - {metric.replace("_", " ").title()}', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(testproblems, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'comparison_{metric}.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def compute_statistics(results):
    """Compute statistics across all results.

    Args:
        results: Dictionary of results

    Returns:
        Dictionary of statistics
    """
    stats = defaultdict(lambda: defaultdict(list))

    for (testproblem, optimizer), data in results.items():
        if 'test_accuracies' in data:
            final_acc = data['test_accuracies'][-1]
            stats[optimizer]['accuracies'].append(final_acc)
            stats[optimizer]['testproblems'].append(testproblem)

        if 'test_losses' in data:
            final_loss = data['test_losses'][-1]
            stats[optimizer]['losses'].append(final_loss)

    # Compute summary statistics
    summary = {}
    for optimizer, data in stats.items():
        if data['accuracies']:
            summary[optimizer] = {
                'mean_accuracy': np.mean(data['accuracies']),
                'std_accuracy': np.std(data['accuracies']),
                'min_accuracy': np.min(data['accuracies']),
                'max_accuracy': np.max(data['accuracies']),
                'num_problems': len(data['accuracies']),
            }

    return summary


def print_statistics(stats):
    """Print statistics in a formatted table.

    Args:
        stats: Dictionary of statistics from compute_statistics
    """
    print("\n" + "=" * 70)
    print("STATISTICS: Optimizer Performance Summary")
    print("=" * 70)
    print(f"\n{'Optimizer':<20} {'Mean Acc':<12} {'Std Acc':<12} {'Min Acc':<12} {'Max Acc':<12}")
    print("-" * 70)

    for optimizer, data in sorted(stats.items()):
        print(f"{optimizer:<20} "
              f"{data['mean_accuracy']:<12.4f} "
              f"{data['std_accuracy']:<12.4f} "
              f"{data['min_accuracy']:<12.4f} "
              f"{data['max_accuracy']:<12.4f}")

    print("-" * 70)


def create_performance_profile(results, output_dir='./results/plots'):
    """Create a performance profile plot.

    Performance profiles show the fraction of problems solved
    within a certain factor of the best optimizer.

    Args:
        results: Dictionary of results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect final accuracies
    testproblems = sorted(set(tp for tp, _ in results.keys()))
    optimizers = sorted(set(opt for _, opt in results.keys()))

    # Matrix: testproblems x optimizers
    accuracy_matrix = np.zeros((len(testproblems), len(optimizers)))

    for i, testproblem in enumerate(testproblems):
        for j, optimizer in enumerate(optimizers):
            if (testproblem, optimizer) in results:
                res = results[(testproblem, optimizer)]
                if 'test_accuracies' in res:
                    accuracy_matrix[i, j] = res['test_accuracies'][-1]

    # Compute performance ratios (higher accuracy is better)
    best_per_problem = accuracy_matrix.max(axis=1, keepdims=True)
    # Avoid division by zero
    best_per_problem[best_per_problem == 0] = 1e-8
    performance_ratios = accuracy_matrix / best_per_problem

    # Create performance profile
    fig, ax = plt.subplots(figsize=(10, 6))

    tau_values = np.linspace(0.8, 1.0, 100)

    for j, optimizer in enumerate(optimizers):
        fractions = []
        for tau in tau_values:
            # Fraction of problems where optimizer is within tau of best
            fraction = np.mean(performance_ratios[:, j] >= tau)
            fractions.append(fraction)

        ax.plot(tau_values, fractions, label=optimizer, linewidth=2)

    ax.set_xlabel('Performance Ratio (τ)', fontsize=12)
    ax.set_ylabel('Fraction of Problems Solved', fontsize=12)
    ax.set_title('Performance Profile', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.8, 1.0])
    ax.set_ylim([0, 1.05])

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'performance_profile.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def analyze_convergence_speed(results, threshold=0.9):
    """Analyze convergence speed of optimizers.

    Args:
        results: Dictionary of results
        threshold: Accuracy threshold to consider converged (default: 90%)

    Returns:
        Dictionary of convergence speeds
    """
    convergence = defaultdict(dict)

    for (testproblem, optimizer), data in results.items():
        if 'test_accuracies' in data:
            accuracies = data['test_accuracies']

            # Find first epoch where accuracy exceeds threshold
            converged_epoch = None
            for epoch, acc in enumerate(accuracies):
                if acc >= threshold:
                    converged_epoch = epoch + 1
                    break

            convergence[testproblem][optimizer] = converged_epoch

    return convergence


def print_convergence_analysis(convergence):
    """Print convergence analysis.

    Args:
        convergence: Dictionary from analyze_convergence_speed
    """
    print("\n" + "=" * 70)
    print("CONVERGENCE ANALYSIS (Epochs to reach 90% accuracy)")
    print("=" * 70)

    for testproblem in sorted(convergence.keys()):
        print(f"\n{testproblem}:")
        for optimizer, epoch in sorted(convergence[testproblem].items()):
            if epoch is not None:
                print(f"  {optimizer:<20} {epoch:>3} epochs")
            else:
                print(f"  {optimizer:<20} Did not converge")


def main():
    """Main function for result analysis."""
    print("=" * 70)
    print("DeepOBS PyTorch - Result Analysis Example")
    print("=" * 70)

    # Load results
    results_dir = './results'
    print(f"\nLoading results from: {results_dir}")

    results = load_results(results_dir)

    if not results:
        print("\nNo results found!")
        print("Please run one of the following first:")
        print("  - basic_usage.py")
        print("  - custom_optimizer_benchmark.py")
        print("  - multiple_test_problems.py")
        return

    print(f"Loaded {len(results)} result files")

    # Compute statistics
    stats = compute_statistics(results)
    print_statistics(stats)

    # Analyze convergence
    convergence = analyze_convergence_speed(results, threshold=0.9)
    print_convergence_analysis(convergence)

    # Create plots
    print("\n" + "=" * 70)
    print("Creating plots...")
    print("=" * 70)

    try:
        plot_learning_curves(results)
        plot_comparison_bar(results, metric='final_accuracy')
        plot_comparison_bar(results, metric='final_loss')
        create_performance_profile(results)

        print("\n" + "=" * 70)
        print("All plots created successfully!")
        print("Check ./results/plots/ directory")
        print("=" * 70)

    except Exception as e:
        print(f"\nError creating plots: {e}")
        print("Make sure matplotlib is installed: pip install matplotlib")


if __name__ == '__main__':
    main()
