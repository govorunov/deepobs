"""
Multiple test problems example for DeepOBS PyTorch.

This script demonstrates:
- Running the same optimizer on multiple test problems
- Aggregating results across problems
- Comparing performance across datasets and architectures

Run: python multiple_test_problems.py
"""

import torch
import torch.optim as optim
from deepobs.pytorch import testproblems
import json
import os
from collections import defaultdict


# List of test problems to benchmark
TEST_PROBLEMS = [
    # MNIST problems (fast to train)
    'mnist_logreg',
    'mnist_mlp',
    'mnist_2c2d',

    # Fashion-MNIST problems
    'fmnist_mlp',
    'fmnist_2c2d',

    # CIFAR-10 (more challenging)
    # 'cifar10_3c3d',  # Uncomment for longer benchmark
]

# Hyperparameters for each test problem
HYPERPARAMS = {
    'mnist_logreg': {'lr': 0.1, 'epochs': 5},
    'mnist_mlp': {'lr': 0.01, 'epochs': 10},
    'mnist_2c2d': {'lr': 0.01, 'epochs': 10},
    'fmnist_mlp': {'lr': 0.01, 'epochs': 10},
    'fmnist_2c2d': {'lr': 0.01, 'epochs': 10},
    'cifar10_3c3d': {'lr': 0.001, 'epochs': 20},
}


def train_and_evaluate(test_problem_name, optimizer_class, hyperparams, device):
    """Train and evaluate on a single test problem.

    Args:
        test_problem_name: Name of the test problem
        optimizer_class: PyTorch optimizer class
        hyperparams: Dictionary with 'lr' and 'epochs'
        device: Device to use

    Returns:
        Dictionary with results
    """
    print(f"\n{'=' * 70}")
    print(f"Test Problem: {test_problem_name}")
    print(f"{'=' * 70}")

    # Get test problem
    tproblem_fn = getattr(testproblems, test_problem_name)
    tproblem = tproblem_fn(batch_size=128, device=device)
    tproblem.set_up()

    # Create optimizer
    lr = hyperparams['lr']
    num_epochs = hyperparams['epochs']

    optimizer = optimizer_class(tproblem.model.parameters(), lr=lr, momentum=0.9)

    print(f"Learning rate: {lr}")
    print(f"Number of epochs: {num_epochs}")
    print(f"Model parameters: {sum(p.numel() for p in tproblem.model.parameters()):,}")

    # Training metrics
    train_losses = []
    test_losses = []
    test_accuracies = []

    # Training loop
    for epoch in range(num_epochs):
        # Training
        tproblem.model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        for batch in tproblem.dataset.train_loader:
            optimizer.zero_grad()
            losses, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
            loss = losses.mean()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = epoch_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        # Evaluation
        tproblem.model.eval()
        epoch_test_loss = 0.0
        epoch_test_accuracy = 0.0
        num_test_batches = 0

        with torch.no_grad():
            for batch in tproblem.dataset.test_loader:
                losses, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
                epoch_test_loss += losses.mean().item()
                epoch_test_accuracy += accuracy if accuracy is not None else 0.0
                num_test_batches += 1

        avg_test_loss = epoch_test_loss / num_test_batches
        avg_test_accuracy = epoch_test_accuracy / num_test_batches
        test_losses.append(avg_test_loss)
        test_accuracies.append(avg_test_accuracy)

        print(f"Epoch {epoch + 1:2d}/{num_epochs}: "
              f"Train Loss = {avg_train_loss:.4f}, "
              f"Test Loss = {avg_test_loss:.4f}, "
              f"Test Acc = {avg_test_accuracy:.4f}")

    # Results summary
    results = {
        'test_problem': test_problem_name,
        'final_train_loss': train_losses[-1],
        'final_test_loss': test_losses[-1],
        'final_test_accuracy': test_accuracies[-1],
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'hyperparams': hyperparams,
    }

    return results


def save_results(all_results, output_file):
    """Save results to JSON file.

    Args:
        all_results: List of result dictionaries
        output_file: Path to output JSON file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Results saved to: {output_file}")
    print(f"{'=' * 70}")


def print_summary(all_results):
    """Print summary of results across all test problems.

    Args:
        all_results: List of result dictionaries
    """
    print(f"\n{'=' * 70}")
    print("SUMMARY: Performance Across All Test Problems")
    print(f"{'=' * 70}")
    print(f"\n{'Test Problem':<20} {'Final Loss':<15} {'Final Accuracy':<15}")
    print("-" * 70)

    for result in all_results:
        test_problem = result['test_problem']
        final_loss = result['final_test_loss']
        final_accuracy = result['final_test_accuracy']

        print(f"{test_problem:<20} {final_loss:<15.4f} {final_accuracy:<15.4f}")

    print("-" * 70)

    # Compute average accuracy
    avg_accuracy = sum(r['final_test_accuracy'] for r in all_results) / len(all_results)
    print(f"\nAverage Final Accuracy: {avg_accuracy:.4f}")


def analyze_convergence(all_results):
    """Analyze convergence speed across test problems.

    Args:
        all_results: List of result dictionaries
    """
    print(f"\n{'=' * 70}")
    print("CONVERGENCE ANALYSIS")
    print(f"{'=' * 70}")

    for result in all_results:
        test_problem = result['test_problem']
        test_losses = result['test_losses']

        # Find epoch where loss drops below threshold
        initial_loss = test_losses[0]
        threshold = initial_loss * 0.5  # 50% of initial loss

        converged_epoch = None
        for epoch, loss in enumerate(test_losses):
            if loss < threshold:
                converged_epoch = epoch + 1
                break

        if converged_epoch:
            print(f"{test_problem:<20} Converged at epoch {converged_epoch}")
        else:
            print(f"{test_problem:<20} Did not converge (threshold: {threshold:.4f})")


def main():
    """Main function."""
    print("=" * 70)
    print("DeepOBS PyTorch - Multiple Test Problems Example")
    print("=" * 70)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Number of test problems: {len(TEST_PROBLEMS)}")

    # Run benchmarks
    all_results = []

    for test_problem_name in TEST_PROBLEMS:
        hyperparams = HYPERPARAMS[test_problem_name]

        results = train_and_evaluate(
            test_problem_name=test_problem_name,
            optimizer_class=optim.SGD,
            hyperparams=hyperparams,
            device=device
        )

        all_results.append(results)

    # Save results
    output_file = './results/multiple_problems_benchmark.json'
    save_results(all_results, output_file)

    # Print summary
    print_summary(all_results)

    # Analyze convergence
    analyze_convergence(all_results)

    print(f"\n{'=' * 70}")
    print("Benchmark complete!")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
