"""
Learning rate scheduling example for DeepOBS PyTorch.

This script demonstrates:
- MultiStepLR scheduling
- CosineAnnealingLR scheduling
- Custom learning rate schedules
- Comparing different schedules

Run: python learning_rate_schedule.py
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import (
    MultiStepLR,
    CosineAnnealingLR,
    ExponentialLR,
    ReduceLROnPlateau,
    LambdaLR
)
from deepobs.pytorch import testproblems
import matplotlib.pyplot as plt
import os


def train_with_schedule(tproblem, optimizer, scheduler, num_epochs, schedule_name):
    """Train with a learning rate schedule.

    Args:
        tproblem: DeepOBS test problem
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        schedule_name: Name of the schedule (for logging)

    Returns:
        Dictionary with training history
    """
    print(f"\n{'=' * 70}")
    print(f"Training with {schedule_name}")
    print(f"{'=' * 70}")

    train_losses = []
    test_losses = []
    test_accuracies = []
    learning_rates = []

    for epoch in range(num_epochs):
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)

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

        # Step scheduler (for ReduceLROnPlateau, pass test loss)
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_test_loss)
        else:
            scheduler.step()

        print(f"Epoch {epoch + 1:2d}/{num_epochs}: "
              f"LR = {current_lr:.6f}, "
              f"Train Loss = {avg_train_loss:.4f}, "
              f"Test Loss = {avg_test_loss:.4f}, "
              f"Test Acc = {avg_test_accuracy:.4f}")

    return {
        'schedule_name': schedule_name,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies,
        'learning_rates': learning_rates,
        'final_accuracy': test_accuracies[-1],
    }


def plot_results(results_list, output_dir='./results/plots'):
    """Plot learning curves for different schedules.

    Args:
        results_list: List of result dictionaries
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot learning rates
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    for results in results_list:
        plt.plot(results['learning_rates'], label=results['schedule_name'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedules')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot test loss
    plt.subplot(1, 3, 2)
    for results in results_list:
        plt.plot(results['test_losses'], label=results['schedule_name'])
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss')
    plt.title('Test Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot test accuracy
    plt.subplot(1, 3, 3)
    for results in results_list:
        plt.plot(results['test_accuracies'], label=results['schedule_name'])
    plt.xlabel('Epoch')
    plt.ylabel('Test Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'lr_schedule_comparison.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_file}")
    plt.close()


def main():
    """Main function demonstrating different LR schedules."""
    print("=" * 70)
    print("DeepOBS PyTorch - Learning Rate Scheduling Example")
    print("=" * 70)

    # Configuration
    batch_size = 128
    num_epochs = 30
    base_lr = 0.1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\nUsing device: {device}")
    print(f"Test problem: cifar10_3c3d")
    print(f"Base learning rate: {base_lr}")
    print(f"Number of epochs: {num_epochs}")

    results_list = []

    # 1. MultiStepLR: Reduce LR at specific epochs
    print("\n" + "=" * 70)
    print("1. MultiStepLR Schedule")
    print("=" * 70)
    tproblem = testproblems.cifar10_3c3d(batch_size=batch_size, device=device)
    tproblem.set_up()

    optimizer = optim.SGD(tproblem.model.parameters(), lr=base_lr, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)

    results = train_with_schedule(
        tproblem, optimizer, scheduler, num_epochs,
        schedule_name='MultiStepLR (epochs 10, 20, γ=0.1)'
    )
    results_list.append(results)

    # 2. CosineAnnealingLR: Smooth cosine decay
    print("\n" + "=" * 70)
    print("2. CosineAnnealingLR Schedule")
    print("=" * 70)
    tproblem = testproblems.cifar10_3c3d(batch_size=batch_size, device=device)
    tproblem.set_up()

    optimizer = optim.SGD(tproblem.model.parameters(), lr=base_lr, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)

    results = train_with_schedule(
        tproblem, optimizer, scheduler, num_epochs,
        schedule_name='CosineAnnealingLR (T_max=30)'
    )
    results_list.append(results)

    # 3. ExponentialLR: Exponential decay
    print("\n" + "=" * 70)
    print("3. ExponentialLR Schedule")
    print("=" * 70)
    tproblem = testproblems.cifar10_3c3d(batch_size=batch_size, device=device)
    tproblem.set_up()

    optimizer = optim.SGD(tproblem.model.parameters(), lr=base_lr, momentum=0.9)
    scheduler = ExponentialLR(optimizer, gamma=0.95)

    results = train_with_schedule(
        tproblem, optimizer, scheduler, num_epochs,
        schedule_name='ExponentialLR (γ=0.95)'
    )
    results_list.append(results)

    # 4. Custom schedule using LambdaLR
    print("\n" + "=" * 70)
    print("4. Custom Schedule (Linear Warmup + Decay)")
    print("=" * 70)
    tproblem = testproblems.cifar10_3c3d(batch_size=batch_size, device=device)
    tproblem.set_up()

    optimizer = optim.SGD(tproblem.model.parameters(), lr=base_lr, momentum=0.9)

    # Custom schedule: linear warmup for 5 epochs, then cosine decay
    def lr_lambda(epoch):
        warmup_epochs = 5
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine decay after warmup
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    results = train_with_schedule(
        tproblem, optimizer, scheduler, num_epochs,
        schedule_name='Custom (Warmup 5 + Cosine)'
    )
    results_list.append(results)

    # 5. Constant LR (baseline)
    print("\n" + "=" * 70)
    print("5. Constant LR (Baseline)")
    print("=" * 70)
    tproblem = testproblems.cifar10_3c3d(batch_size=batch_size, device=device)
    tproblem.set_up()

    optimizer = optim.SGD(tproblem.model.parameters(), lr=0.01, momentum=0.9)
    # Use a dummy scheduler that doesn't change LR
    scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)

    results = train_with_schedule(
        tproblem, optimizer, scheduler, num_epochs,
        schedule_name='Constant LR (0.01)'
    )
    results_list.append(results)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY: Final Test Accuracies")
    print("=" * 70)
    print(f"{'Schedule':<40} {'Final Accuracy':<15}")
    print("-" * 70)

    for results in results_list:
        print(f"{results['schedule_name']:<40} {results['final_accuracy']:<15.4f}")

    print("-" * 70)

    # Find best schedule
    best_results = max(results_list, key=lambda x: x['final_accuracy'])
    print(f"\nBest schedule: {best_results['schedule_name']}")
    print(f"Final accuracy: {best_results['final_accuracy']:.4f}")

    # Plot results
    try:
        plot_results(results_list)
        print("\n" + "=" * 70)
        print("Plots saved successfully!")
        print("=" * 70)
    except ImportError:
        print("\nNote: matplotlib not available, skipping plots")


def show_available_schedulers():
    """Print information about available PyTorch schedulers."""
    print("\n" + "=" * 70)
    print("Available PyTorch Learning Rate Schedulers")
    print("=" * 70)

    schedulers = [
        ("StepLR", "Decay LR by gamma every step_size epochs"),
        ("MultiStepLR", "Decay LR by gamma at specified milestones"),
        ("ExponentialLR", "Decay LR by gamma every epoch"),
        ("CosineAnnealingLR", "Cosine annealing schedule"),
        ("ReduceLROnPlateau", "Reduce LR when metric plateaus"),
        ("CyclicLR", "Cycle LR between two boundaries"),
        ("OneCycleLR", "One cycle LR policy (warmup + decay)"),
        ("LambdaLR", "Custom schedule using lambda function"),
    ]

    for name, description in schedulers:
        print(f"  {name:<25} {description}")

    print("\nSee PyTorch documentation for more details:")
    print("https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Learning rate scheduling example')
    parser.add_argument(
        '--show-schedulers',
        action='store_true',
        help='Show available schedulers and exit'
    )

    args = parser.parse_args()

    if args.show_schedulers:
        show_available_schedulers()
    else:
        main()
