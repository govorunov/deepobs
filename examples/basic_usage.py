"""
Basic usage example for DeepOBS PyTorch.

This script demonstrates:
- Creating a test problem
- Setting up a model and optimizer
- Training loop
- Evaluation
- Basic logging

Run: python basic_usage.py
"""

import torch
import torch.optim as optim
from deepobs.pytorch import testproblems


def train_one_epoch(tproblem, optimizer, device, verbose=True):
    """Train for one epoch.

    Args:
        tproblem: DeepOBS test problem instance
        optimizer: PyTorch optimizer
        device: Device to use
        verbose: Whether to print progress

    Returns:
        Average training loss and accuracy
    """
    tproblem.model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(tproblem.dataset.train_loader):
        # Zero gradients
        optimizer.zero_grad()

        # Forward pass - get per-example losses and accuracy
        losses, accuracy = tproblem.get_batch_loss_and_accuracy(batch)

        # Compute mean loss
        loss = losses.mean()

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_accuracy += accuracy if accuracy is not None else 0.0
        num_batches += 1

        # Print progress
        if verbose and batch_idx % 100 == 0:
            print(f'  Batch {batch_idx}/{len(tproblem.dataset.train_loader)}: '
                  f'Loss = {loss.item():.4f}, Accuracy = {accuracy:.4f}')

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def evaluate(tproblem, device):
    """Evaluate model on test set.

    Args:
        tproblem: DeepOBS test problem instance
        device: Device to use

    Returns:
        Average test loss and accuracy
    """
    tproblem.model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in tproblem.dataset.test_loader:
            # Forward pass
            losses, accuracy = tproblem.get_batch_loss_and_accuracy(batch)

            # Accumulate metrics
            total_loss += losses.mean().item()
            total_accuracy += accuracy if accuracy is not None else 0.0
            num_batches += 1

    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    return avg_loss, avg_accuracy


def main():
    """Main function demonstrating basic DeepOBS usage."""
    print("=" * 70)
    print("DeepOBS PyTorch - Basic Usage Example")
    print("=" * 70)

    # Configuration
    batch_size = 128
    num_epochs = 5
    learning_rate = 0.01
    momentum = 0.9

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    # Create test problem
    print("\n1. Creating test problem: mnist_mlp")
    tproblem = testproblems.mnist_mlp(batch_size=batch_size, device=device)
    tproblem.set_up()
    print(f"   Model parameters: {sum(p.numel() for p in tproblem.model.parameters()):,}")

    # Create optimizer
    print("\n2. Creating optimizer: SGD with momentum")
    optimizer = optim.SGD(
        tproblem.model.parameters(),
        lr=learning_rate,
        momentum=momentum
    )
    print(f"   Learning rate: {learning_rate}")
    print(f"   Momentum: {momentum}")

    # Training loop
    print("\n3. Training for {} epochs".format(num_epochs))
    print("-" * 70)

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Train
        train_loss, train_acc = train_one_epoch(
            tproblem, optimizer, device, verbose=(epoch == 0)
        )

        # Evaluate
        test_loss, test_acc = evaluate(tproblem, device)

        # Print epoch summary
        print(f"  Train: Loss = {train_loss:.4f}, Accuracy = {train_acc:.4f}")
        print(f"  Test:  Loss = {test_loss:.4f}, Accuracy = {test_acc:.4f}")

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70)

    # Final evaluation
    print("\nFinal Test Performance:")
    final_loss, final_acc = evaluate(tproblem, device)
    print(f"  Loss: {final_loss:.4f}")
    print(f"  Accuracy: {final_acc:.4f}")


if __name__ == '__main__':
    main()
