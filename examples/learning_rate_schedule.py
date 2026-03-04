"""
Learning rate scheduling example for DeepOBS PyTorch.

DeepOBS has built-in LR schedule support via the YAML config:

    optimizers:
      - name: SGD
        learning_rate: 0.1
        lr_schedule:
          epochs: [30, 60, 90]
          factors: [0.1, 0.1, 0.1]

This example shows the equivalent using PyTorch's scheduler API directly,
which is useful when you need schedule types not supported by the built-in
(epochs/factors) mechanism.

Run: uv run python examples/learning_rate_schedule.py
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from deepobs.pytorch import testproblems


def train_with_schedule(tp, optimizer, scheduler, num_epochs, name):
    """Train one epoch, step the scheduler, and print per-epoch stats."""
    print(f"\n--- {name} ---")
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]["lr"]

        # Training
        tp.model.train()
        for batch in tp.dataset.train_loader:
            optimizer.zero_grad()
            loss, _ = tp.get_batch_loss_and_accuracy(batch)
            loss.backward()
            optimizer.step()

        # Evaluation
        tp.model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in tp.dataset.test_loader:
                loss, acc = tp.get_batch_loss_and_accuracy(batch)
                total_loss += loss.item()
                total_acc += acc.item()
                n += 1

        scheduler.step()

        print(f"  Epoch {epoch + 1:2d}: lr={current_lr:.5f}, "
              f"test loss={total_loss / n:.4f}, acc={total_acc / n:.4f}")

    return total_acc / n   # return final accuracy


def main():
    batch_size = 128
    num_epochs = 10
    base_lr = 0.1

    # 1. MultiStepLR — reduce LR by ×0.1 at fixed epoch milestones
    tp = testproblems.cifar10_3c3d(batch_size=batch_size)
    tp.set_up()
    optimizer = optim.SGD(tp.model.parameters(), lr=base_lr, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=[5, 8], gamma=0.1)
    acc_multi = train_with_schedule(tp, optimizer, scheduler, num_epochs,
                                    "MultiStepLR (milestones=[5,8], γ=0.1)")

    # 2. CosineAnnealingLR — smooth cosine decay to eta_min
    tp = testproblems.cifar10_3c3d(batch_size=batch_size)
    tp.set_up()
    optimizer = optim.SGD(tp.model.parameters(), lr=base_lr, momentum=0.9)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-4)
    acc_cosine = train_with_schedule(tp, optimizer, scheduler, num_epochs,
                                     "CosineAnnealingLR (T_max=10)")

    print(f"\nFinal accuracy — MultiStepLR: {acc_multi:.4f}, "
          f"CosineAnnealing: {acc_cosine:.4f}")

    print("\nNote: for benchmarking with LR schedules use the YAML config:")
    print("  lr_schedule:")
    print("    epochs: [5, 8]")
    print("    factors: [0.1, 0.01]")


if __name__ == "__main__":
    main()
