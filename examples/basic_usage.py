"""
Basic programmatic usage of DeepOBS PyTorch.

Shows how to create a test problem, attach an optimizer, and run a
training loop using the low-level API.

The recommended workflow for benchmarking is YAML-based:
    uv run deepobs benchmark examples/benchmark_config.yaml
    uv run deepobs analyze

Use this API when you need custom training logic not supported by the CLI.

Run: uv run python examples/basic_usage.py
"""

import torch
import torch.optim as optim
from deepobs.pytorch import testproblems


def main():
    # Auto-selects MPS (Apple Silicon) → CUDA → CPU
    tp = testproblems.mnist_mlp(batch_size=128)
    tp.set_up()
    print(f"Device: {tp.device}")
    print(f"Parameters: {sum(p.numel() for p in tp.model.parameters()):,}")

    optimizer = optim.SGD(tp.model.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(5):
        # --- training ---
        tp.model.train()
        for batch in tp.dataset.train_loader:
            optimizer.zero_grad()
            # get_batch_loss_and_accuracy returns (loss, accuracy).
            # Default reduction='mean' → loss is a scalar tensor already.
            loss, _ = tp.get_batch_loss_and_accuracy(batch)
            loss.backward()
            optimizer.step()

        # --- evaluation ---
        tp.model.eval()
        total_loss, total_acc, n = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in tp.dataset.test_loader:
                loss, acc = tp.get_batch_loss_and_accuracy(batch)
                total_loss += loss.item()
                total_acc += acc.item()
                n += 1

        print(f"Epoch {epoch + 1:2d}: "
              f"test loss={total_loss / n:.4f}, acc={total_acc / n:.4f}")


if __name__ == "__main__":
    main()
