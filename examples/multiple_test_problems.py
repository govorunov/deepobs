"""
Running multiple test problems with DeepOBS PyTorch.

The recommended way to benchmark an optimizer on many problems is via
the YAML CLI, which handles logging, reproducibility, and analysis:

    uv run deepobs benchmark examples/benchmark_config_adamw_small.yaml
    uv run deepobs analyze

This script demonstrates the equivalent programmatic approach, which is
useful when you need custom metrics or training logic per problem.

Run: uv run python examples/multiple_test_problems.py
"""

import torch
import torch.optim as optim
from deepobs.pytorch import testproblems

# Problems and per-problem hyperparameters.
# For a comprehensive list of all 28 test problems see README.md.
PROBLEMS = {
    "mnist_logreg": {"lr": 0.1,  "epochs": 3},
    "mnist_mlp":    {"lr": 0.01, "epochs": 3},
    "fmnist_mlp":   {"lr": 0.01, "epochs": 3},
}


def run_one(problem_name, lr, num_epochs):
    """Train SGD on a single test problem and return final test accuracy."""
    tp_cls = getattr(testproblems, problem_name)
    tp = tp_cls(batch_size=128)
    tp.set_up()

    optimizer = optim.SGD(tp.model.parameters(), lr=lr, momentum=0.9)

    for epoch in range(num_epochs):
        tp.model.train()
        for batch in tp.dataset.train_loader:
            optimizer.zero_grad()
            loss, _ = tp.get_batch_loss_and_accuracy(batch)
            loss.backward()
            optimizer.step()

    # Final evaluation
    tp.model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch in tp.dataset.test_loader:
            loss, acc = tp.get_batch_loss_and_accuracy(batch)
            total_loss += loss.item()
            total_acc += acc.item()
            n += 1

    return total_loss / n, total_acc / n


def main():
    print(f"{'Problem':<20} {'Loss':>8} {'Accuracy':>10}")
    print("-" * 42)

    for problem, hp in PROBLEMS.items():
        final_loss, final_acc = run_one(problem, hp["lr"], hp["epochs"])
        print(f"{problem:<20} {final_loss:8.4f} {final_acc:10.4f}")

    print("\nFor a full benchmark across all 28 problems, use the CLI:")
    print("  uv run deepobs benchmark examples/benchmark_config_adamw_small.yaml")


if __name__ == "__main__":
    main()
