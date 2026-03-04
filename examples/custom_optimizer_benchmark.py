"""
Custom optimizer benchmarking example for DeepOBS PyTorch.

Two ways to benchmark a custom optimizer:

  1. YAML config + CLI (recommended for proper benchmarking):
       Add your optimizer under `optimizers:` in a YAML config file,
       specifying the import path via `optimizer_class:`, then run:
           uv run deepobs benchmark my_config.yaml
           uv run deepobs analyze

  2. StandardRunner (programmatic, for scripted sweeps):
       Shown in runner_benchmark() below.

Run: uv run python examples/custom_optimizer_benchmark.py
"""

import torch
from torch.optim import Optimizer
from deepobs.pytorch import testproblems
from deepobs.pytorch.runners import StandardRunner


class AdamWithWarmup(Optimizer):
    """Adam with a linear learning-rate warmup phase.

    A minimal custom optimizer that illustrates the DeepOBS interface.
    Any torch.optim.Optimizer subclass works with both usage paths.

    Args:
        params: Model parameters.
        lr: Peak learning rate.
        warmup_steps: Steps over which to linearly ramp up the LR.
        betas: Adam momentum coefficients.
        eps: Numerical stability term.
    """

    def __init__(self, params, lr=1e-3, warmup_steps=1000,
                 betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, warmup_steps=warmup_steps, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, warmup = group["lr"], group["warmup_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                m, v = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                m.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)

                # Bias correction + linear warmup
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                warmup_factor = min(t / warmup, 1.0)
                step_size = lr * warmup_factor * (bc2 ** 0.5) / bc1

                p.addcdiv_(m, v.sqrt().add_(group["eps"]), value=-step_size)

        return loss


# ---------------------------------------------------------------------------
# Option 1: direct training loop (quick experiments, no logging)
# ---------------------------------------------------------------------------

def direct_training():
    """Train with the custom optimizer using the programmatic API."""
    print("=== Direct training loop ===")
    tp = testproblems.mnist_mlp(batch_size=128)
    tp.set_up()

    optimizer = AdamWithWarmup(tp.model.parameters(), lr=1e-3, warmup_steps=300)

    for epoch in range(3):
        tp.model.train()
        for batch in tp.dataset.train_loader:
            optimizer.zero_grad()
            loss, _ = tp.get_batch_loss_and_accuracy(batch)
            loss.backward()
            optimizer.step()

        tp.model.eval()
        total_loss, n = 0.0, 0
        with torch.no_grad():
            for batch in tp.dataset.test_loader:
                loss, _ = tp.get_batch_loss_and_accuracy(batch)
                total_loss += loss.item()
                n += 1
        print(f"  Epoch {epoch + 1}: test loss = {total_loss / n:.4f}")


# ---------------------------------------------------------------------------
# Option 2: StandardRunner — structured JSON logging, CLI-compatible
# ---------------------------------------------------------------------------

def runner_benchmark():
    """Benchmark via StandardRunner for structured JSON output."""
    print("\n=== StandardRunner benchmark ===")

    # Declare optimizer hyperparameters (besides lr).
    # Only expose warmup_steps; betas/eps use the optimizer's defaults.
    hyperparams = [{"name": "warmup_steps", "type": int, "default": 1000}]

    runner = StandardRunner(AdamWithWarmup, hyperparams)

    # Any arg not supplied here becomes a required CLI argument.
    # Supply all args programmatically to avoid interactive prompts.
    runner.run(
        testproblem="mnist_mlp",
        batch_size=128,
        num_epochs=5,
        learning_rate=1e-3,
        warmup_steps=300,   # our custom hyperparam
        random_seed=42,
        output_dir="./results",
    )
    print("Results saved to ./results/mnist_mlp/AdamWithWarmup/")


if __name__ == "__main__":
    direct_training()
    runner_benchmark()

    print("\nFor full multi-problem benchmarking use the CLI:")
    print("  uv run deepobs benchmark examples/benchmark_config.yaml")
    print("  uv run deepobs analyze")
