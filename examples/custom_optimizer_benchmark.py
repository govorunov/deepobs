"""
Custom optimizer benchmarking example for DeepOBS PyTorch.

This script demonstrates:
- Implementing a custom optimizer
- Using StandardRunner for benchmarking
- Comparing with baseline optimizers
- Saving results

Run: python custom_optimizer_benchmark.py
"""

import torch
from torch.optim.optimizer import Optimizer
from deepobs.pytorch.runners import StandardRunner


class AdamWithWarmup(Optimizer):
    """Custom optimizer: Adam with linear learning rate warmup.

    This is a simple example showing how to implement a custom optimizer
    that can be benchmarked with DeepOBS.

    Args:
        params: Model parameters
        lr: Learning rate
        betas: Coefficients for computing running averages of gradient
            and its square (default: (0.9, 0.999))
        eps: Term added to denominator for numerical stability
            (default: 1e-8)
        weight_decay: Weight decay (L2 penalty) (default: 0)
        warmup_steps: Number of warmup steps (default: 1000)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, warmup_steps=1000):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps
        )
        super(AdamWithWarmup, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure: A closure that reevaluates the model and returns the loss.

        Returns:
            Loss value if closure is provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            warmup_steps = group['warmup_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('AdamWithWarmup does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Add weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Compute step size with warmup
                if state['step'] < warmup_steps:
                    # Linear warmup
                    warmup_factor = state['step'] / warmup_steps
                    step_size = lr * warmup_factor
                else:
                    step_size = lr

                step_size = step_size * (bias_correction2 ** 0.5) / bias_correction1

                # Update parameters
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def benchmark_custom_optimizer():
    """Benchmark the custom optimizer using StandardRunner."""
    print("=" * 70)
    print("DeepOBS PyTorch - Custom Optimizer Benchmark")
    print("=" * 70)

    # Define hyperparameters for the custom optimizer
    # These will be exposed as command-line arguments by StandardRunner
    hyperparams = [
        {"name": "betas", "type": str, "default": "0.9,0.999"},  # Will be parsed
        {"name": "eps", "type": float, "default": 1e-8},
        {"name": "warmup_steps", "type": int, "default": 1000},
    ]

    # Create StandardRunner
    runner = StandardRunner(AdamWithWarmup, hyperparams)

    print("\n1. Benchmarking AdamWithWarmup on mnist_mlp")
    print("-" * 70)

    # Run benchmark
    runner.run(
        testproblem='mnist_mlp',
        batch_size=128,
        num_epochs=10,
        learning_rate=0.001,
        warmup_steps=500,
        random_seed=42,
        output_dir='./results',
        print_train_iter=False
    )

    print("\n" + "=" * 70)
    print("Benchmark complete! Results saved to ./results/mnist_mlp/AdamWithWarmup/")
    print("=" * 70)


def compare_with_baseline():
    """Compare custom optimizer with baseline SGD and Adam."""
    print("\n" + "=" * 70)
    print("Comparing AdamWithWarmup with baseline optimizers")
    print("=" * 70)

    # Benchmark SGD
    print("\n2. Benchmarking SGD (baseline)")
    print("-" * 70)
    sgd_hyperparams = [
        {"name": "momentum", "type": float, "default": 0.9},
        {"name": "nesterov", "type": bool, "default": False},
    ]
    sgd_runner = StandardRunner(torch.optim.SGD, sgd_hyperparams)
    sgd_runner.run(
        testproblem='mnist_mlp',
        batch_size=128,
        num_epochs=10,
        learning_rate=0.01,
        momentum=0.9,
        random_seed=42,
        output_dir='./results',
        print_train_iter=False
    )

    # Benchmark Adam
    print("\n3. Benchmarking Adam (baseline)")
    print("-" * 70)
    adam_hyperparams = [
        {"name": "betas", "type": str, "default": "0.9,0.999"},
        {"name": "eps", "type": float, "default": 1e-8},
    ]
    adam_runner = StandardRunner(torch.optim.Adam, adam_hyperparams)
    adam_runner.run(
        testproblem='mnist_mlp',
        batch_size=128,
        num_epochs=10,
        learning_rate=0.001,
        random_seed=42,
        output_dir='./results',
        print_train_iter=False
    )

    print("\n" + "=" * 70)
    print("All benchmarks complete!")
    print("Results saved to ./results/mnist_mlp/")
    print("  - AdamWithWarmup/")
    print("  - SGD/")
    print("  - Adam/")
    print("=" * 70)
    print("\nUse result_analysis.py to visualize and compare results.")


def simple_training_example():
    """Simple example of using the custom optimizer without StandardRunner."""
    print("\n" + "=" * 70)
    print("Simple Training Example with AdamWithWarmup")
    print("=" * 70)

    from deepobs.pytorch import testproblems

    # Create test problem
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    tproblem = testproblems.mnist_mlp(batch_size=128, device=device)
    tproblem.set_up()

    # Create custom optimizer
    optimizer = AdamWithWarmup(
        tproblem.model.parameters(),
        lr=0.001,
        warmup_steps=500
    )

    print("\nTraining for 3 epochs...")
    for epoch in range(3):
        tproblem.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in tproblem.dataset.train_loader:
            optimizer.zero_grad()
            losses, accuracy = tproblem.get_batch_loss_and_accuracy(batch)
            loss = losses.mean()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

    print("\nTraining complete!")


def main():
    """Main function."""
    import argparse

    parser = argparse.ArgumentParser(
        description='DeepOBS custom optimizer benchmark example'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='simple',
        choices=['simple', 'benchmark', 'compare'],
        help='Execution mode: simple, benchmark, or compare'
    )

    args = parser.parse_args()

    if args.mode == 'simple':
        simple_training_example()
    elif args.mode == 'benchmark':
        benchmark_custom_optimizer()
    elif args.mode == 'compare':
        benchmark_custom_optimizer()
        compare_with_baseline()


if __name__ == '__main__':
    main()
