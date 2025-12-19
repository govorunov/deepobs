"""Quick smoke test for all 26 test problems.

This test instantiates each problem and runs one forward + backward pass
to verify basic functionality.
"""

import pytest
import torch

from deepobs.pytorch import testproblems
from tests.test_utils import get_available_test_problems, set_seed


# Get all available test problems
AVAILABLE_PROBLEMS, MANUAL_PROBLEMS = get_available_test_problems()


class TestAllProblems:
    """Quick smoke test for all test problems."""

    @pytest.mark.parametrize("problem_name", AVAILABLE_PROBLEMS)
    def test_problem_smoke(self, problem_name):
        """Run quick smoke test on each problem.

        This test:
        1. Instantiates the problem
        2. Gets one batch
        3. Runs forward pass
        4. Runs backward pass
        5. Verifies gradients exist
        """
        set_seed(42)

        try:
            # Instantiate problem
            problem = testproblems.testproblem(problem_name, batch_size=32)

            # Get a batch
            batch = next(iter(problem.train_loader))
            x, y = batch

            # Verify batch shapes are reasonable
            assert x.shape[0] == 32, f"{problem_name}: wrong batch size"
            assert isinstance(x, torch.Tensor), f"{problem_name}: x not a tensor"
            assert isinstance(y, torch.Tensor), f"{problem_name}: y not a tensor"

            # Forward pass
            problem.model.train()
            losses, accuracy = problem.get_batch_loss_and_accuracy(batch)

            # Verify losses
            assert losses.shape[0] == 32, f"{problem_name}: wrong loss shape"
            assert torch.all(torch.isfinite(losses)), f"{problem_name}: non-finite losses"

            # Backward pass
            loss = losses.mean()

            # Add regularization if available
            if hasattr(problem, 'get_regularization_loss'):
                reg_loss = problem.get_regularization_loss()
                if reg_loss is not None and reg_loss > 0:
                    loss = loss + reg_loss

            loss.backward()

            # Verify gradients
            has_gradients = False
            for param in problem.model.parameters():
                if param.requires_grad:
                    if param.grad is not None:
                        has_gradients = True
                        # Check gradient is finite
                        assert torch.all(torch.isfinite(param.grad)), \
                            f"{problem_name}: non-finite gradients"

            assert has_gradients, f"{problem_name}: no gradients found"

            print(f"✓ {problem_name:30s} loss={loss.item():.4f}")

        except FileNotFoundError as e:
            pytest.skip(f"Data not available for {problem_name}: {e}")
        except Exception as e:
            pytest.fail(f"{problem_name} failed: {e}")


class TestManualProblems:
    """Tests for problems requiring manual data setup."""

    @pytest.mark.skip(reason="Requires manual data setup")
    @pytest.mark.parametrize("problem_name", MANUAL_PROBLEMS)
    def test_manual_problem_smoke(self, problem_name):
        """Smoke test for manually-setup problems."""
        set_seed(42)

        try:
            problem = testproblems.testproblem(problem_name, batch_size=32)
            batch = next(iter(problem.train_loader))

            problem.model.train()
            losses, accuracy = problem.get_batch_loss_and_accuracy(batch)
            loss = losses.mean()
            loss.backward()

            print(f"✓ {problem_name:30s} loss={loss.item():.4f}")

        except FileNotFoundError:
            pytest.skip(f"Data not available for {problem_name}")


class TestProblemSummary:
    """Generate summary of all test problems."""

    def test_generate_problem_summary(self):
        """Generate summary of all test problems (counts and status)."""
        total_problems = len(AVAILABLE_PROBLEMS) + len(MANUAL_PROBLEMS)

        print(f"\n{'='*60}")
        print(f"DeepOBS PyTorch Test Problems Summary")
        print(f"{'='*60}")
        print(f"Total test problems: {total_problems}")
        print(f"  Available (auto-download): {len(AVAILABLE_PROBLEMS)}")
        print(f"  Manual setup required: {len(MANUAL_PROBLEMS)}")
        print(f"\nAvailable Problems:")

        # Group by dataset
        problem_groups = {
            'MNIST': [p for p in AVAILABLE_PROBLEMS if p.startswith('mnist_')],
            'Fashion-MNIST': [p for p in AVAILABLE_PROBLEMS if p.startswith('fmnist_')],
            'CIFAR-10': [p for p in AVAILABLE_PROBLEMS if p.startswith('cifar10_')],
            'CIFAR-100': [p for p in AVAILABLE_PROBLEMS if p.startswith('cifar100_')],
            'SVHN': [p for p in AVAILABLE_PROBLEMS if p.startswith('svhn_')],
            'Synthetic': [p for p in AVAILABLE_PROBLEMS if p.startswith(('quadratic_', 'two_d_'))],
        }

        for dataset, problems in problem_groups.items():
            if problems:
                print(f"\n  {dataset} ({len(problems)}):")
                for p in sorted(problems):
                    print(f"    - {p}")

        if MANUAL_PROBLEMS:
            print(f"\nManual Setup Required:")
            for p in sorted(MANUAL_PROBLEMS):
                print(f"    - {p}")

        print(f"\n{'='*60}")


class TestProblemCoverage:
    """Verify test problem coverage."""

    def test_mnist_coverage(self):
        """Verify all MNIST problems are present."""
        mnist_problems = [p for p in AVAILABLE_PROBLEMS if p.startswith('mnist_')]
        expected = {'mnist_logreg', 'mnist_mlp', 'mnist_2c2d', 'mnist_vae'}

        assert set(mnist_problems) == expected, \
            f"Missing MNIST problems: {expected - set(mnist_problems)}"

    def test_fmnist_coverage(self):
        """Verify all Fashion-MNIST problems are present."""
        fmnist_problems = [p for p in AVAILABLE_PROBLEMS if p.startswith('fmnist_')]
        expected = {'fmnist_logreg', 'fmnist_mlp', 'fmnist_2c2d', 'fmnist_vae'}

        assert set(fmnist_problems) == expected, \
            f"Missing Fashion-MNIST problems: {expected - set(fmnist_problems)}"

    def test_cifar10_coverage(self):
        """Verify all CIFAR-10 problems are present."""
        cifar10_problems = [p for p in AVAILABLE_PROBLEMS if p.startswith('cifar10_')]
        expected = {'cifar10_3c3d', 'cifar10_vgg16', 'cifar10_vgg19'}

        assert set(cifar10_problems) == expected, \
            f"Missing CIFAR-10 problems: {expected - set(cifar10_problems)}"

    def test_cifar100_coverage(self):
        """Verify all CIFAR-100 problems are present."""
        cifar100_problems = [p for p in AVAILABLE_PROBLEMS if p.startswith('cifar100_')]
        expected = {'cifar100_3c3d', 'cifar100_allcnnc', 'cifar100_vgg16',
                   'cifar100_vgg19', 'cifar100_wrn404'}

        assert set(cifar100_problems) == expected, \
            f"Missing CIFAR-100 problems: {expected - set(cifar100_problems)}"

    def test_svhn_coverage(self):
        """Verify all SVHN problems are present."""
        svhn_problems = [p for p in AVAILABLE_PROBLEMS if p.startswith('svhn_')]
        expected = {'svhn_3c3d', 'svhn_wrn164'}

        assert set(svhn_problems) == expected, \
            f"Missing SVHN problems: {expected - set(svhn_problems)}"

    def test_synthetic_coverage(self):
        """Verify all synthetic problems are present."""
        synthetic_problems = [p for p in AVAILABLE_PROBLEMS
                             if p.startswith(('quadratic_', 'two_d_'))]
        expected = {'quadratic_deep', 'two_d_rosenbrock',
                   'two_d_beale', 'two_d_branin'}

        assert set(synthetic_problems) == expected, \
            f"Missing synthetic problems: {expected - set(synthetic_problems)}"

    def test_total_problem_count(self):
        """Verify total number of test problems."""
        total = len(AVAILABLE_PROBLEMS) + len(MANUAL_PROBLEMS)

        # Expected: 4 (MNIST) + 4 (Fashion-MNIST) + 3 (CIFAR-10) + 5 (CIFAR-100)
        #         + 2 (SVHN) + 3 (ImageNet) + 4 (Synthetic) = 25
        assert total == 25, f"Expected 25 test problems, found {total}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
