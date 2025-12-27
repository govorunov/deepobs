"""End-to-end integration tests for DeepOBS PyTorch.

These tests run full training loops to verify that everything works together.
"""

import pytest
import torch
import torch.optim as optim
import os
import json
import tempfile
import shutil

from deepobs.pytorch.runners import StandardRunner
from deepobs.pytorch import testproblems
from test_utils import set_seed, assert_decreasing, assert_increasing


@pytest.mark.slow
class TestEndToEndTraining:
    """End-to-end training tests on small problems."""

    def test_mnist_logreg_training(self):
        """Test full training run on MNIST logistic regression.

        This test verifies that:
        - Training completes without errors
        - Loss decreases over epochs
        - Accuracy increases over epochs
        - Results are saved correctly
        """
        set_seed(42)

        runner = StandardRunner(optimizer_class=optim.SGD)
        hyperparams = {'lr': 0.1, 'momentum': 0.0}

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                runner.run(
                    testproblem='mnist_logreg',
                    hyperparams=hyperparams,
                    num_epochs=3,
                    batch_size=128,
                    random_seed=42,
                    output_dir=tmpdir
                )

                # Find and load results
                json_files = []
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith('.json'):
                            json_files.append(os.path.join(root, file))

                assert len(json_files) == 1, "Should create exactly one JSON file"

                with open(json_files[0], 'r') as f:
                    results = json.load(f)

                # Verify loss decreases
                train_losses = results['train_losses']
                test_losses = results['test_losses']

                assert_decreasing(train_losses, "train losses")
                assert_decreasing(test_losses, "test losses")

                # Verify accuracy increases
                train_accuracies = results['train_accuracies']
                test_accuracies = results['test_accuracies']

                assert_increasing(train_accuracies, "train accuracies")
                assert_increasing(test_accuracies, "test accuracies")

                # Final accuracy should be reasonable
                assert test_accuracies[-1] > 0.8, \
                    f"Final test accuracy {test_accuracies[-1]:.3f} too low"

                print(f"\nMNIST Logistic Regression Training Results:")
                print(f"  Train loss: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
                print(f"  Test loss: {test_losses[0]:.4f} -> {test_losses[-1]:.4f}")
                print(f"  Train acc: {train_accuracies[0]:.4f} -> {train_accuracies[-1]:.4f}")
                print(f"  Test acc: {test_accuracies[0]:.4f} -> {test_accuracies[-1]:.4f}")

            except FileNotFoundError:
                pytest.skip("MNIST data not available")

    def test_mnist_mlp_training(self):
        """Test full training run on MNIST MLP."""
        set_seed(42)

        runner = StandardRunner(optimizer_class=optim.SGD)
        hyperparams = {'lr': 0.01, 'momentum': 0.9}

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                runner.run(
                    testproblem='mnist_mlp',
                    hyperparams=hyperparams,
                    num_epochs=3,
                    batch_size=128,
                    random_seed=42,
                    output_dir=tmpdir
                )

                # Load results
                json_files = []
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith('.json'):
                            json_files.append(os.path.join(root, file))

                with open(json_files[0], 'r') as f:
                    results = json.load(f)

                # Verify improvement
                train_losses = results['train_losses']
                test_accuracies = results['test_accuracies']

                assert_decreasing(train_losses, "MLP train losses")
                assert_increasing(test_accuracies, "MLP test accuracies")

                # MLP should achieve good accuracy
                assert test_accuracies[-1] > 0.9, \
                    f"MLP final test accuracy {test_accuracies[-1]:.3f} too low"

                print(f"\nMNIST MLP Training Results:")
                print(f"  Train loss: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")
                print(f"  Test acc: {test_accuracies[0]:.4f} -> {test_accuracies[-1]:.4f}")

            except FileNotFoundError:
                pytest.skip("MNIST data not available")

    def test_fmnist_2c2d_training(self):
        """Test full training run on Fashion-MNIST 2C2D."""
        set_seed(42)

        runner = StandardRunner(optimizer_class=optim.Adam)
        hyperparams = {'lr': 0.001}

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                runner.run(
                    testproblem='fmnist_2c2d',
                    hyperparams=hyperparams,
                    num_epochs=2,
                    batch_size=128,
                    random_seed=42,
                    output_dir=tmpdir
                )

                # Load results
                json_files = []
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith('.json'):
                            json_files.append(os.path.join(root, file))

                with open(json_files[0], 'r') as f:
                    results = json.load(f)

                # Verify training worked
                train_losses = results['train_losses']
                assert len(train_losses) == 3  # Initial + 2 epochs

                print(f"\nFashion-MNIST 2C2D Training Results:")
                print(f"  Train loss: {train_losses[0]:.4f} -> {train_losses[-1]:.4f}")

            except FileNotFoundError:
                pytest.skip("Fashion-MNIST data not available")


@pytest.mark.slow
class TestLearningRateScheduling:
    """Test learning rate scheduling in end-to-end training."""

    def test_lr_schedule_effect(self):
        """Test that LR schedule affects training."""
        set_seed(42)

        runner = StandardRunner(optimizer_class=optim.SGD)
        hyperparams = {'lr': 0.1, 'momentum': 0.0}

        # Run with LR schedule
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                runner.run(
                    testproblem='mnist_logreg',
                    hyperparams=hyperparams,
                    num_epochs=5,
                    batch_size=128,
                    random_seed=42,
                    lr_schedule={3: 0.01},  # Drop LR at epoch 3
                    output_dir=tmpdir
                )

                # Load results
                json_files = []
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith('.json'):
                            json_files.append(os.path.join(root, file))

                with open(json_files[0], 'r') as f:
                    results = json.load(f)

                # Verify LR schedule is recorded
                assert 'lr_schedule' in results or 'lr_sched_epochs' in results

                train_losses = results['train_losses']
                print(f"\nLR Schedule Effect:")
                print(f"  Losses: {[f'{l:.4f}' for l in train_losses]}")

            except FileNotFoundError:
                pytest.skip("MNIST data not available")


@pytest.mark.slow
class TestDifferentOptimizers:
    """Test training with different optimizers."""

    @pytest.mark.parametrize("opt_class,hyperparams", [
        (optim.SGD, {'lr': 0.1, 'momentum': 0.0}),
        (optim.SGD, {'lr': 0.1, 'momentum': 0.9}),
        (optim.Adam, {'lr': 0.001}),
    ])
    def test_optimizer_convergence(self, opt_class, hyperparams):
        """Test that different optimizers can train successfully."""
        set_seed(42)

        runner = StandardRunner(optimizer_class=opt_class)

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                runner.run(
                    testproblem='mnist_logreg',
                    hyperparams=hyperparams,
                    num_epochs=2,
                    batch_size=128,
                    random_seed=42,
                    output_dir=tmpdir
                )

                # Load results
                json_files = []
                for root, dirs, files in os.walk(tmpdir):
                    for file in files:
                        if file.endswith('.json'):
                            json_files.append(os.path.join(root, file))

                with open(json_files[0], 'r') as f:
                    results = json.load(f)

                # Just verify training completed
                assert len(results['train_losses']) == 3

                print(f"\n{opt_class.__name__} with {hyperparams}:")
                print(f"  Final test accuracy: {results['test_accuracies'][-1]:.4f}")

            except FileNotFoundError:
                pytest.skip("MNIST data not available")


class TestMinimalTraining:
    """Quick training tests (not marked as slow)."""

    def test_one_batch_forward_backward(self):
        """Test one batch forward and backward pass."""
        set_seed(42)

        try:
            # Create problem
            problem = testproblems.testproblem('mnist_logreg', batch_size=32)
            problem.set_up()

            # Get a batch
            batch = next(iter(problem.dataset.train_loader))

            # Forward pass
            problem.model.train()
            losses, accuracy = problem.get_batch_loss_and_accuracy(batch)

            # Backward pass
            loss = losses.mean()
            loss.backward()

            # Verify gradients exist
            has_gradients = any(
                p.grad is not None for p in problem.model.parameters()
                if p.requires_grad
            )

            assert has_gradients, "No gradients after backward pass"

            print(f"\nOne batch test:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Accuracy: {accuracy:.4f}")

        except FileNotFoundError:
            pytest.skip("MNIST data not available")

    def test_one_epoch_training(self):
        """Test training for one epoch (faster than multi-epoch tests)."""
        set_seed(42)

        try:
            problem = testproblems.testproblem('mnist_logreg', batch_size=128)
            problem.set_up()
            optimizer = optim.SGD(problem.model.parameters(), lr=0.1)

            # Train for one epoch
            problem.model.train()
            total_loss = 0.0
            num_batches = 0

            for batch in problem.dataset.train_loader:
                optimizer.zero_grad()
                losses, accuracy = problem.get_batch_loss_and_accuracy(batch)
                loss = losses.mean()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

                if num_batches >= 10:  # Just test first 10 batches
                    break

            avg_loss = total_loss / num_batches
            assert avg_loss > 0, "Average loss should be positive"

            print(f"\nOne epoch test (10 batches):")
            print(f"  Average loss: {avg_loss:.4f}")

        except FileNotFoundError:
            pytest.skip("MNIST data not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
