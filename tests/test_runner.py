"""Tests for PyTorch runners."""

import pytest
import torch
import torch.optim as optim
import os
import json
import tempfile
import shutil

from deepobs.pytorch.runners import StandardRunner, runner_utils


class TestRunnerUtils:
    """Tests for runner utility functions."""

    def test_float2str(self):
        """Test float to string conversion."""
        # Test basic conversion
        assert runner_utils.float2str(0.001) == "1e-03"
        assert runner_utils.float2str(0.3) == "3e-01"
        assert runner_utils.float2str(1.0) == "1e+00"
        assert runner_utils.float2str(0.0001) == "1e-04"

    def test_make_lr_schedule_no_schedule(self):
        """Test LR schedule creation without scheduling."""
        schedule = runner_utils.make_lr_schedule(0.1, None, None)
        assert schedule == {0: 0.1}

        schedule = runner_utils.make_lr_schedule(0.5, [], [])
        assert schedule == {0: 0.5}

    def test_make_lr_schedule_with_schedule(self):
        """Test LR schedule creation with scheduling."""
        schedule = runner_utils.make_lr_schedule(0.3, [50, 100], [0.1, 0.01])
        expected = {0: 0.3, 50: 0.03, 100: 0.003}
        assert schedule == expected

    def test_make_lr_schedule_errors(self):
        """Test LR schedule creation error handling."""
        # Only one of epochs/factors is None
        with pytest.raises(TypeError):
            runner_utils.make_lr_schedule(0.1, [50], None)

        with pytest.raises(TypeError):
            runner_utils.make_lr_schedule(0.1, None, [0.1])

        # Mismatched lengths
        with pytest.raises(ValueError):
            runner_utils.make_lr_schedule(0.1, [50, 100], [0.1])

    def test_make_run_name(self):
        """Test run name generation."""
        folder, filename = runner_utils.make_run_name(
            weight_decay=0.001,
            batch_size=128,
            num_epochs=10,
            learning_rate=0.01,
            lr_sched_epochs=None,
            lr_sched_factors=None,
            random_seed=42,
            momentum=0.9
        )

        # Check folder name contains expected components
        assert "num_epochs__10" in folder
        assert "batch_size__128" in folder
        assert "weight_decay__1e-03" in folder
        assert "momentum__9e-01" in folder
        assert "lr__1e-02" in folder

        # Check filename format
        assert "random_seed__42" in filename
        assert len(filename.split("__")) == 2  # seed and timestamp

    def test_make_run_name_with_schedule(self):
        """Test run name generation with LR schedule."""
        folder, filename = runner_utils.make_run_name(
            weight_decay=None,
            batch_size=64,
            num_epochs=100,
            learning_rate=0.1,
            lr_sched_epochs=[50, 75],
            lr_sched_factors=[0.1, 0.01],
            random_seed=123
        )

        # Check LR schedule in folder name
        assert "lr_schedule__0_1e-01_50_1e-02_75_1e-03" in folder


class TestStandardRunner:
    """Tests for StandardRunner."""

    def test_initialization(self):
        """Test StandardRunner initialization."""
        optimizer_class = optim.SGD
        hyperparams = [
            {"name": "momentum", "type": float, "default": 0.0},
            {"name": "nesterov", "type": bool, "default": False}
        ]

        runner = StandardRunner(optimizer_class, hyperparams)

        assert runner._optimizer_class == optim.SGD
        assert runner._optimizer_name == "SGD"
        assert runner._hyperparams == hyperparams

    @pytest.mark.slow
    def test_minimal_run(self):
        """Test a minimal training run (2 epochs on mnist_logreg).

        This is marked as slow because it actually trains a model.
        """
        # Create temporary directory for output
        temp_dir = tempfile.mkdtemp()

        try:
            # Set up runner with SGD optimizer
            optimizer_class = optim.SGD
            hyperparams = [
                {"name": "momentum", "type": float, "default": 0.0}
            ]

            runner = StandardRunner(optimizer_class, hyperparams)

            # Run training for 2 epochs
            runner._run(
                testproblem="mnist_logreg",
                weight_decay=None,
                batch_size=128,
                num_epochs=2,
                learning_rate=0.01,
                lr_sched_epochs=None,
                lr_sched_factors=None,
                random_seed=42,
                data_dir=None,
                output_dir=temp_dir,
                train_log_interval=10,
                print_train_iter=False,
                no_logs=False,
                momentum=0.0
            )

            # Find the output JSON file
            output_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.json'):
                        output_files.append(os.path.join(root, file))

            assert len(output_files) == 1, "Expected exactly one JSON output file"

            # Load and validate JSON output
            with open(output_files[0], 'r') as f:
                results = json.load(f)

            # Check required fields
            assert "train_losses" in results
            assert "test_losses" in results
            assert "minibatch_train_losses" in results
            assert "train_accuracies" in results
            assert "test_accuracies" in results

            # Check metadata
            assert results["optimizer"] == "SGD"
            assert results["testproblem"] == "mnist_logreg"
            assert results["batch_size"] == 128
            assert results["num_epochs"] == 2
            assert results["learning_rate"] == 0.01
            assert results["random_seed"] == 42

            # Check hyperparams
            assert "hyperparams" in results
            assert results["hyperparams"]["momentum"] == 0.0

            # Check that we have the right number of evaluations (num_epochs + 1)
            assert len(results["train_losses"]) == 3
            assert len(results["test_losses"]) == 3
            assert len(results["train_accuracies"]) == 3
            assert len(results["test_accuracies"]) == 3

            # Check that losses are reasonable (positive, finite)
            for loss in results["train_losses"]:
                assert loss > 0
                assert loss < float('inf')

            for loss in results["test_losses"]:
                assert loss > 0
                assert loss < float('inf')

            # Check that accuracies are in [0, 1]
            for acc in results["train_accuracies"]:
                assert 0 <= acc <= 1

            for acc in results["test_accuracies"]:
                assert 0 <= acc <= 1

            print(f"\nTraining results:")
            print(f"  Initial train loss: {results['train_losses'][0]:.4f}")
            print(f"  Final train loss: {results['train_losses'][-1]:.4f}")
            print(f"  Initial test loss: {results['test_losses'][0]:.4f}")
            print(f"  Final test loss: {results['test_losses'][-1]:.4f}")
            print(f"  Initial train acc: {results['train_accuracies'][0]:.4f}")
            print(f"  Final train acc: {results['train_accuracies'][-1]:.4f}")
            print(f"  Initial test acc: {results['test_accuracies'][0]:.4f}")
            print(f"  Final test acc: {results['test_accuracies'][-1]:.4f}")

        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)

    @pytest.mark.slow
    def test_learning_rate_schedule(self):
        """Test that learning rate scheduling works correctly."""
        temp_dir = tempfile.mkdtemp()

        try:
            optimizer_class = optim.SGD
            hyperparams = [{"name": "momentum", "type": float, "default": 0.0}]
            runner = StandardRunner(optimizer_class, hyperparams)

            # Run with LR schedule: 0.1 -> 0.01 at epoch 2
            runner._run(
                testproblem="mnist_logreg",
                weight_decay=None,
                batch_size=128,
                num_epochs=3,
                learning_rate=0.1,
                lr_sched_epochs=[2],
                lr_sched_factors=[0.1],
                random_seed=42,
                data_dir=None,
                output_dir=temp_dir,
                train_log_interval=10,
                print_train_iter=False,
                no_logs=False,
                momentum=0.0
            )

            # Find output file
            output_files = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith('.json'):
                        output_files.append(os.path.join(root, file))

            with open(output_files[0], 'r') as f:
                results = json.load(f)

            # Check that LR schedule is recorded
            assert results["lr_sched_epochs"] == [2]
            assert results["lr_sched_factors"] == [0.1]

            print("\nLearning rate schedule test passed")

        finally:
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
