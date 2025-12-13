"""Test PyTorch architectures and test problems.

This test file validates the implementation of Phase 3:
- Simple architectures (LogReg, MLP, 2C2D, 3C3D)
- MNIST test problems (3)
- Fashion-MNIST test problems (3)
- CIFAR test problems (2)
"""

import unittest
import torch
import numpy as np

# Architecture modules
from deepobs.pytorch.testproblems._logreg import LogisticRegression
from deepobs.pytorch.testproblems._mlp import MLP
from deepobs.pytorch.testproblems._2c2d import TwoC2D
from deepobs.pytorch.testproblems._3c3d import ThreeC3D

# Test problems
from deepobs.pytorch.testproblems import (
    mnist_logreg,
    mnist_mlp,
    mnist_2c2d,
    fmnist_logreg,
    fmnist_mlp,
    fmnist_2c2d,
    cifar10_3c3d,
    cifar100_3c3d,
)


class TestArchitectures(unittest.TestCase):
    """Test architecture forward passes and output shapes."""

    def test_logreg_forward(self):
        """Test LogisticRegression forward pass."""
        model = LogisticRegression(num_outputs=10)
        x = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
        output = model(x)

        self.assertEqual(output.shape, (32, 10))
        self.assertTrue(torch.isfinite(output).all())

    def test_logreg_initialization(self):
        """Test LogisticRegression weight initialization (should be zero)."""
        model = LogisticRegression(num_outputs=10)

        # Weights and biases should be initialized to 0.0
        self.assertTrue(torch.allclose(model.fc.weight, torch.zeros_like(model.fc.weight)))
        self.assertTrue(torch.allclose(model.fc.bias, torch.zeros_like(model.fc.bias)))

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        model = MLP(num_outputs=10)
        x = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
        output = model(x)

        self.assertEqual(output.shape, (32, 10))
        self.assertTrue(torch.isfinite(output).all())

    def test_mlp_layer_sizes(self):
        """Test MLP has correct layer sizes."""
        model = MLP(num_outputs=10)

        self.assertEqual(model.fc1.in_features, 784)
        self.assertEqual(model.fc1.out_features, 1000)
        self.assertEqual(model.fc2.in_features, 1000)
        self.assertEqual(model.fc2.out_features, 500)
        self.assertEqual(model.fc3.in_features, 500)
        self.assertEqual(model.fc3.out_features, 100)
        self.assertEqual(model.fc4.in_features, 100)
        self.assertEqual(model.fc4.out_features, 10)

    def test_2c2d_forward(self):
        """Test TwoC2D forward pass."""
        model = TwoC2D(num_outputs=10)
        x = torch.randn(32, 1, 28, 28)  # Batch of 32 MNIST images
        output = model(x)

        self.assertEqual(output.shape, (32, 10))
        self.assertTrue(torch.isfinite(output).all())

    def test_2c2d_layer_sizes(self):
        """Test TwoC2D has correct layer sizes."""
        model = TwoC2D(num_outputs=10)

        # Conv layers
        self.assertEqual(model.conv1.in_channels, 1)
        self.assertEqual(model.conv1.out_channels, 32)
        self.assertEqual(model.conv2.in_channels, 32)
        self.assertEqual(model.conv2.out_channels, 64)

        # FC layers
        self.assertEqual(model.fc1.in_features, 7 * 7 * 64)
        self.assertEqual(model.fc1.out_features, 1024)
        self.assertEqual(model.fc2.in_features, 1024)
        self.assertEqual(model.fc2.out_features, 10)

    def test_3c3d_forward(self):
        """Test ThreeC3D forward pass."""
        model = ThreeC3D(num_outputs=10)
        x = torch.randn(32, 3, 32, 32)  # Batch of 32 CIFAR images
        output = model(x)

        self.assertEqual(output.shape, (32, 10))
        self.assertTrue(torch.isfinite(output).all())

    def test_3c3d_layer_sizes(self):
        """Test ThreeC3D has correct layer sizes."""
        model = ThreeC3D(num_outputs=100)

        # Conv layers
        self.assertEqual(model.conv1.in_channels, 3)
        self.assertEqual(model.conv1.out_channels, 64)
        self.assertEqual(model.conv2.in_channels, 64)
        self.assertEqual(model.conv2.out_channels, 96)
        self.assertEqual(model.conv3.in_channels, 96)
        self.assertEqual(model.conv3.out_channels, 128)

        # FC layers
        self.assertEqual(model.fc1.in_features, 3 * 3 * 128)
        self.assertEqual(model.fc1.out_features, 512)
        self.assertEqual(model.fc2.in_features, 512)
        self.assertEqual(model.fc2.out_features, 256)
        self.assertEqual(model.fc3.in_features, 256)
        self.assertEqual(model.fc3.out_features, 100)


class TestMNISTProblems(unittest.TestCase):
    """Test MNIST test problems."""

    def test_mnist_logreg_setup(self):
        """Test mnist_logreg test problem setup."""
        problem = mnist_logreg(batch_size=32, device='cpu')
        problem.set_up()

        # Check dataset and model are created
        self.assertIsNotNone(problem.dataset)
        self.assertIsNotNone(problem.model)
        self.assertIsInstance(problem.model, LogisticRegression)

    def test_mnist_logreg_loss_computation(self):
        """Test mnist_logreg loss computation."""
        problem = mnist_logreg(batch_size=32, device='cpu')
        problem.set_up()

        # Create dummy batch
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        batch = (x, y)

        # Compute loss and accuracy
        loss, accuracy = problem.get_batch_loss_and_accuracy(batch, reduction='mean')

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar
        self.assertTrue(torch.isfinite(loss).all())

        self.assertIsInstance(accuracy, torch.Tensor)
        self.assertTrue(0.0 <= accuracy.item() <= 1.0)

    def test_mnist_logreg_per_example_loss(self):
        """Test mnist_logreg per-example loss computation."""
        problem = mnist_logreg(batch_size=32, device='cpu')
        problem.set_up()

        # Create dummy batch
        x = torch.randn(32, 1, 28, 28)
        y = torch.randint(0, 10, (32,))
        batch = (x, y)

        # Compute per-example losses
        losses, accuracy = problem.get_batch_loss_and_accuracy(batch, reduction='none')

        self.assertEqual(losses.shape, (32,))
        self.assertTrue(torch.isfinite(losses).all())

    def test_mnist_mlp_setup(self):
        """Test mnist_mlp test problem setup."""
        problem = mnist_mlp(batch_size=32, device='cpu')
        problem.set_up()

        self.assertIsNotNone(problem.dataset)
        self.assertIsNotNone(problem.model)
        self.assertIsInstance(problem.model, MLP)

    def test_mnist_2c2d_setup(self):
        """Test mnist_2c2d test problem setup."""
        problem = mnist_2c2d(batch_size=32, device='cpu')
        problem.set_up()

        self.assertIsNotNone(problem.dataset)
        self.assertIsNotNone(problem.model)
        self.assertIsInstance(problem.model, TwoC2D)


class TestFashionMNISTProblems(unittest.TestCase):
    """Test Fashion-MNIST test problems."""

    def test_fmnist_logreg_setup(self):
        """Test fmnist_logreg test problem setup."""
        problem = fmnist_logreg(batch_size=32, device='cpu')
        problem.set_up()

        self.assertIsNotNone(problem.dataset)
        self.assertIsNotNone(problem.model)
        self.assertIsInstance(problem.model, LogisticRegression)

    def test_fmnist_mlp_setup(self):
        """Test fmnist_mlp test problem setup."""
        problem = fmnist_mlp(batch_size=32, device='cpu')
        problem.set_up()

        self.assertIsNotNone(problem.dataset)
        self.assertIsNotNone(problem.model)
        self.assertIsInstance(problem.model, MLP)

    def test_fmnist_2c2d_setup(self):
        """Test fmnist_2c2d test problem setup."""
        problem = fmnist_2c2d(batch_size=32, device='cpu')
        problem.set_up()

        self.assertIsNotNone(problem.dataset)
        self.assertIsNotNone(problem.model)
        self.assertIsInstance(problem.model, TwoC2D)


class TestCIFARProblems(unittest.TestCase):
    """Test CIFAR test problems."""

    def test_cifar10_3c3d_setup(self):
        """Test cifar10_3c3d test problem setup."""
        problem = cifar10_3c3d(batch_size=32, device='cpu')
        problem.set_up()

        self.assertIsNotNone(problem.dataset)
        self.assertIsNotNone(problem.model)
        self.assertIsInstance(problem.model, ThreeC3D)

        # Check weight decay default
        self.assertEqual(problem.weight_decay, 0.002)

    def test_cifar10_3c3d_loss_computation(self):
        """Test cifar10_3c3d loss computation."""
        problem = cifar10_3c3d(batch_size=32, device='cpu')
        problem.set_up()

        # Create dummy batch
        x = torch.randn(32, 3, 32, 32)
        y = torch.randint(0, 10, (32,))
        batch = (x, y)

        # Compute loss and accuracy
        loss, accuracy = problem.get_batch_loss_and_accuracy(batch, reduction='mean')

        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.shape, ())  # Scalar
        self.assertTrue(torch.isfinite(loss).all())

    def test_cifar100_3c3d_setup(self):
        """Test cifar100_3c3d test problem setup."""
        problem = cifar100_3c3d(batch_size=32, device='cpu')
        problem.set_up()

        self.assertIsNotNone(problem.dataset)
        self.assertIsNotNone(problem.model)
        self.assertIsInstance(problem.model, ThreeC3D)

        # Check output size is 100 for CIFAR-100
        self.assertEqual(problem.model.fc3.out_features, 100)


class TestWeightDecay(unittest.TestCase):
    """Test weight decay handling."""

    def test_no_weight_decay_warning(self):
        """Test that warning is printed when weight_decay is set for models that don't use it."""
        import io
        import sys

        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output

        # Create problem with weight_decay
        problem = mnist_mlp(batch_size=32, weight_decay=0.001, device='cpu')

        # Reset stdout
        sys.stdout = sys.__stdout__

        # Check warning was printed
        output = captured_output.getvalue()
        self.assertIn("WARNING", output)
        self.assertIn("Weight decay", output)

    def test_weight_decay_set_for_3c3d(self):
        """Test that weight_decay is properly set for 3c3d problems."""
        # Default weight decay
        problem = cifar10_3c3d(batch_size=32, device='cpu')
        self.assertEqual(problem.weight_decay, 0.002)

        # Custom weight decay
        problem = cifar10_3c3d(batch_size=32, weight_decay=0.005, device='cpu')
        self.assertEqual(problem.weight_decay, 0.005)


if __name__ == '__main__':
    unittest.main()
