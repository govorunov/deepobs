"""Tests for DeepOBS PyTorch datasets.

This module tests the PyTorch dataset implementations including:
- MNIST
- Fashion-MNIST
- CIFAR-10
- CIFAR-100

Each test verifies:
- Dataset loading and initialization
- Batch shapes and types
- Data ranges and normalization
- Label formats (class indices, not one-hot)
- Train/test/eval loader creation
"""

import pytest
import torch
from deepobs.pytorch.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
from deepobs.pytorch import config


class TestMNIST:
    """Tests for MNIST dataset."""

    def test_init(self):
        """Test MNIST dataset initialization."""
        dataset = MNIST(batch_size=32)
        assert dataset.batch_size == 32
        assert dataset.train_loader is not None
        assert dataset.test_loader is not None
        assert dataset.train_eval_loader is not None

    def test_batch_shapes(self):
        """Test MNIST batch shapes."""
        dataset = MNIST(batch_size=32)

        # Get a batch from train loader
        images, labels = next(iter(dataset.train_loader))

        # Check shapes: (batch_size, channels, height, width)
        assert images.shape == (32, 1, 28, 28)
        assert labels.shape == (32,)  # Class indices, not one-hot

        # Check types
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64

    def test_data_range(self):
        """Test MNIST data is normalized to [0, 1]."""
        dataset = MNIST(batch_size=32)
        images, _ = next(iter(dataset.train_loader))

        # MNIST should be in [0, 1] range
        assert images.min() >= 0.0
        assert images.max() <= 1.0

    def test_label_range(self):
        """Test MNIST labels are class indices in [0, 9]."""
        dataset = MNIST(batch_size=32)
        _, labels = next(iter(dataset.train_loader))

        assert labels.min() >= 0
        assert labels.max() < 10

    def test_train_eval_size(self):
        """Test custom train_eval_size."""
        dataset = MNIST(batch_size=32, train_eval_size=1000)

        # Count batches in train_eval_loader
        num_batches = len(dataset.train_eval_loader)
        expected_batches = 1000 // 32

        assert num_batches == expected_batches


class TestFashionMNIST:
    """Tests for Fashion-MNIST dataset."""

    def test_init(self):
        """Test Fashion-MNIST dataset initialization."""
        dataset = FashionMNIST(batch_size=64)
        assert dataset.batch_size == 64
        assert dataset.train_loader is not None
        assert dataset.test_loader is not None
        assert dataset.train_eval_loader is not None

    def test_batch_shapes(self):
        """Test Fashion-MNIST batch shapes."""
        dataset = FashionMNIST(batch_size=64)

        # Get a batch from train loader
        images, labels = next(iter(dataset.train_loader))

        # Check shapes: (batch_size, channels, height, width)
        assert images.shape == (64, 1, 28, 28)
        assert labels.shape == (64,)  # Class indices, not one-hot

        # Check types
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64

    def test_data_range(self):
        """Test Fashion-MNIST data is normalized to [0, 1]."""
        dataset = FashionMNIST(batch_size=64)
        images, _ = next(iter(dataset.train_loader))

        # Fashion-MNIST should be in [0, 1] range
        assert images.min() >= 0.0
        assert images.max() <= 1.0

    def test_label_range(self):
        """Test Fashion-MNIST labels are class indices in [0, 9]."""
        dataset = FashionMNIST(batch_size=64)
        _, labels = next(iter(dataset.train_loader))

        assert labels.min() >= 0
        assert labels.max() < 10


class TestCIFAR10:
    """Tests for CIFAR-10 dataset."""

    def test_init_with_augmentation(self):
        """Test CIFAR-10 dataset initialization with augmentation."""
        dataset = CIFAR10(batch_size=128, data_augmentation=True)
        assert dataset.batch_size == 128
        assert dataset.train_loader is not None
        assert dataset.test_loader is not None
        assert dataset.train_eval_loader is not None

    def test_init_without_augmentation(self):
        """Test CIFAR-10 dataset initialization without augmentation."""
        dataset = CIFAR10(batch_size=128, data_augmentation=False)
        assert dataset.batch_size == 128

    def test_batch_shapes(self):
        """Test CIFAR-10 batch shapes."""
        dataset = CIFAR10(batch_size=128)

        # Get a batch from train loader
        images, labels = next(iter(dataset.train_loader))

        # Check shapes: (batch_size, channels, height, width)
        assert images.shape == (128, 3, 32, 32)
        assert labels.shape == (128,)  # Class indices, not one-hot

        # Check types
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64

    def test_per_image_standardization(self):
        """Test CIFAR-10 uses per-image standardization."""
        dataset = CIFAR10(batch_size=128, data_augmentation=False)
        images, _ = next(iter(dataset.test_loader))

        # Each image should have approximately zero mean and unit variance
        # Check first image in batch
        img = images[0]
        mean = img.mean()
        std = img.std()

        # Mean should be close to 0 (within tolerance)
        assert abs(mean.item()) < 0.1

        # Std should be close to 1 (within tolerance)
        assert abs(std.item() - 1.0) < 0.3

    def test_label_range(self):
        """Test CIFAR-10 labels are class indices in [0, 9]."""
        dataset = CIFAR10(batch_size=128)
        _, labels = next(iter(dataset.train_loader))

        assert labels.min() >= 0
        assert labels.max() < 10

    def test_train_eval_size(self):
        """Test custom train_eval_size."""
        dataset = CIFAR10(batch_size=128, train_eval_size=5000)

        # Count batches in train_eval_loader
        num_batches = len(dataset.train_eval_loader)
        expected_batches = 5000 // 128

        assert num_batches == expected_batches


class TestCIFAR100:
    """Tests for CIFAR-100 dataset."""

    def test_init_with_augmentation(self):
        """Test CIFAR-100 dataset initialization with augmentation."""
        dataset = CIFAR100(batch_size=128, data_augmentation=True)
        assert dataset.batch_size == 128
        assert dataset.train_loader is not None
        assert dataset.test_loader is not None
        assert dataset.train_eval_loader is not None

    def test_init_without_augmentation(self):
        """Test CIFAR-100 dataset initialization without augmentation."""
        dataset = CIFAR100(batch_size=128, data_augmentation=False)
        assert dataset.batch_size == 128

    def test_batch_shapes(self):
        """Test CIFAR-100 batch shapes."""
        dataset = CIFAR100(batch_size=128)

        # Get a batch from train loader
        images, labels = next(iter(dataset.train_loader))

        # Check shapes: (batch_size, channels, height, width)
        assert images.shape == (128, 3, 32, 32)
        assert labels.shape == (128,)  # Class indices, not one-hot

        # Check types
        assert images.dtype == torch.float32
        assert labels.dtype == torch.int64

    def test_per_image_standardization(self):
        """Test CIFAR-100 uses per-image standardization."""
        dataset = CIFAR100(batch_size=128, data_augmentation=False)
        images, _ = next(iter(dataset.test_loader))

        # Each image should have approximately zero mean and unit variance
        # Check first image in batch
        img = images[0]
        mean = img.mean()
        std = img.std()

        # Mean should be close to 0 (within tolerance)
        assert abs(mean.item()) < 0.1

        # Std should be close to 1 (within tolerance)
        assert abs(std.item() - 1.0) < 0.3

    def test_label_range(self):
        """Test CIFAR-100 labels are class indices in [0, 99]."""
        dataset = CIFAR100(batch_size=128)
        _, labels = next(iter(dataset.train_loader))

        assert labels.min() >= 0
        assert labels.max() < 100

    def test_train_eval_size(self):
        """Test custom train_eval_size."""
        dataset = CIFAR100(batch_size=128, train_eval_size=5000)

        # Count batches in train_eval_loader
        num_batches = len(dataset.train_eval_loader)
        expected_batches = 5000 // 128

        assert num_batches == expected_batches


class TestDatasetComparison:
    """Tests comparing different datasets."""

    def test_mnist_vs_fmnist_shapes(self):
        """Test MNIST and Fashion-MNIST have same shapes."""
        mnist = MNIST(batch_size=32)
        fmnist = FashionMNIST(batch_size=32)

        mnist_images, mnist_labels = next(iter(mnist.train_loader))
        fmnist_images, fmnist_labels = next(iter(fmnist.train_loader))

        assert mnist_images.shape == fmnist_images.shape
        assert mnist_labels.shape == fmnist_labels.shape

    def test_cifar10_vs_cifar100_shapes(self):
        """Test CIFAR-10 and CIFAR-100 have same image shapes."""
        cifar10 = CIFAR10(batch_size=64)
        cifar100 = CIFAR100(batch_size=64)

        c10_images, c10_labels = next(iter(cifar10.train_loader))
        c100_images, c100_labels = next(iter(cifar100.train_loader))

        # Images should have same shape
        assert c10_images.shape == c100_images.shape

        # Labels should have same shape (both are class indices)
        assert c10_labels.shape == c100_labels.shape


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
