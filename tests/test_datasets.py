"""Comprehensive tests for all DeepOBS PyTorch datasets."""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader

from deepobs.pytorch.datasets import (
    mnist, fmnist, cifar10, cifar100, svhn,
    imagenet, tolstoi, quadratic, two_d
)
from tests.test_utils import set_seed, assert_shape


class TestMNIST:
    """Tests for MNIST dataset."""

    def test_instantiation(self):
        """Test dataset can be instantiated."""
        dataset = mnist.mnist(batch_size=128)
        assert dataset is not None
        assert dataset.batch_size == 128

    def test_data_loading(self):
        """Test data can be loaded."""
        dataset = mnist.mnist(batch_size=32)
        train_loader = dataset.train_loader
        test_loader = dataset.test_loader

        assert train_loader is not None
        assert test_loader is not None

        # Get a batch
        x, y = next(iter(train_loader))
        assert_shape(x, (32, 1, 28, 28), "MNIST train images")
        assert_shape(y, (32,), "MNIST train labels")

    def test_batch_types(self):
        """Test batch data types."""
        dataset = mnist.mnist(batch_size=64)
        x, y = next(iter(dataset.train_loader))

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.dtype == torch.float32
        assert y.dtype == torch.long

    def test_reproducibility(self):
        """Test data loading is reproducible with fixed seed."""
        set_seed(42)
        dataset1 = mnist.mnist(batch_size=32)
        x1, y1 = next(iter(dataset1.train_loader))

        set_seed(42)
        dataset2 = mnist.mnist(batch_size=32)
        x2, y2 = next(iter(dataset2.train_loader))

        assert torch.allclose(x1, x2)
        assert torch.all(y1 == y2)


class TestFashionMNIST:
    """Tests for Fashion-MNIST dataset."""

    def test_instantiation(self):
        """Test dataset can be instantiated."""
        dataset = fmnist.fmnist(batch_size=128)
        assert dataset is not None

    def test_data_loading(self):
        """Test data can be loaded."""
        dataset = fmnist.fmnist(batch_size=32)
        x, y = next(iter(dataset.train_loader))
        assert_shape(x, (32, 1, 28, 28), "Fashion-MNIST images")
        assert_shape(y, (32,), "Fashion-MNIST labels")


class TestCIFAR10:
    """Tests for CIFAR-10 dataset."""

    def test_instantiation(self):
        """Test dataset can be instantiated."""
        dataset = cifar10.cifar10(batch_size=128)
        assert dataset is not None

    def test_data_loading(self):
        """Test data can be loaded."""
        dataset = cifar10.cifar10(batch_size=32)
        x, y = next(iter(dataset.train_loader))
        assert_shape(x, (32, 3, 32, 32), "CIFAR-10 images")
        assert_shape(y, (32,), "CIFAR-10 labels")

    def test_normalization(self):
        """Test images are normalized."""
        dataset = cifar10.cifar10(batch_size=32)
        x, y = next(iter(dataset.train_loader))

        # Check that values are roughly normalized
        mean = x.mean()
        std = x.std()

        # Should be roughly zero mean and unit variance
        assert -1.0 < mean < 1.0
        assert 0.5 < std < 2.0


class TestCIFAR100:
    """Tests for CIFAR-100 dataset."""

    def test_instantiation(self):
        """Test dataset can be instantiated."""
        dataset = cifar100.cifar100(batch_size=128)
        assert dataset is not None

    def test_data_loading(self):
        """Test data can be loaded."""
        dataset = cifar100.cifar100(batch_size=32)
        x, y = next(iter(dataset.train_loader))
        assert_shape(x, (32, 3, 32, 32), "CIFAR-100 images")
        assert_shape(y, (32,), "CIFAR-100 labels")

    def test_num_classes(self):
        """Test that labels are in correct range for 100 classes."""
        dataset = cifar100.cifar100(batch_size=128)
        x, y = next(iter(dataset.train_loader))

        assert y.min() >= 0
        assert y.max() < 100


class TestSVHN:
    """Tests for SVHN dataset."""

    def test_instantiation(self):
        """Test dataset can be instantiated."""
        dataset = svhn.svhn(batch_size=128)
        assert dataset is not None

    def test_data_loading(self):
        """Test data can be loaded."""
        dataset = svhn.svhn(batch_size=32)
        x, y = next(iter(dataset.train_loader))
        assert_shape(x, (32, 3, 32, 32), "SVHN images")
        assert_shape(y, (32,), "SVHN labels")


@pytest.mark.skip(reason="ImageNet requires manual download and setup")
class TestImageNet:
    """Tests for ImageNet dataset."""

    def test_instantiation(self):
        """Test dataset can be instantiated."""
        try:
            dataset = imagenet.imagenet(batch_size=32)
            assert dataset is not None
        except FileNotFoundError:
            pytest.skip("ImageNet data not available")

    def test_data_loading(self):
        """Test data can be loaded."""
        try:
            dataset = imagenet.imagenet(batch_size=32)
            x, y = next(iter(dataset.train_loader))
            assert_shape(x, (32, 3, 224, 224), "ImageNet images")
            assert_shape(y, (32,), "ImageNet labels")
        except FileNotFoundError:
            pytest.skip("ImageNet data not available")


@pytest.mark.skip(reason="Tolstoi requires manual download")
class TestTolstoi:
    """Tests for Tolstoi dataset."""

    def test_instantiation(self):
        """Test dataset can be instantiated."""
        try:
            dataset = tolstoi.tolstoi(batch_size=32)
            assert dataset is not None
        except FileNotFoundError:
            pytest.skip("Tolstoi data not available")

    def test_data_loading(self):
        """Test data can be loaded."""
        try:
            dataset = tolstoi.tolstoi(batch_size=32)
            x, y = next(iter(dataset.train_loader))
            assert x.dtype == torch.long  # Character indices
            assert y.dtype == torch.long
        except FileNotFoundError:
            pytest.skip("Tolstoi data not available")


class TestQuadratic:
    """Tests for Quadratic dataset."""

    def test_instantiation(self):
        """Test dataset can be instantiated."""
        dataset = quadratic.quadratic(batch_size=128)
        assert dataset is not None

    def test_data_loading(self):
        """Test data can be loaded."""
        dataset = quadratic.quadratic(batch_size=32)
        x, y = next(iter(dataset.train_loader))

        # Quadratic dataset returns data and targets
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_batch_size(self):
        """Test that batch size is respected."""
        batch_size = 64
        dataset = quadratic.quadratic(batch_size=batch_size)
        x, y = next(iter(dataset.train_loader))

        assert x.shape[0] == batch_size


class TestTwoD:
    """Tests for Two-D dataset."""

    def test_instantiation(self):
        """Test dataset can be instantiated."""
        dataset = two_d.two_d(batch_size=128)
        assert dataset is not None

    def test_data_loading(self):
        """Test data can be loaded."""
        dataset = two_d.two_d(batch_size=32)
        x, y = next(iter(dataset.train_loader))

        # Two-D dataset returns 2D points
        assert isinstance(x, torch.Tensor)
        assert x.shape[1] == 2  # 2D points

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        for batch_size in [16, 32, 64, 128]:
            dataset = two_d.two_d(batch_size=batch_size)
            x, y = next(iter(dataset.train_loader))
            assert x.shape[0] == batch_size


# Parametrized tests for all available datasets
@pytest.mark.parametrize("dataset_fn,expected_shape", [
    (mnist.mnist, (32, 1, 28, 28)),
    (fmnist.fmnist, (32, 1, 28, 28)),
    (cifar10.cifar10, (32, 3, 32, 32)),
    (cifar100.cifar100, (32, 3, 32, 32)),
    (svhn.svhn, (32, 3, 32, 32)),
])
def test_dataset_train_test_split(dataset_fn, expected_shape):
    """Test that all datasets have separate train and test loaders."""
    dataset = dataset_fn(batch_size=32)

    # Check train loader
    x_train, y_train = next(iter(dataset.train_loader))
    assert_shape(x_train, expected_shape, f"{dataset_fn.__name__} train images")

    # Check test loader
    x_test, y_test = next(iter(dataset.test_loader))
    assert_shape(x_test, expected_shape, f"{dataset_fn.__name__} test images")


@pytest.mark.parametrize("dataset_fn", [
    mnist.mnist,
    fmnist.fmnist,
    cifar10.cifar10,
    cifar100.cifar100,
    svhn.svhn,
])
def test_dataset_iteration(dataset_fn):
    """Test that datasets can be iterated multiple times."""
    dataset = dataset_fn(batch_size=32)

    # First iteration
    batches1 = []
    for i, (x, y) in enumerate(dataset.train_loader):
        batches1.append((x, y))
        if i >= 2:  # Just test a few batches
            break

    # Second iteration
    batches2 = []
    for i, (x, y) in enumerate(dataset.train_loader):
        batches2.append((x, y))
        if i >= 2:
            break

    assert len(batches1) == len(batches2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
