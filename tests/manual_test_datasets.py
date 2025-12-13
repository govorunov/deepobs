#!/usr/bin/env python3
"""Manual test script for PyTorch datasets.

This script performs basic smoke tests on the dataset implementations.
It requires PyTorch and torchvision to be installed.

Usage:
    python tests/manual_test_datasets.py
"""

import sys
import torch

# Test imports
print("Testing imports...")
try:
    from deepobs.pytorch.datasets import MNIST, FashionMNIST, CIFAR10, CIFAR100
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_mnist():
    """Test MNIST dataset."""
    print("\nTesting MNIST...")
    dataset = MNIST(batch_size=32, train_eval_size=1000)

    # Check loaders exist
    assert dataset.train_loader is not None
    assert dataset.test_loader is not None
    assert dataset.train_eval_loader is not None

    # Get a batch
    images, labels = next(iter(dataset.train_loader))

    # Check shapes
    assert images.shape == (32, 1, 28, 28), f"Expected (32, 1, 28, 28), got {images.shape}"
    assert labels.shape == (32,), f"Expected (32,), got {labels.shape}"

    # Check types
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64

    # Check data range
    assert images.min() >= 0.0 and images.max() <= 1.0

    # Check label range
    assert labels.min() >= 0 and labels.max() < 10

    print("✓ MNIST passed all checks")


def test_fmnist():
    """Test Fashion-MNIST dataset."""
    print("\nTesting Fashion-MNIST...")
    dataset = FashionMNIST(batch_size=64, train_eval_size=1000)

    # Get a batch
    images, labels = next(iter(dataset.train_loader))

    # Check shapes
    assert images.shape == (64, 1, 28, 28), f"Expected (64, 1, 28, 28), got {images.shape}"
    assert labels.shape == (64,), f"Expected (64,), got {labels.shape}"

    # Check types
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64

    # Check data range
    assert images.min() >= 0.0 and images.max() <= 1.0

    # Check label range
    assert labels.min() >= 0 and labels.max() < 10

    print("✓ Fashion-MNIST passed all checks")


def test_cifar10():
    """Test CIFAR-10 dataset."""
    print("\nTesting CIFAR-10...")
    dataset = CIFAR10(batch_size=128, data_augmentation=True, train_eval_size=5000)

    # Get a batch
    images, labels = next(iter(dataset.train_loader))

    # Check shapes
    assert images.shape == (128, 3, 32, 32), f"Expected (128, 3, 32, 32), got {images.shape}"
    assert labels.shape == (128,), f"Expected (128,), got {labels.shape}"

    # Check types
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64

    # Check label range
    assert labels.min() >= 0 and labels.max() < 10

    # Check per-image standardization
    img = images[0]
    mean = img.mean().item()
    std = img.std().item()
    print(f"  Sample image - mean: {mean:.4f}, std: {std:.4f}")
    assert abs(mean) < 0.5, "Mean should be close to 0 after standardization"

    print("✓ CIFAR-10 passed all checks")


def test_cifar100():
    """Test CIFAR-100 dataset."""
    print("\nTesting CIFAR-100...")
    dataset = CIFAR100(batch_size=128, data_augmentation=False, train_eval_size=5000)

    # Get a batch
    images, labels = next(iter(dataset.test_loader))

    # Check shapes
    assert images.shape == (128, 3, 32, 32), f"Expected (128, 3, 32, 32), got {images.shape}"
    assert labels.shape == (128,), f"Expected (128,), got {labels.shape}"

    # Check types
    assert images.dtype == torch.float32
    assert labels.dtype == torch.int64

    # Check label range
    assert labels.min() >= 0 and labels.max() < 100

    # Check per-image standardization
    img = images[0]
    mean = img.mean().item()
    std = img.std().item()
    print(f"  Sample image - mean: {mean:.4f}, std: {std:.4f}")
    assert abs(mean) < 0.5, "Mean should be close to 0 after standardization"

    print("✓ CIFAR-100 passed all checks")


def main():
    """Run all tests."""
    print("=" * 60)
    print("DeepOBS PyTorch Datasets - Manual Test Suite")
    print("=" * 60)

    try:
        test_mnist()
        test_fmnist()
        test_cifar10()
        test_cifar100()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print("\n" + "=" * 60)
        print(f"✗ Test failed: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
