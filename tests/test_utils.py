"""Test utilities for DeepOBS PyTorch tests."""

import torch
import numpy as np
from typing import Tuple, Optional


def get_dummy_batch(batch_size: int = 32,
                   image_shape: Tuple[int, int, int] = (1, 28, 28),
                   num_classes: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random test data batch.

    Args:
        batch_size: Number of samples in batch
        image_shape: Shape of images (C, H, W)
        num_classes: Number of classes for labels

    Returns:
        Tuple of (images, labels)
    """
    images = torch.randn(batch_size, *image_shape)
    labels = torch.randint(0, num_classes, (batch_size,))
    return images, labels


def get_dummy_sequence_batch(batch_size: int = 32,
                            sequence_length: int = 50,
                            vocab_size: int = 83) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate random sequence data batch.

    Args:
        batch_size: Number of sequences
        sequence_length: Length of each sequence
        vocab_size: Size of vocabulary

    Returns:
        Tuple of (sequences, targets)
    """
    sequences = torch.randint(0, vocab_size, (batch_size, sequence_length))
    targets = torch.randint(0, vocab_size, (batch_size, sequence_length))
    return sequences, targets


def assert_shape(tensor: torch.Tensor,
                expected_shape: Tuple[int, ...],
                name: str = "tensor"):
    """Assert that tensor has expected shape.

    Args:
        tensor: Tensor to check
        expected_shape: Expected shape tuple
        name: Name for error message
    """
    actual_shape = tuple(tensor.shape)
    assert actual_shape == expected_shape, \
        f"{name} shape mismatch: expected {expected_shape}, got {actual_shape}"


def assert_decreasing(values: list, name: str = "values"):
    """Assert that values are generally decreasing (allowing some noise).

    Args:
        values: List of values to check
        name: Name for error message
    """
    if len(values) < 2:
        return

    # Check that last value is less than first (allowing for noise)
    assert values[-1] < values[0], \
        f"{name} not decreasing: first={values[0]:.4f}, last={values[-1]:.4f}"


def assert_increasing(values: list, name: str = "values"):
    """Assert that values are generally increasing (allowing some noise).

    Args:
        values: List of values to check
        name: Name for error message
    """
    if len(values) < 2:
        return

    # Check that last value is greater than first (allowing for noise)
    assert values[-1] > values[0], \
        f"{name} not increasing: first={values[0]:.4f}, last={values[-1]:.4f}"


def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of trainable parameters in model.

    Args:
        model: PyTorch model

    Returns:
        Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def check_gpu_available() -> bool:
    """Check if GPU is available.

    Returns:
        True if CUDA is available, False otherwise
    """
    return torch.cuda.is_available()


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def test_gradient_flow(model: torch.nn.Module,
                       loss: torch.Tensor) -> bool:
    """Test that gradients flow through all parameters.

    Args:
        model: PyTorch model
        loss: Loss tensor (after backward)

    Returns:
        True if all parameters have gradients, False otherwise
    """
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"No gradient for parameter: {name}")
            return False
    return True


def compare_outputs(output1: torch.Tensor,
                   output2: torch.Tensor,
                   rtol: float = 1e-5,
                   atol: float = 1e-8) -> bool:
    """Compare two tensor outputs with tolerance.

    Args:
        output1: First tensor
        output2: Second tensor
        rtol: Relative tolerance
        atol: Absolute tolerance

    Returns:
        True if tensors are close, False otherwise
    """
    return torch.allclose(output1, output2, rtol=rtol, atol=atol)


def get_available_test_problems():
    """Get list of test problems that should be testable.

    Returns:
        List of test problem names
    """
    # MNIST-based (4)
    mnist_problems = ['mnist_logreg', 'mnist_mlp', 'mnist_2c2d', 'mnist_vae']

    # Fashion-MNIST-based (4)
    fmnist_problems = ['fmnist_logreg', 'fmnist_mlp', 'fmnist_2c2d', 'fmnist_vae']

    # CIFAR-10-based (3)
    cifar10_problems = ['cifar10_3c3d', 'cifar10_vgg16', 'cifar10_vgg19']

    # CIFAR-100-based (5)
    cifar100_problems = ['cifar100_3c3d', 'cifar100_allcnnc',
                         'cifar100_vgg16', 'cifar100_vgg19', 'cifar100_wrn404']

    # SVHN-based (2)
    svhn_problems = ['svhn_3c3d', 'svhn_wrn164']

    # ImageNet-based (3) - may not be available
    imagenet_problems = ['imagenet_vgg16', 'imagenet_vgg19', 'imagenet_inception_v3']

    # Synthetic problems (4)
    synthetic_problems = ['quadratic_deep', 'two_d_rosenbrock',
                         'two_d_beale', 'two_d_branin']

    # Problems that should work with downloaded data
    available = (mnist_problems + fmnist_problems + cifar10_problems +
                cifar100_problems + svhn_problems + synthetic_problems)

    # Problems requiring manual setup
    manual = imagenet_problems

    return available, manual
