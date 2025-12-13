"""CIFAR-10 dataset for DeepOBS PyTorch backend."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset
from .dataset import DataSet
from .. import config


class PerImageStandardization:
    """Apply per-image standardization (zero mean, unit variance).

    This transform standardizes each image individually by subtracting its mean
    and dividing by its standard deviation. This matches TensorFlow's
    tf.image.per_image_standardization behavior.
    """

    def __call__(self, tensor):
        """Standardize a single image tensor.

        Args:
            tensor (torch.Tensor): Image tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Standardized image tensor.
        """
        mean = tensor.mean()
        std = tensor.std()
        # Avoid division by zero
        std_adjusted = torch.max(std, torch.tensor(1.0 / (tensor.numel() ** 0.5)))
        return (tensor - mean) / std_adjusted


class CIFAR10(DataSet):
    """DeepOBS dataset class for CIFAR-10.

    The CIFAR-10 dataset consists of 50,000 training images and 10,000 test images
    across 10 classes (airplane, automobile, bird, cat, deer, dog, frog, horse,
    ship, truck). Images are 32x32 RGB.

    This implementation uses torchvision.datasets.CIFAR10 for automatic downloading
    and loading. Images are returned in NCHW format (batch, channels, height, width).
    Labels are returned as class indices (not one-hot encoded).

    Data augmentation (applied to training data only when enabled):
    - Pad to 36x36, then random crop back to 32x32
    - Random horizontal flip
    - Random brightness adjustment (max delta = 63/255)
    - Random saturation adjustment (range [0.5, 1.5])
    - Random contrast adjustment (range [0.2, 1.8])
    - Per-image standardization (zero mean, unit variance)

    Args:
        batch_size (int): The mini-batch size to use. Note that if batch_size
            is not a divisor of the dataset size (50,000 for train, 10,000 for
            test), the remainder is dropped in each epoch (after shuffling).
        data_augmentation (bool): If True, applies data augmentation to training
            data. Defaults to True.
        train_eval_size (int, optional): Number of training examples to use for
            evaluation during training. Defaults to 10,000 (size of test set).
        num_workers (int): Number of subprocesses for data loading. Defaults to 4.

    Attributes:
        train_loader (DataLoader): DataLoader for training data (shuffled).
        train_eval_loader (DataLoader): DataLoader for training evaluation
            (not shuffled, limited size).
        test_loader (DataLoader): DataLoader for test data (not shuffled).

    References:
        https://www.cs.toronto.edu/~kriz/cifar.html
    """

    def __init__(
        self,
        batch_size: int,
        data_augmentation: bool = True,
        train_eval_size: int = 10000,
        num_workers: int = 4
    ):
        """Creates a new CIFAR-10 dataset instance.

        Args:
            batch_size (int): The mini-batch size to use.
            data_augmentation (bool): Whether to apply data augmentation to training data.
            train_eval_size (int): Size of training evaluation set. Defaults to 10,000.
            num_workers (int): Number of worker processes for data loading.
        """
        self._data_augmentation = data_augmentation
        super().__init__(batch_size, train_eval_size, num_workers)

    def _make_train_dataset(self) -> TorchDataset:
        """Creates the CIFAR-10 training dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding training data.
        """
        data_dir = config.get_data_dir()

        if self._data_augmentation:
            # Training transform with data augmentation
            transform = transforms.Compose([
                transforms.ToTensor(),  # Convert to tensor, scales to [0, 1]
                transforms.Pad(4, padding_mode='edge'),  # Pad to 36x36
                transforms.RandomCrop(32),  # Random crop back to 32x32
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(
                    brightness=63.0/255.0,  # max_delta in TensorFlow
                    saturation=(0.5, 1.5),
                    contrast=(0.2, 1.8)
                ),
                PerImageStandardization()  # Per-image normalization
            ])
        else:
            # Training without augmentation (just standardization)
            transform = transforms.Compose([
                transforms.ToTensor(),
                PerImageStandardization()
            ])

        dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )

        return dataset

    def _make_test_dataset(self) -> TorchDataset:
        """Creates the CIFAR-10 test dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding test data.
        """
        data_dir = config.get_data_dir()

        # Test transform (no augmentation, only standardization)
        transform = transforms.Compose([
            transforms.ToTensor(),
            PerImageStandardization()
        ])

        dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )

        return dataset
