"""Fashion-MNIST dataset for DeepOBS PyTorch backend."""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset
from .dataset import DataSet
from .. import config


class FashionMNIST(DataSet):
    """DeepOBS dataset class for Fashion-MNIST.

    The Fashion-MNIST dataset consists of 60,000 training images and 10,000 test
    images of fashion items (10 categories: T-shirt/top, Trouser, Pullover, Dress,
    Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot). Images are 28x28 grayscale.

    This implementation uses torchvision.datasets.FashionMNIST for automatic
    downloading and loading. Images are normalized to [0, 1] and returned in NCHW
    format (batch, channels, height, width). Labels are returned as class indices
    (not one-hot encoded).

    Args:
        batch_size (int): The mini-batch size to use. Note that if batch_size
            is not a divisor of the dataset size (60,000 for train, 10,000 for
            test), the remainder is dropped in each epoch (after shuffling).
        train_eval_size (int, optional): Number of training examples to use for
            evaluation during training. Defaults to 10,000 (size of test set).
        num_workers (int): Number of subprocesses for data loading. Defaults to 4.

    Attributes:
        train_loader (DataLoader): DataLoader for training data (shuffled).
        train_eval_loader (DataLoader): DataLoader for training evaluation
            (not shuffled, limited size).
        test_loader (DataLoader): DataLoader for test data (not shuffled).

    References:
        https://github.com/zalandoresearch/fashion-mnist
    """

    def __init__(
        self,
        batch_size: int,
        train_eval_size: int = 10000,
        num_workers: int = 4
    ):
        """Creates a new Fashion-MNIST dataset instance.

        Args:
            batch_size (int): The mini-batch size to use.
            train_eval_size (int): Size of training evaluation set. Defaults to 10,000.
            num_workers (int): Number of worker processes for data loading.
        """
        super().__init__(batch_size, train_eval_size, num_workers)

    def _make_train_dataset(self) -> TorchDataset:
        """Creates the Fashion-MNIST training dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding training data.
        """
        data_dir = config.get_data_dir()

        # Transform: Convert to tensor and ensure shape is (1, 28, 28)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0, 1]
        ])

        dataset = torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transform
        )

        return dataset

    def _make_test_dataset(self) -> TorchDataset:
        """Creates the Fashion-MNIST test dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding test data.
        """
        data_dir = config.get_data_dir()

        # Same transform as training (no augmentation for Fashion-MNIST)
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        dataset = torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=False,
            download=True,
            transform=transform
        )

        return dataset
