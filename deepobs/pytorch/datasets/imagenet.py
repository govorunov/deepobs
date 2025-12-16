"""ImageNet dataset for DeepOBS PyTorch backend."""

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


class ImageNet(DataSet):
    """DeepOBS dataset class for ImageNet.

    The ImageNet dataset (ILSVRC2012) consists of approximately 1.28 million
    training images and 50,000 validation images across 1000 classes. This
    implementation uses 1001 classes (including an additional 'background' class
    at index 0), as used by Inception models.

    Images are resized to 224x224 for training and testing. Images are returned
    in NCHW format (batch, channels, height, width). Labels are returned as
    class indices (not one-hot encoded).

    Data augmentation (applied to training data only when enabled):
    - Resize to preserve aspect ratio (smaller side = 256px)
    - Pad to 256x256, then random crop to 224x224
    - Random horizontal flip
    - Per-image standardization (zero mean, unit variance)

    Note:
        ImageNet must be downloaded manually and placed in the data directory.
        The directory structure should follow the standard torchvision.datasets.ImageNet
        format with 'train' and 'val' subdirectories.

    Args:
        batch_size (int): The mini-batch size to use. Note that if batch_size
            is not a divisor of the dataset size, the remainder is dropped in
            each epoch (after shuffling).
        data_augmentation (bool): If True, applies data augmentation to training
            data. Defaults to True.
        train_eval_size (int, optional): Number of training examples to use for
            evaluation during training. Defaults to 50,000 (size of validation set).
        num_workers (int): Number of subprocesses for data loading. Defaults to 4.

    Attributes:
        train_loader (DataLoader): DataLoader for training data (shuffled).
        train_eval_loader (DataLoader): DataLoader for training evaluation
            (not shuffled, limited size).
        test_loader (DataLoader): DataLoader for test data (not shuffled).

    References:
        http://www.image-net.org/
        https://arxiv.org/abs/1409.0575 (ImageNet classification paper)
    """

    def __init__(
        self,
        batch_size: int,
        data_augmentation: bool = True,
        train_eval_size: int = 50000,
        num_workers: int = 4
    ):
        """Creates a new ImageNet dataset instance.

        Args:
            batch_size (int): The mini-batch size to use.
            data_augmentation (bool): Whether to apply data augmentation to training data.
            train_eval_size (int): Size of training evaluation set. Defaults to 50,000.
            num_workers (int): Number of worker processes for data loading.
        """
        self._data_augmentation = data_augmentation
        super().__init__(batch_size, train_eval_size, num_workers)

    def _make_train_dataset(self) -> TorchDataset:
        """Creates the ImageNet training dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding training data.
        """
        data_dir = config.get_data_dir()

        if self._data_augmentation:
            # Training transform with data augmentation
            # Matches TensorFlow implementation: resize with aspect ratio preservation,
            # then random crop and flip
            transform = transforms.Compose([
                transforms.Resize(256),  # Resize smaller side to 256
                transforms.RandomCrop(224),  # Random crop to 224x224
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Convert to tensor, scales to [0, 1]
                PerImageStandardization()  # Per-image normalization
            ])
        else:
            # Training without augmentation (just standardization)
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                PerImageStandardization()
            ])

        dataset = torchvision.datasets.ImageNet(
            root=data_dir,
            split='train',
            transform=transform
        )

        return dataset

    def _make_test_dataset(self) -> TorchDataset:
        """Creates the ImageNet test/validation dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding test data.
        """
        data_dir = config.get_data_dir()

        # Test transform (no augmentation, center crop, standardization)
        transform = transforms.Compose([
            transforms.Resize(256),  # Resize smaller side to 256
            transforms.CenterCrop(224),  # Center crop to 224x224
            transforms.ToTensor(),
            PerImageStandardization()
        ])

        dataset = torchvision.datasets.ImageNet(
            root=data_dir,
            split='val',
            transform=transform
        )

        return dataset
