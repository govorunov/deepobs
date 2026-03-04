"""Food101 dataset for DeepOBS PyTorch backend."""

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


class Food101(DataSet):
    """DeepOBS dataset class for Food-101.

    The Food-101 dataset consists of 101 food categories, with 101,000 images total.
    For each class, 750 training images and 250 test images are provided.
    Training set: 75,750 images
    Test set: 25,250 images

    This implementation uses torchvision.datasets.Food101 for automatic downloading
    and loading. Images are RGB with varying sizes, resized to ``target_size`` for
    model input. Images are returned in NCHW format (batch, channels, height, width).
    Labels are returned as class indices (not one-hot encoded).

    Data augmentation (applied to training data only when enabled):
    - Resize shorter side to ``int(target_size * 256 / 224)`` pixels (preserving aspect ratio)
    - Random crop to ``target_size × target_size``
    - Random horizontal flip
    - Per-image standardization (zero mean, unit variance)

    Test data preprocessing:
    - Resize shorter side proportionally
    - Center crop to ``target_size × target_size``
    - Per-image standardization

    Args:
        batch_size (int): The mini-batch size to use. Note that if batch_size
            is not a divisor of the dataset size (75,750 for train, 25,250 for
            test), the remainder is dropped in each epoch (after shuffling).
        data_augmentation (bool): If True, applies data augmentation to training
            data. Defaults to True.
        train_eval_size (int, optional): Number of training examples to use for
            evaluation during training. Defaults to 25,250 (size of test set).
        num_workers (int): Number of subprocesses for data loading. Defaults to 4.
        target_size (int): Target image side length after cropping. Defaults to
            224 (suitable for VGG-style networks). Use 96 for ViT-Micro.

    Attributes:
        train_loader (DataLoader): DataLoader for training data (shuffled).
        train_eval_loader (DataLoader): DataLoader for training evaluation
            (not shuffled, limited size).
        test_loader (DataLoader): DataLoader for test data (not shuffled).

    References:
        https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/
        "Food-101 -- Mining Discriminative Components with Random Forests"
        Bossard, Guillaumin, Van Gool - ECCV 2014
    """

    def __init__(
        self,
        batch_size: int,
        data_augmentation: bool = True,
        train_eval_size: int = 25250,
        num_workers: int = 4,
        target_size: int = 224,
    ):
        """Creates a new Food-101 dataset instance.

        Args:
            batch_size (int): The mini-batch size to use.
            data_augmentation (bool): Whether to apply data augmentation to training data.
            train_eval_size (int): Size of training evaluation set. Defaults to 25,250.
            num_workers (int): Number of worker processes for data loading.
            target_size (int): Target image side length (square crop). Defaults to 224.
        """
        self._data_augmentation = data_augmentation
        self._target_size = target_size
        super().__init__(batch_size, train_eval_size, num_workers)

    def _make_train_dataset(self) -> TorchDataset:
        """Creates the Food-101 training dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding training data.
        """
        data_dir = config.get_data_dir()
        # Scale resize proportionally to the target crop size (same ratio as 256→224)
        resize_size = int(self._target_size * 256 / 224)

        if self._data_augmentation:
            # Training transform with data augmentation (ImageNet-style)
            transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.RandomCrop(self._target_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Convert to tensor, scales to [0, 1]
                PerImageStandardization()  # Per-image normalization
            ])
        else:
            # Training without augmentation
            transform = transforms.Compose([
                transforms.Resize(resize_size),
                transforms.CenterCrop(self._target_size),
                transforms.ToTensor(),
                PerImageStandardization()
            ])

        dataset = torchvision.datasets.Food101(
            root=data_dir,
            split='train',
            download=True,
            transform=transform
        )

        return dataset

    def _make_test_dataset(self) -> TorchDataset:
        """Creates the Food-101 test dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding test data.
        """
        data_dir = config.get_data_dir()
        # Scale resize proportionally to the target crop size (same ratio as 256→224)
        resize_size = int(self._target_size * 256 / 224)

        # Test transform (no augmentation, center crop)
        transform = transforms.Compose([
            transforms.Resize(resize_size),
            transforms.CenterCrop(self._target_size),
            transforms.ToTensor(),
            PerImageStandardization()
        ])

        dataset = torchvision.datasets.Food101(
            root=data_dir,
            split='test',
            download=True,
            transform=transform
        )

        return dataset
