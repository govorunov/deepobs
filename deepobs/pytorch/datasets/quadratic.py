"""Quadratic dataset for DeepOBS PyTorch backend."""

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset, TensorDataset
from .dataset import DataSet


class Quadratic(DataSet):
    """DeepOBS dataset class to create an n-dimensional stochastic quadratic test problem.

    This synthetic dataset consists of a fixed number (train_size) of iid draws
    from a zero-mean normal distribution in 'dim' dimensions with isotropic
    covariance specified by 'noise_level'.

    This is used for testing optimizers on simple quadratic optimization problems
    where the loss function is: L(theta) = 0.5 * (theta - x)^T * Q * (theta - x),
    where Q is a condition matrix and x is sampled from this dataset.

    Args:
        batch_size (int): The mini-batch size to use. Note that if batch_size
            is not a divisor of the dataset size (train_size), the remainder
            is dropped in each epoch (after shuffling).
        dim (int): Dimensionality of the quadratic. Defaults to 100.
        train_size (int): Size of the dataset; will be used for train, train eval,
            and test datasets. Defaults to 1000.
        noise_level (float): Standard deviation of the data points around the mean.
            The data points are drawn from a Gaussian distribution. Defaults to 0.6.
        num_workers (int): Number of subprocesses for data loading. Defaults to 4.

    Attributes:
        train_loader (DataLoader): DataLoader for training data (shuffled).
        train_eval_loader (DataLoader): DataLoader for training evaluation
            (not shuffled, uses full training set).
        test_loader (DataLoader): DataLoader for test data (not shuffled).

    Returns:
        Each batch is a tensor X of shape (batch_size, dim) containing samples
        from a zero-mean Gaussian distribution.
    """

    def __init__(
        self,
        batch_size: int,
        dim: int = 100,
        train_size: int = 1000,
        noise_level: float = 0.6,
        num_workers: int = 4
    ):
        """Creates a new Quadratic dataset instance.

        Args:
            batch_size (int): The mini-batch size to use.
            dim (int): Dimensionality of the quadratic. Defaults to 100.
            train_size (int): Size of the dataset. Defaults to 1000.
            noise_level (float): Standard deviation of the Gaussian samples. Defaults to 0.6.
            num_workers (int): Number of worker processes for data loading.
        """
        self._dim = dim
        self._train_size = train_size
        self._noise_level = noise_level
        # For quadratic, train_eval uses full training set
        super().__init__(batch_size, train_eval_size=train_size, num_workers=num_workers)

    def _make_train_dataset(self) -> TorchDataset:
        """Creates the quadratic training dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding training data.
        """
        # Draw data from a random generator with a fixed seed to always get the same data
        rng = np.random.RandomState(42)
        X = rng.normal(0.0, self._noise_level, (self._train_size, self._dim))
        X = X.astype(np.float32)

        # Convert to tensor
        X_tensor = torch.from_numpy(X)

        # Create a TensorDataset (wraps tensors as a dataset)
        dataset = TensorDataset(X_tensor)

        return dataset

    def _make_test_dataset(self) -> TorchDataset:
        """Creates the quadratic test dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding test data.
        """
        # Draw data from a random generator with a different seed for test set
        rng = np.random.RandomState(43)
        X = rng.normal(0.0, self._noise_level, (self._train_size, self._dim))
        X = X.astype(np.float32)

        # Convert to tensor
        X_tensor = torch.from_numpy(X)

        # Create a TensorDataset
        dataset = TensorDataset(X_tensor)

        return dataset
