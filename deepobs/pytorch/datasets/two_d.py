"""Two-D dataset for DeepOBS PyTorch backend."""

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset, TensorDataset
from .dataset import DataSet


class TwoD(DataSet):
    """DeepOBS dataset class to create two-dimensional stochastic test problems.

    This synthetic dataset consists of a fixed number (train_size) of iid draws
    from two scalar zero-mean normal distributions with standard deviation
    specified by 'noise_level'.

    This is used for testing optimizers on 2D optimization problems like
    Rosenbrock, Beale, and Branin functions. The dataset provides noisy samples
    that can be used to create stochastic versions of these classic test functions.

    Args:
        batch_size (int): The mini-batch size to use. Note that if batch_size
            is not a divisor of the dataset size (train_size), the remainder
            is dropped in each epoch (after shuffling).
        train_size (int): Size of the training data set. This will also be used as
            the train_eval and test set size. Defaults to 10,000.
        noise_level (float): Standard deviation of the data points around the mean.
            The data points are drawn from a Gaussian distribution. Defaults to 1.0.
        num_workers (int): Number of subprocesses for data loading. Defaults to 4.

    Attributes:
        train_loader (DataLoader): DataLoader for training data (shuffled).
        train_eval_loader (DataLoader): DataLoader for training evaluation
            (not shuffled, limited size).
        test_loader (DataLoader): DataLoader for test data (not shuffled).

    Returns:
        Each batch is a tuple (x, y) of tensors with shape (batch_size,) containing
        random samples that can be used to create noisy 2D test problems.

    Note:
        For the test dataset, zeros are returned instead of random noise, allowing
        evaluation on the deterministic (non-stochastic) version of the 2D function.
    """

    def __init__(
        self,
        batch_size: int,
        train_size: int = 10000,
        noise_level: float = 1.0,
        num_workers: int = 4
    ):
        """Creates a new TwoD dataset instance.

        Args:
            batch_size (int): The mini-batch size to use.
            train_size (int): Size of the training data set. Defaults to 10,000.
            noise_level (float): Standard deviation of the Gaussian samples. Defaults to 1.0.
            num_workers (int): Number of worker processes for data loading.
        """
        self._train_size = train_size
        self._noise_level = noise_level
        super().__init__(batch_size, train_eval_size=train_size, num_workers=num_workers)

    def _make_train_dataset(self) -> TorchDataset:
        """Creates the 2D training dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding training data.
        """
        # Draw data from a random generator with a fixed seed to always get the same data
        rng = np.random.RandomState(42)
        data_x = rng.normal(0.0, self._noise_level, self._train_size)
        data_y = rng.normal(0.0, self._noise_level, self._train_size)

        # Convert to float32
        data_x = data_x.astype(np.float32)
        data_y = data_y.astype(np.float32)

        # Stack into 2D array: each row is [x, y]
        data = np.stack([data_x, data_y], axis=1)  # Shape: (train_size, 2)

        # Convert to tensor
        data_tensor = torch.from_numpy(data)

        # For 2D problems, inputs and targets are the same (the noise samples)
        # The loss is computed as: f(u, v) + u*x + v*y, where (x, y) is the data
        dataset = TensorDataset(data_tensor, data_tensor)

        return dataset

    def _make_test_dataset(self) -> TorchDataset:
        """Creates the 2D test dataset.

        For testing, we use zeros to recover the deterministic 2D function
        (without noise).

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding test data.
        """
        # Use zeros for the deterministic 2D function
        data = np.zeros((self._train_size, 2), dtype=np.float32)

        # Convert to tensor
        data_tensor = torch.from_numpy(data)

        # For 2D problems, inputs and targets are the same
        dataset = TensorDataset(data_tensor, data_tensor)

        return dataset
