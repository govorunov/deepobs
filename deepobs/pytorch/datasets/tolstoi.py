"""Tolstoi dataset for DeepOBS PyTorch backend."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
from .dataset import DataSet
from .. import config


class TolstoiDataset(TorchDataset):
    """PyTorch dataset for character-level text sequences from War and Peace.

    This dataset loads pre-processed character sequences from numpy files
    and yields input-output pairs where the output is the input shifted by
    one character (for next-character prediction).

    Args:
        filepath (str): Path to the .npy file containing character indices.
        batch_size (int): The mini-batch size.
        seq_length (int): Sequence length for each example.
    """

    def __init__(self, filepath: str, batch_size: int, seq_length: int):
        """Creates a new TolstoiDataset instance.

        Args:
            filepath (str): Path to the .npy file containing character indices.
            batch_size (int): The mini-batch size.
            seq_length (int): Sequence length for each example.
        """
        # Load the array of character ids
        arr = np.load(filepath)

        # Determine the number of batches that can be produced
        num_batches = int(
            np.floor((np.size(arr) - 1) / (batch_size * seq_length))
        )

        if num_batches == 0:
            raise ValueError(
                "This dataset is too small to use with this batch size "
                "and sequence length."
            )

        # Create input and output, where output is the text shifted by one character
        x = arr[:num_batches * batch_size * seq_length]
        y = arr[1:num_batches * batch_size * seq_length + 1]

        # Split into batches: X[i, :] is the i-th batch
        x_batches = np.split(x.reshape(batch_size, -1), num_batches, 1)
        y_batches = np.split(y.reshape(batch_size, -1), num_batches, 1)

        # Store as numpy arrays
        self.X = np.array(x_batches, dtype=np.int64)  # Shape: (num_batches, batch_size, seq_length)
        self.Y = np.array(y_batches, dtype=np.int64)

    def __len__(self):
        """Returns the number of batches."""
        return len(self.X)

    def __getitem__(self, idx):
        """Returns a single batch.

        Args:
            idx (int): Index of the batch to return.

        Returns:
            tuple: (x, y) where both are tensors of shape (batch_size, seq_length).
        """
        # Convert to tensors
        x = torch.from_numpy(self.X[idx])
        y = torch.from_numpy(self.Y[idx])
        return x, y


class Tolstoi(DataSet):
    """DeepOBS dataset class for character prediction on War and Peace by Leo Tolstoi.

    This dataset contains character-level sequences from War and Peace for
    next-character prediction tasks. The text is pre-processed into character
    indices (vocabulary size: 83 characters).

    The dataset yields batches of sequences where the target is the input
    shifted by one character.

    Args:
        batch_size (int): The mini-batch size to use. Note that if batch_size
            is not a divisor of the dataset size, the remainder is dropped in
            each epoch (after shuffling).
        seq_length (int): Sequence length to be modeled in each step.
            Defaults to 50.
        train_eval_size (int, optional): Number of training examples to use for
            evaluation during training. Defaults to 653,237 (size of test set).
        num_workers (int): Number of subprocesses for data loading. Defaults to 4.

    Attributes:
        train_loader (DataLoader): DataLoader for training data (not shuffled).
        train_eval_loader (DataLoader): DataLoader for training evaluation
            (not shuffled, limited size).
        test_loader (DataLoader): DataLoader for test data (not shuffled).

    Note:
        The Tolstoi dataset must be downloaded using the deepobs_prepare_data.sh
        script before use. The data files (train.npy, test.npy) should be placed
        in the data directory under 'tolstoi/'.

    Returns:
        Each batch is a tuple (x, y) where:
        - x has shape (batch_size, seq_length) with character indices
        - y has shape (batch_size, seq_length) with target character indices (x shifted by 1)
    """

    def __init__(
        self,
        batch_size: int,
        seq_length: int = 50,
        train_eval_size: int = 653237,
        num_workers: int = 4
    ):
        """Creates a new Tolstoi dataset instance.

        Args:
            batch_size (int): The mini-batch size to use.
            seq_length (int): Sequence length for each example. Defaults to 50.
            train_eval_size (int): Size of training evaluation set. Defaults to 653,237.
            num_workers (int): Number of worker processes for data loading.
        """
        self._seq_length = seq_length
        # Note: For Tolstoi, we don't shuffle because the sequences are pre-batched
        # to maintain temporal structure
        super().__init__(batch_size, train_eval_size, num_workers)

    def _make_train_dataset(self) -> TorchDataset:
        """Creates the Tolstoi training dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding training data.
        """
        data_dir = config.get_data_dir()
        filepath = os.path.join(data_dir, "tolstoi", "train.npy")

        dataset = TolstoiDataset(
            filepath=filepath,
            batch_size=self._batch_size,
            seq_length=self._seq_length
        )

        return dataset

    def _make_test_dataset(self) -> TorchDataset:
        """Creates the Tolstoi test dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding test data.
        """
        data_dir = config.get_data_dir()
        filepath = os.path.join(data_dir, "tolstoi", "test.npy")

        dataset = TolstoiDataset(
            filepath=filepath,
            batch_size=self._batch_size,
            seq_length=self._seq_length
        )

        return dataset
