"""Base class for DeepOBS datasets."""

from abc import ABC, abstractmethod
from typing import Optional
import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader


class DataSet(ABC):
    """Base class for DeepOBS datasets.

    This class provides a unified interface for loading and batching training,
    validation, and test data. Subclasses must implement the abstract methods
    to create the actual datasets.

    Args:
        batch_size (int): The mini-batch size to use.
        train_eval_size (int, optional): Number of training examples to use for
            evaluation during training. If None, uses the full training set.
        num_workers (int): Number of subprocesses to use for data loading.
            Defaults to 4.

    Attributes:
        batch_size (int): The mini-batch size.
        train_loader (DataLoader): DataLoader for training data (shuffled).
        train_eval_loader (DataLoader): DataLoader for training evaluation
            (not shuffled, limited size).
        test_loader (DataLoader): DataLoader for test data (not shuffled).
    """

    def __init__(
        self,
        batch_size: int,
        train_eval_size: Optional[int] = None,
        num_workers: int = 4
    ):
        """Creates a new DataSet instance.

        Args:
            batch_size (int): The mini-batch size to use.
            train_eval_size (int, optional): Number of training examples to use
                for evaluation. If None, uses full training set.
            num_workers (int): Number of worker processes for data loading.
        """
        self._batch_size = batch_size
        self._train_eval_size = train_eval_size
        self._num_workers = num_workers

        # Create the underlying datasets
        self._train_dataset = self._make_train_dataset()
        self._test_dataset = self._make_test_dataset()

        # Create train_eval dataset (subset of training data)
        if train_eval_size is not None:
            # Use a subset of training data for evaluation
            indices = list(range(min(train_eval_size, len(self._train_dataset))))
            self._train_eval_dataset = torch.utils.data.Subset(
                self._train_dataset, indices
            )
        else:
            # Use full training set for evaluation
            self._train_eval_dataset = self._train_dataset

        # Create data loaders
        # Pin memory for faster data transfer to GPU (CUDA only, MPS doesn't support it yet)
        use_pin_memory = torch.cuda.is_available()

        self.train_loader = DataLoader(
            self._train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,  # Drop incomplete batches
            pin_memory=use_pin_memory
        )

        self.train_eval_loader = DataLoader(
            self._train_eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=use_pin_memory
        )

        self.test_loader = DataLoader(
            self._test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            drop_last=True,
            pin_memory=use_pin_memory
        )

    @property
    def batch_size(self) -> int:
        """The batch size used by this dataset."""
        return self._batch_size

    @abstractmethod
    def _make_train_dataset(self) -> TorchDataset:
        """Creates the training dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding training data.
        """
        raise NotImplementedError(
            "'DataSet' is an abstract base class. "
            "Subclasses must implement '_make_train_dataset'."
        )

    @abstractmethod
    def _make_test_dataset(self) -> TorchDataset:
        """Creates the test dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding test data.
        """
        raise NotImplementedError(
            "'DataSet' is an abstract base class. "
            "Subclasses must implement '_make_test_dataset'."
        )
