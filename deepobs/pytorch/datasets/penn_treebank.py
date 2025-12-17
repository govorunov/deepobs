"""Penn Treebank dataset for DeepOBS PyTorch backend."""

import os
import torch
from torch.utils.data import Dataset as TorchDataset
from .dataset import DataSet
from .. import config


class PTBDataset(TorchDataset):
    """PyTorch dataset for character-level text sequences from Penn Treebank.

    This dataset loads character-level text from Penn Treebank and yields
    input-output pairs where the output is the input shifted by one character
    (for next-character prediction).

    Args:
        data (str): Text data as a string.
        char_to_idx (dict): Mapping from characters to indices.
        seq_length (int): Sequence length for each example.
    """

    def __init__(self, data: str, char_to_idx: dict, seq_length: int):
        """Creates a new PTBDataset instance.

        Args:
            data (str): Text data as a string.
            char_to_idx (dict): Mapping from characters to indices.
            seq_length (int): Sequence length for each example.
        """
        # Convert characters to indices
        arr = torch.tensor([char_to_idx.get(c, 0) for c in data], dtype=torch.long)

        # Calculate number of complete sequences
        num_sequences = (len(arr) - 1) // seq_length

        if num_sequences == 0:
            raise ValueError(
                "This dataset is too small to use with this sequence length."
            )

        # Trim to fit complete sequences
        total_length = num_sequences * seq_length

        # Create input and output (shifted by one)
        self.X = arr[:total_length].view(num_sequences, seq_length)
        self.Y = arr[1:total_length + 1].view(num_sequences, seq_length)

    def __len__(self):
        """Returns the number of sequences."""
        return len(self.X)

    def __getitem__(self, idx):
        """Returns a single sequence.

        Args:
            idx (int): Index of the sequence to return.

        Returns:
            tuple: (x, y) where both are tensors of shape (seq_length,).
        """
        return self.X[idx], self.Y[idx]


def _download_and_prepare_ptb(data_dir: str):
    """Download and prepare Penn Treebank dataset.

    Args:
        data_dir (str): Directory to save the data.

    Returns:
        tuple: (train_text, test_text, char_to_idx, idx_to_char, vocab_size)
    """
    import urllib.request
    import zipfile
    import tempfile

    # Create data directory
    ptb_dir = os.path.join(data_dir, "penn_treebank")
    os.makedirs(ptb_dir, exist_ok=True)

    cache_file = os.path.join(ptb_dir, "ptb_processed.pt")

    # Check if already processed
    if os.path.exists(cache_file):
        print(f"Loading cached Penn Treebank from {cache_file}")
        cached = torch.load(cache_file)
        return (
            cached['train_text'],
            cached['test_text'],
            cached['char_to_idx'],
            cached['idx_to_char'],
            cached['vocab_size']
        )

    print("Downloading and preparing Penn Treebank dataset...")

    # Download URL for Penn Treebank (char level)
    # Using the simple version from raw text
    base_url = "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/"

    train_file = os.path.join(ptb_dir, "ptb.train.txt")
    test_file = os.path.join(ptb_dir, "ptb.test.txt")

    # Download train and test files if they don't exist
    if not os.path.exists(train_file):
        print(f"Downloading training data...")
        urllib.request.urlretrieve(base_url + "ptb.train.txt", train_file)

    if not os.path.exists(test_file):
        print(f"Downloading test data...")
        urllib.request.urlretrieve(base_url + "ptb.test.txt", test_file)

    # Read the text files
    with open(train_file, 'r', encoding='utf-8') as f:
        train_text = f.read()

    with open(test_file, 'r', encoding='utf-8') as f:
        test_text = f.read()

    # Build character vocabulary
    all_chars = set(train_text + test_text)
    char_to_idx = {char: idx for idx, char in enumerate(sorted(all_chars))}
    idx_to_char = {idx: char for char, idx in char_to_idx.items()}
    vocab_size = len(char_to_idx)

    print(f"Penn Treebank prepared:")
    print(f"  Training characters: {len(train_text):,}")
    print(f"  Test characters: {len(test_text):,}")
    print(f"  Vocabulary size: {vocab_size}")

    # Cache the processed data
    torch.save({
        'train_text': train_text,
        'test_text': test_text,
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size
    }, cache_file)
    print(f"Cached processed data to {cache_file}")

    return train_text, test_text, char_to_idx, idx_to_char, vocab_size


class PennTreebank(DataSet):
    """DeepOBS dataset class for character prediction on Penn Treebank.

    This dataset contains character-level sequences from Penn Treebank for
    next-character prediction tasks. The text is automatically downloaded
    from the standard repository and processed into character indices.

    The dataset yields batches of sequences where the target is the input
    shifted by one character.

    Args:
        batch_size (int): The mini-batch size to use. Note that if batch_size
            is not a divisor of the dataset size, the remainder is dropped in
            each epoch (after shuffling).
        seq_length (int): Sequence length to be modeled in each step.
            Defaults to 50.
        train_eval_size (int, optional): Number of training examples to use for
            evaluation during training. Defaults to None (use test set size).
        num_workers (int): Number of subprocesses for data loading. Defaults to 0
            (single-process loading to avoid issues on macOS).

    Attributes:
        train_loader (DataLoader): DataLoader for training data (not shuffled).
        train_eval_loader (DataLoader): DataLoader for training evaluation
            (not shuffled, limited size).
        test_loader (DataLoader): DataLoader for test data (not shuffled).
        vocab_size (int): Size of character vocabulary.
        char_to_idx (dict): Mapping from characters to indices.
        idx_to_char (dict): Mapping from indices to characters.

    Note:
        The dataset is automatically downloaded on first use and cached for
        future runs. No manual setup required.

    Returns:
        Each batch is a tuple (x, y) where:
        - x has shape (batch_size, seq_length) with character indices
        - y has shape (batch_size, seq_length) with target character indices (x shifted by 1)
    """

    def __init__(
        self,
        batch_size: int,
        seq_length: int = 50,
        train_eval_size: int = None,
        num_workers: int = 0
    ):
        """Creates a new Penn Treebank dataset instance.

        Args:
            batch_size (int): The mini-batch size to use.
            seq_length (int): Sequence length for each example. Defaults to 50.
            train_eval_size (int): Size of training evaluation set. Defaults to None.
            num_workers (int): Number of worker processes for data loading.
        """
        self._seq_length = seq_length

        # Download and prepare the dataset
        data_dir = config.get_data_dir()
        (
            self._train_text,
            self._test_text,
            self.char_to_idx,
            self.idx_to_char,
            self.vocab_size
        ) = _download_and_prepare_ptb(data_dir)

        # Default train_eval_size to test set size
        if train_eval_size is None:
            train_eval_size = len(self._test_text)

        # Note: For PTB, we don't shuffle because the sequences are pre-batched
        # to maintain temporal structure
        super().__init__(batch_size, train_eval_size, num_workers)

    def _make_train_dataset(self) -> TorchDataset:
        """Creates the Penn Treebank training dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding training data.
        """
        dataset = PTBDataset(
            data=self._train_text,
            char_to_idx=self.char_to_idx,
            seq_length=self._seq_length
        )

        return dataset

    def _make_test_dataset(self) -> TorchDataset:
        """Creates the Penn Treebank test dataset.

        Returns:
            torch.utils.data.Dataset: A PyTorch dataset yielding test data.
        """
        dataset = PTBDataset(
            data=self._test_text,
            char_to_idx=self.char_to_idx,
            seq_length=self._seq_length
        )

        return dataset
