"""Configuration module for DeepOBS PyTorch backend.

This module provides global configuration settings for:
- Data directory location (where datasets are stored)
- Baseline directory location (where baseline results are stored)
- Default floating point precision
"""

import os
import torch
from typing import Union

# Global configuration variables
_DATA_DIR = os.path.abspath("data")
_BASELINE_DIR = os.path.abspath("baselines")
_DTYPE = torch.float32


def get_data_dir() -> str:
    """Get the current data directory path.

    Returns:
        str: Path to the data directory where datasets are stored.
    """
    return _DATA_DIR


def set_data_dir(data_dir: str) -> None:
    """Set the data directory path.

    Args:
        data_dir (str): Path to the data directory where datasets should be stored.
            Relative paths will be converted to absolute paths.
    """
    global _DATA_DIR
    _DATA_DIR = os.path.abspath(data_dir)


def get_baseline_dir() -> str:
    """Get the current baseline directory path.

    Returns:
        str: Path to the baseline directory where baseline results are stored.
    """
    return _BASELINE_DIR


def set_baseline_dir(baseline_dir: str) -> None:
    """Set the baseline directory path.

    Args:
        baseline_dir (str): Path to the baseline directory where results should be stored.
            Relative paths will be converted to absolute paths.
    """
    global _BASELINE_DIR
    _BASELINE_DIR = os.path.abspath(baseline_dir)


def get_dtype() -> torch.dtype:
    """Get the default floating point dtype.

    Returns:
        torch.dtype: The default dtype to use (torch.float32 or torch.float64).
    """
    return _DTYPE


def set_dtype(dtype: Union[torch.dtype, str]) -> None:
    """Set the default floating point dtype.

    Args:
        dtype (torch.dtype or str): The dtype to use. Can be torch.float32,
            torch.float64, or strings 'float32', 'float64'.

    Raises:
        ValueError: If dtype is not a valid floating point type.
    """
    global _DTYPE

    # Handle string inputs
    if isinstance(dtype, str):
        if dtype == 'float32':
            dtype = torch.float32
        elif dtype == 'float64':
            dtype = torch.float64
        else:
            raise ValueError(f"Invalid dtype string: {dtype}. Use 'float32' or 'float64'.")

    # Validate torch.dtype
    if dtype not in (torch.float32, torch.float64):
        raise ValueError(f"Invalid dtype: {dtype}. Use torch.float32 or torch.float64.")

    _DTYPE = dtype
