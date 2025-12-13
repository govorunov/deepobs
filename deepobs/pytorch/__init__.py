"""DeepOBS PyTorch backend.

This package provides PyTorch implementations of DeepOBS datasets and test problems
for benchmarking deep learning optimizers.
"""

from . import config
from .datasets import dataset
from .testproblems import testproblem

# Expose configuration functions at package level
from .config import (
    get_data_dir,
    set_data_dir,
    get_baseline_dir,
    set_baseline_dir,
    get_dtype,
    set_dtype,
)

__version__ = '1.2.0-pytorch'

__all__ = [
    'config',
    'dataset',
    'testproblem',
    'get_data_dir',
    'set_data_dir',
    'get_baseline_dir',
    'set_baseline_dir',
    'get_dtype',
    'set_dtype',
]
