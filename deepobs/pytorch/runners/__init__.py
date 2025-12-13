"""Runners for executing optimizer benchmarks on DeepOBS test problems.

This module provides the StandardRunner class for training optimizers on
DeepOBS test problems with automatic logging and metric tracking.

Classes:
    StandardRunner: Main runner class for executing optimizer benchmarks.
"""

from .standard_runner import StandardRunner
from . import runner_utils

__all__ = ['StandardRunner', 'runner_utils']
