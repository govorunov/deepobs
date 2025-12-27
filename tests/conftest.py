"""Pytest configuration for DeepOBS tests."""

import sys
from pathlib import Path

# Add tests directory to path so test_utils can be imported
tests_dir = Path(__file__).parent
if str(tests_dir) not in sys.path:
    sys.path.insert(0, str(tests_dir))
