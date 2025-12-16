"""Tests for DeepOBS PyTorch configuration system."""

import pytest
import torch
import os
import tempfile
import shutil

from deepobs.pytorch import config


class TestConfig:
    """Tests for configuration functions."""

    def test_default_data_dir(self):
        """Test that default data directory exists."""
        data_dir = config.get_data_dir()
        assert data_dir is not None
        assert isinstance(data_dir, str)

    def test_set_data_dir(self):
        """Test setting custom data directory."""
        original_dir = config.get_data_dir()

        # Create temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            # Set new data directory
            config.set_data_dir(tmpdir)

            # Check it was set
            assert config.get_data_dir() == tmpdir

            # Restore original
            config.set_data_dir(original_dir)
            assert config.get_data_dir() == original_dir

    def test_default_baseline_dir(self):
        """Test that default baseline directory exists."""
        baseline_dir = config.get_baseline_dir()
        assert baseline_dir is not None
        assert isinstance(baseline_dir, str)

    def test_set_baseline_dir(self):
        """Test setting custom baseline directory."""
        original_dir = config.get_baseline_dir()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.set_baseline_dir(tmpdir)
            assert config.get_baseline_dir() == tmpdir

            config.set_baseline_dir(original_dir)
            assert config.get_baseline_dir() == original_dir

    def test_default_dtype(self):
        """Test default dtype is float32."""
        dtype = config.get_dtype()
        assert dtype == torch.float32

    def test_set_dtype_float32(self):
        """Test setting dtype to float32."""
        config.set_dtype(torch.float32)
        assert config.get_dtype() == torch.float32

    def test_set_dtype_float64(self):
        """Test setting dtype to float64."""
        original_dtype = config.get_dtype()

        config.set_dtype(torch.float64)
        assert config.get_dtype() == torch.float64

        # Restore original
        config.set_dtype(original_dtype)

    def test_invalid_dtype(self):
        """Test that invalid dtype raises error."""
        with pytest.raises((ValueError, TypeError)):
            config.set_dtype("invalid")

    def test_config_persistence(self):
        """Test that config changes persist within session."""
        original_data_dir = config.get_data_dir()

        with tempfile.TemporaryDirectory() as tmpdir:
            config.set_data_dir(tmpdir)

            # Check multiple times
            assert config.get_data_dir() == tmpdir
            assert config.get_data_dir() == tmpdir

            # Restore
            config.set_data_dir(original_data_dir)


class TestConfigPaths:
    """Tests for path handling in config."""

    def test_data_dir_is_absolute(self):
        """Test that data directory is absolute path."""
        data_dir = config.get_data_dir()
        assert os.path.isabs(data_dir)

    def test_baseline_dir_is_absolute(self):
        """Test that baseline directory is absolute path."""
        baseline_dir = config.get_baseline_dir()
        assert os.path.isabs(baseline_dir)

    def test_set_relative_path(self):
        """Test setting relative path (should be converted to absolute)."""
        original_dir = config.get_data_dir()

        # Set relative path
        config.set_data_dir("./test_data")

        # Should be converted to absolute
        new_dir = config.get_data_dir()
        assert os.path.isabs(new_dir)

        # Restore
        config.set_data_dir(original_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
