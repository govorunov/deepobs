"""Simple test for runner utilities without pytest."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from deepobs.pytorch.runners import runner_utils


def test_float2str():
    """Test float to string conversion."""
    assert runner_utils.float2str(0.001) == "1e-03"
    assert runner_utils.float2str(0.3) == "3e-01"
    assert runner_utils.float2str(1.0) == "1e+00"
    print("✓ float2str tests passed")


def test_make_lr_schedule():
    """Test LR schedule creation."""
    # No schedule
    schedule = runner_utils.make_lr_schedule(0.1, None, None)
    assert schedule == {0: 0.1}

    # With schedule
    schedule = runner_utils.make_lr_schedule(0.3, [50, 100], [0.1, 0.01])
    expected = {0: 0.3, 50: 0.03, 100: 0.003}
    assert schedule == expected
    print("✓ make_lr_schedule tests passed")


def test_make_run_name():
    """Test run name generation."""
    folder, filename = runner_utils.make_run_name(
        weight_decay=0.001,
        batch_size=128,
        num_epochs=10,
        learning_rate=0.01,
        lr_sched_epochs=None,
        lr_sched_factors=None,
        random_seed=42,
        momentum=0.9
    )

    assert "num_epochs__10" in folder
    assert "batch_size__128" in folder
    assert "weight_decay__1e-03" in folder
    assert "momentum__9e-01" in folder
    assert "lr__1e-02" in folder
    assert "random_seed__42" in filename
    print("✓ make_run_name tests passed")
    print(f"  Example folder: {folder[:80]}...")
    print(f"  Example filename: {filename}")


if __name__ == "__main__":
    print("Testing runner utilities...")
    test_float2str()
    test_make_lr_schedule()
    test_make_run_name()
    print("\n✓ All utility tests passed!")
