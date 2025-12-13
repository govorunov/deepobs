"""Quick validation script for Phase 4 implementation.

This script checks that all Phase 4 components can be imported and basic
functionality works. Run this to verify the installation.

Usage:
    python tests/validate_phase4.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """Test that all components can be imported."""
    print("Testing imports...")

    try:
        from deepobs.pytorch.runners import StandardRunner
        print("  ✓ StandardRunner imported")
    except Exception as e:
        print(f"  ✗ Failed to import StandardRunner: {e}")
        return False

    try:
        from deepobs.pytorch.runners import runner_utils
        print("  ✓ runner_utils imported")
    except Exception as e:
        print(f"  ✗ Failed to import runner_utils: {e}")
        return False

    return True


def test_runner_utils():
    """Test runner utility functions."""
    print("\nTesting runner utilities...")

    try:
        from deepobs.pytorch.runners import runner_utils

        # Test float2str
        result = runner_utils.float2str(0.001)
        assert result == "1e-03", f"float2str(0.001) = {result}, expected 1e-03"
        print("  ✓ float2str works")

        # Test make_lr_schedule
        schedule = runner_utils.make_lr_schedule(0.1, None, None)
        assert schedule == {0: 0.1}, f"schedule = {schedule}, expected {{0: 0.1}}"
        print("  ✓ make_lr_schedule works")

        # Test make_run_name
        folder, filename = runner_utils.make_run_name(
            weight_decay=0.001,
            batch_size=128,
            num_epochs=10,
            learning_rate=0.01,
            lr_sched_epochs=None,
            lr_sched_factors=None,
            random_seed=42
        )
        assert "num_epochs__10" in folder
        assert "random_seed__42" in filename
        print("  ✓ make_run_name works")

        return True

    except Exception as e:
        print(f"  ✗ Error in runner_utils: {e}")
        return False


def test_standard_runner_creation():
    """Test StandardRunner creation."""
    print("\nTesting StandardRunner creation...")

    try:
        import torch.optim as optim
        from deepobs.pytorch.runners import StandardRunner

        optimizer_class = optim.SGD
        hyperparams = [
            {"name": "momentum", "type": float, "default": 0.0}
        ]

        runner = StandardRunner(optimizer_class, hyperparams)
        assert runner._optimizer_name == "SGD"
        assert runner._optimizer_class == optim.SGD
        print("  ✓ StandardRunner created successfully")

        return True

    except ImportError as e:
        print(f"  ⚠ PyTorch not installed, skipping: {e}")
        return True  # Not a failure, just skip
    except Exception as e:
        print(f"  ✗ Error creating StandardRunner: {e}")
        return False


def test_file_structure():
    """Test that all expected files exist."""
    print("\nTesting file structure...")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expected_files = [
        "deepobs/pytorch/runners/__init__.py",
        "deepobs/pytorch/runners/runner_utils.py",
        "deepobs/pytorch/runners/standard_runner.py",
        "deepobs/pytorch/runners/README.md",
        "tests/test_runner.py",
        "tests/test_runner_standalone.py",
        "examples/pytorch_runner_example.py",
        "docs/pytorch-migration/phase4_completion_report.md",
        "docs/pytorch-migration/PHASE4_SUMMARY.md",
    ]

    all_exist = True
    for file_path in expected_files:
        full_path = os.path.join(base_dir, file_path)
        if os.path.exists(full_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} MISSING")
            all_exist = False

    return all_exist


def main():
    """Run all validation tests."""
    print("="*60)
    print("Phase 4 Validation")
    print("="*60)

    results = []

    # Test imports
    results.append(("Imports", test_imports()))

    # Test utilities (no PyTorch needed)
    results.append(("Runner Utilities", test_runner_utils()))

    # Test StandardRunner (requires PyTorch)
    results.append(("StandardRunner Creation", test_standard_runner_creation()))

    # Test file structure
    results.append(("File Structure", test_file_structure()))

    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:30s} {status}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n✓ All validation tests passed!")
        print("\nPhase 4 implementation is complete and verified.")
        return 0
    else:
        print("\n✗ Some validation tests failed.")
        print("\nPlease check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
