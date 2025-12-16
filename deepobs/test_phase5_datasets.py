#!/usr/bin/env python3
"""Quick test script to verify Phase 5 datasets can be instantiated.

This script tests that all 5 new datasets from Phase 5 can be created
without errors (assuming required data is available).

Note: Some datasets require manual data download:
- ImageNet: Must be manually downloaded
- Tolstoi: Requires .npy files from deepobs_prepare_data.sh

For datasets that auto-download (SVHN), this will work out of the box.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_dataset_instantiation(dataset_class, dataset_name, **kwargs):
    """Test that a dataset can be instantiated.

    Args:
        dataset_class: The dataset class to test
        dataset_name: Name of the dataset (for printing)
        **kwargs: Additional arguments to pass to dataset constructor
    """
    try:
        print(f"Testing {dataset_name}...", end=" ")
        dataset = dataset_class(**kwargs)
        print(f"✅ SUCCESS - Train loader: {len(dataset.train_loader)} batches, "
              f"Test loader: {len(dataset.test_loader)} batches")
        return True
    except FileNotFoundError as e:
        print(f"⚠️  SKIP - Data not available: {e}")
        return None
    except Exception as e:
        print(f"❌ FAILED - {type(e).__name__}: {e}")
        return False

def main():
    """Test all Phase 5 datasets."""
    print("=" * 80)
    print("Phase 5 Dataset Instantiation Tests")
    print("=" * 80)
    print()

    # Import datasets
    try:
        from deepobs.pytorch.datasets import SVHN, ImageNet, Tolstoi, Quadratic, TwoD
        print("✅ All dataset imports successful")
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("Note: PyTorch may not be installed in this environment")
        return

    print()
    print("-" * 80)
    print("Testing Dataset Instantiation")
    print("-" * 80)
    print()

    results = {}

    # Test SVHN (should auto-download)
    results['SVHN'] = test_dataset_instantiation(
        SVHN, "SVHN",
        batch_size=128,
        data_augmentation=True
    )

    # Test ImageNet (requires manual download)
    results['ImageNet'] = test_dataset_instantiation(
        ImageNet, "ImageNet",
        batch_size=64,
        data_augmentation=True
    )

    # Test Tolstoi (requires .npy files)
    results['Tolstoi'] = test_dataset_instantiation(
        Tolstoi, "Tolstoi",
        batch_size=64,
        seq_length=50
    )

    # Test Quadratic (synthetic, should always work)
    results['Quadratic'] = test_dataset_instantiation(
        Quadratic, "Quadratic",
        batch_size=32,
        dim=100,
        train_size=1000
    )

    # Test Two-D (synthetic, should always work)
    results['TwoD'] = test_dataset_instantiation(
        TwoD, "Two-D",
        batch_size=64,
        train_size=10000
    )

    print()
    print("-" * 80)
    print("Summary")
    print("-" * 80)
    print()

    success_count = sum(1 for v in results.values() if v is True)
    skip_count = sum(1 for v in results.values() if v is None)
    fail_count = sum(1 for v in results.values() if v is False)

    print(f"✅ Success: {success_count}/5")
    print(f"⚠️  Skipped (data not available): {skip_count}/5")
    print(f"❌ Failed: {fail_count}/5")
    print()

    if fail_count > 0:
        print("Some tests failed. Please check the error messages above.")
        sys.exit(1)
    elif success_count == 0:
        print("Note: All tests were skipped due to missing data.")
        print("This is expected if PyTorch is not installed or data is not downloaded.")
    else:
        print("All available datasets instantiated successfully!")

    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
