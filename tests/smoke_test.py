#!/usr/bin/env python
"""Quick smoke test for DeepOBS PyTorch implementation.

This script performs a rapid validation of the PyTorch implementation by:
1. Testing imports of all modules
2. Instantiating 3 simple test problems
3. Running 1 training iteration on each
4. Reporting PASS/FAIL with appropriate exit code

Usage:
    python smoke_test.py
"""

import sys
import traceback
import torch


def test_imports():
    """Test that all modules can be imported."""
    print("\n" + "="*60)
    print("SMOKE TEST: Module Imports")
    print("="*60)

    modules = [
        ('deepobs.pytorch', 'Base module'),
        ('deepobs.pytorch.config', 'Configuration'),
        ('deepobs.pytorch.datasets', 'Datasets'),
        ('deepobs.pytorch.testproblems', 'Test problems'),
        ('deepobs.pytorch.runners', 'Runners'),
    ]

    failed = []

    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"✓ {module_name:40s} ({description})")
        except Exception as e:
            print(f"✗ {module_name:40s} FAILED: {e}")
            failed.append(module_name)

    if failed:
        print(f"\n✗ Import test FAILED ({len(failed)} modules failed)")
        return False
    else:
        print(f"\n✓ Import test PASSED (all {len(modules)} modules imported)")
        return True


def test_problem_instantiation():
    """Test instantiating simple test problems."""
    print("\n" + "="*60)
    print("SMOKE TEST: Problem Instantiation")
    print("="*60)

    from deepobs.pytorch import testproblems

    problems = [
        ('mnist_logreg', 'MNIST Logistic Regression'),
        ('mnist_mlp', 'MNIST Multi-Layer Perceptron'),
        ('fmnist_2c2d', 'Fashion-MNIST 2C2D'),
    ]

    failed = []

    for problem_name, description in problems:
        try:
            # Get the test problem class by name from the testproblems module
            problem_class = getattr(testproblems, problem_name)
            problem = problem_class(batch_size=32)
            print(f"✓ {problem_name:20s} ({description})")
        except FileNotFoundError as e:
            print(f"⊘ {problem_name:20s} SKIPPED: Data not available")
        except Exception as e:
            print(f"✗ {problem_name:20s} FAILED: {e}")
            failed.append(problem_name)

    if failed:
        print(f"\n✗ Instantiation test FAILED ({len(failed)} problems failed)")
        return False
    else:
        print(f"\n✓ Instantiation test PASSED")
        return True


def test_training_iteration():
    """Test one training iteration on simple problems."""
    print("\n" + "="*60)
    print("SMOKE TEST: Training Iteration")
    print("="*60)

    from deepobs.pytorch import testproblems
    import torch.optim as optim

    problems = [
        'mnist_logreg',
        'mnist_mlp',
    ]

    failed = []
    skipped = []

    for problem_name in problems:
        try:
            # Set seed for reproducibility
            torch.manual_seed(42)

            # Create problem
            problem_class = getattr(testproblems, problem_name)
            problem = problem_class(batch_size=32)
            problem.set_up()

            # Create optimizer
            optimizer = optim.SGD(problem.model.parameters(), lr=0.1)

            # Get a batch
            batch = next(iter(problem.dataset.train_loader))

            # Training step
            problem.model.train()
            optimizer.zero_grad()

            losses, accuracy = problem.get_batch_loss_and_accuracy(batch)
            loss = losses.mean()

            # Add regularization if available
            if hasattr(problem, 'get_regularization_loss'):
                reg_loss = problem.get_regularization_loss()
                if reg_loss is not None and reg_loss > 0:
                    loss = loss + reg_loss

            loss.backward()
            optimizer.step()

            # Verify results
            assert torch.isfinite(loss), "Loss is not finite"
            assert loss.item() > 0, "Loss should be positive"

            has_gradients = any(
                p.grad is not None for p in problem.model.parameters()
                if p.requires_grad
            )
            assert has_gradients, "No gradients computed"

            print(f"✓ {problem_name:20s} loss={loss.item():.4f}, acc={accuracy:.4f}")

        except FileNotFoundError:
            print(f"⊘ {problem_name:20s} SKIPPED: Data not available")
            skipped.append(problem_name)
        except Exception as e:
            print(f"✗ {problem_name:20s} FAILED: {e}")
            traceback.print_exc()
            failed.append(problem_name)

    if failed:
        print(f"\n✗ Training test FAILED ({len(failed)} problems failed)")
        return False
    elif skipped and not any([p not in skipped for p in problems]):
        print(f"\n⊘ Training test SKIPPED (all problems skipped - data not available)")
        return None  # Neither pass nor fail
    else:
        print(f"\n✓ Training test PASSED")
        return True


def main():
    """Run all smoke tests."""
    print("\n" + "="*60)
    print("DeepOBS PyTorch - Smoke Test")
    print("="*60)
    print("\nThis script performs quick validation of the PyTorch implementation.")
    print("For comprehensive testing, run: pytest tests/")

    results = []

    # Test imports
    results.append(('imports', test_imports()))

    # Test problem instantiation
    results.append(('instantiation', test_problem_instantiation()))

    # Test training iteration
    results.append(('training', test_training_iteration()))

    # Summary
    print("\n" + "="*60)
    print("SMOKE TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result is True)
    failed = sum(1 for _, result in results if result is False)
    skipped = sum(1 for _, result in results if result is None)

    for test_name, result in results:
        if result is True:
            status = "✓ PASSED"
        elif result is False:
            status = "✗ FAILED"
        else:
            status = "⊘ SKIPPED"
        print(f"{status:12s} {test_name}")

    print(f"\nTotal: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\n✗ SMOKE TEST FAILED")
        print("\nSome tests failed. Please check the output above for details.")
        print("This may indicate:")
        print("  - Missing dependencies")
        print("  - Implementation errors")
        print("  - Data not downloaded")
        return 1
    elif passed == 0:
        print("\n⊘ SMOKE TEST INCONCLUSIVE")
        print("\nAll tests were skipped. This likely means:")
        print("  - Data not downloaded (run: python -m deepobs.pytorch.datasets.download)")
        return 2
    else:
        print("\n✓ SMOKE TEST PASSED")
        print("\nAll basic functionality is working!")
        print("\nNext steps:")
        print("  - Run full test suite: pytest tests/")
        print("  - Run specific tests: pytest tests/test_datasets.py")
        print("  - Run integration tests: pytest tests/integration/")
        return 0


if __name__ == "__main__":
    sys.exit(main())
