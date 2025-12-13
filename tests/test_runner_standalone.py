"""Standalone test for runner utilities (no dependencies needed)."""

import time


def float2str(x):
    """Convert a float to a compact string representation."""
    s = "{:.10e}".format(x)
    mantissa, exponent = s.split("e")
    return mantissa.rstrip("0").rstrip(".") + "e" + exponent


def make_lr_schedule(lr_base, lr_sched_epochs, lr_sched_factors):
    """Creates a learning rate schedule in the form of a dictionary."""
    if lr_sched_epochs is None and lr_sched_factors is None:
        return {0: lr_base}

    if lr_sched_epochs is None or lr_sched_factors is None:
        raise TypeError(
            """Specify *both* lr_sched_epochs and lr_sched_factors.""")
    if ((not isinstance(lr_sched_epochs, list))
            or (not isinstance(lr_sched_factors, list))
            or (len(lr_sched_epochs) != len(lr_sched_factors))):
        raise ValueError(
            """lr_sched_epochs and lr_sched_factors must be lists of
                     the same length.""")

    sched = {n: f * lr_base for n, f in zip(lr_sched_epochs, lr_sched_factors)}
    sched[0] = lr_base
    return sched


def make_run_name(weight_decay, batch_size, num_epochs, learning_rate,
                  lr_sched_epochs, lr_sched_factors, random_seed,
                  **optimizer_hyperparams):
    """Creates a name for the output file of an optimizer run."""
    run_folder_name = "num_epochs__" + str(
        num_epochs) + "__batch_size__" + str(batch_size) + "__"
    if weight_decay is not None:
        run_folder_name += "weight_decay__{0:s}__".format(
            float2str(weight_decay))

    for hp_name, hp_value in sorted(optimizer_hyperparams.items()):
        run_folder_name += "{0:s}__".format(hp_name)
        run_folder_name += "{0:s}__".format(
            float2str(hp_value) if isinstance(hp_value, float
                                              ) else str(hp_value))
    if lr_sched_epochs is None:
        run_folder_name += "lr__{0:s}".format(float2str(learning_rate))
    else:
        run_folder_name += ("lr_schedule__{0:d}_{1:s}".format(
            0, float2str(learning_rate)))
        for epoch, factor in zip(lr_sched_epochs, lr_sched_factors):
            run_folder_name += ("_{0:d}_{1:s}".format(
                epoch, float2str(factor * learning_rate)))
    file_name = "random_seed__{0:d}__".format(random_seed)
    file_name += time.strftime("%Y-%m-%d-%H-%M-%S")
    return run_folder_name, file_name


def test_float2str():
    """Test float to string conversion."""
    tests = [
        (0.001, "1e-03"),
        (0.3, "3e-01"),
        (1.0, "1e+00"),
        (0.0001, "1e-04"),
        (10.0, "1e+01"),
    ]
    for value, expected in tests:
        result = float2str(value)
        assert result == expected, f"float2str({value}) = {result}, expected {expected}"
    print("✓ float2str: All tests passed")


def test_make_lr_schedule():
    """Test LR schedule creation."""
    # Test 1: No schedule
    schedule = make_lr_schedule(0.1, None, None)
    assert schedule == {0: 0.1}
    print("  ✓ No schedule: {0: 0.1}")

    # Test 2: Empty schedule
    schedule = make_lr_schedule(0.5, [], [])
    assert schedule == {0: 0.5}
    print("  ✓ Empty schedule: {0: 0.5}")

    # Test 3: With schedule
    schedule = make_lr_schedule(0.3, [50, 100], [0.1, 0.01])
    expected = {0: 0.3, 50: 0.03, 100: 0.003}
    assert schedule == expected
    print(f"  ✓ With schedule: {expected}")

    # Test 4: Error handling - only epochs
    try:
        make_lr_schedule(0.1, [50], None)
        assert False, "Should have raised TypeError"
    except TypeError:
        print("  ✓ TypeError raised for mismatched None")

    # Test 5: Error handling - mismatched lengths
    try:
        make_lr_schedule(0.1, [50, 100], [0.1])
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  ✓ ValueError raised for mismatched lengths")

    print("✓ make_lr_schedule: All tests passed")


def test_make_run_name():
    """Test run name generation."""
    # Test 1: Basic run name
    folder, filename = make_run_name(
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
    print(f"  ✓ Basic folder: {folder[:60]}...")
    print(f"  ✓ Basic filename: {filename}")

    # Test 2: With LR schedule
    folder, filename = make_run_name(
        weight_decay=None,
        batch_size=64,
        num_epochs=100,
        learning_rate=0.1,
        lr_sched_epochs=[50, 75],
        lr_sched_factors=[0.1, 0.01],
        random_seed=123
    )

    assert "lr_schedule__0_1e-01_50_1e-02_75_1e-03" in folder
    assert "random_seed__123" in filename
    print(f"  ✓ LR schedule folder: {folder[:60]}...")
    print(f"  ✓ LR schedule filename: {filename}")

    # Test 3: No weight decay
    folder, filename = make_run_name(
        weight_decay=None,
        batch_size=32,
        num_epochs=5,
        learning_rate=0.001,
        lr_sched_epochs=None,
        lr_sched_factors=None,
        random_seed=1
    )

    assert "weight_decay" not in folder
    print(f"  ✓ No weight decay: {folder[:60]}...")

    print("✓ make_run_name: All tests passed")


if __name__ == "__main__":
    print("="*60)
    print("Testing Runner Utilities (Standalone)")
    print("="*60)
    print()
    test_float2str()
    print()
    test_make_lr_schedule()
    print()
    test_make_run_name()
    print()
    print("="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
