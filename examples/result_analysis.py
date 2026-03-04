"""
Result analysis example for DeepOBS PyTorch.

The built-in CLI command handles analysis and generates an interactive
HTML report automatically:

    uv run deepobs analyze ./results

Use this script for custom post-processing of the JSON result files that
the benchmark produces (e.g. additional statistics, custom plots).

Result file location:
    results/<testproblem>/<optimizer>/<hyperparams_folder>/<seed>_<timestamp>.json

Run: uv run python examples/result_analysis.py
"""

import json
import os
import glob
from collections import defaultdict


def load_results(results_dir="./results"):
    """Load all JSON result files under results_dir.

    Returns a dict keyed by (testproblem, optimizer) → result data.
    Each entry corresponds to one run file produced by StandardRunner.
    """
    # Result files are named  random_seed__<N>__<timestamp>.json
    # and live four levels deep: <dir>/<problem>/<optimizer>/<params>/<file>
    pattern = os.path.join(results_dir, "**", "*.json")
    files = glob.glob(pattern, recursive=True)
    print(f"Found {len(files)} result file(s) in '{results_dir}'")

    results = {}
    for path in files:
        parts = path.replace("\\", "/").split("/")
        # Structure: …/testproblem/optimizer/params_folder/filename.json
        if len(parts) < 4:
            continue
        testproblem = parts[-4]
        optimizer = parts[-3]
        with open(path) as f:
            data = json.load(f)
        # Keep the last run if multiple exist for the same (problem, optimizer)
        results[(testproblem, optimizer)] = data

    return results


def print_summary(results):
    """Print a table of final test loss and accuracy for each run."""
    if not results:
        print("No results to display.")
        return

    print(f"\n{'Problem':<22} {'Optimizer':<16} {'Test Loss':>10} {'Test Acc':>10}")
    print("-" * 62)

    for (problem, optimizer), data in sorted(results.items()):
        # test_losses/accuracies are lists over epochs (including epoch-0 eval)
        test_losses = data.get("test_losses", [])
        test_accs = data.get("test_accuracies", [])

        final_loss = test_losses[-1] if test_losses else float("nan")
        final_acc = test_accs[-1] if test_accs else float("nan")

        acc_str = f"{final_acc:10.4f}" if final_acc == final_acc else "       N/A"
        print(f"{problem:<22} {optimizer:<16} {final_loss:10.4f} {acc_str}")


def per_optimizer_stats(results):
    """Compute mean / std of final test accuracy per optimizer."""
    import statistics

    by_optimizer = defaultdict(list)
    for (_, optimizer), data in results.items():
        accs = data.get("test_accuracies", [])
        if accs:
            by_optimizer[optimizer].append(accs[-1])

    if not by_optimizer:
        return

    print(f"\n{'Optimizer':<20} {'#Problems':>10} {'Mean Acc':>10} {'Std Acc':>10}")
    print("-" * 54)
    for opt, accs in sorted(by_optimizer.items()):
        mean = statistics.mean(accs)
        std = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"{opt:<20} {len(accs):>10} {mean:>10.4f} {std:>10.4f}")


def main():
    results_dir = "./results"
    results = load_results(results_dir)

    if not results:
        print(f"\nNo results found in '{results_dir}'.")
        print("Run a benchmark first:")
        print("  uv run deepobs benchmark examples/benchmark_config.yaml")
        print("\nOr generate an interactive HTML report with:")
        print("  uv run deepobs analyze")
        return

    print_summary(results)
    per_optimizer_stats(results)

    print("\nFor publication-quality plots and an interactive report run:")
    print("  uv run deepobs analyze ./results")


if __name__ == "__main__":
    main()
