#!/usr/bin/env python
"""
DeepOBS Command Line Interface

Main entry point for the DeepOBS benchmark suite.
Provides commands for running benchmarks and analyzing results.
"""

import sys
import argparse


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="deepobs",
        description="DeepOBS - Deep Learning Optimizer Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with default config
  deepobs benchmark

  # Run benchmark with custom config
  deepobs benchmark my_config.yaml

  # Run only specific test problems
  deepobs benchmark my_config.yaml --problems mnist_mlp cifar10_3c3d

  # Run only specific optimizers
  deepobs benchmark my_config.yaml --optimizers SGD Adam

  # Run specific problems with specific optimizers
  deepobs benchmark my_config.yaml --problems mnist_logreg --optimizers SGD

  # Analyze results
  deepobs analyze

  # Analyze results from custom directory
  deepobs analyze ./my_results

  # Analyze only specific test problems
  deepobs analyze --problems mnist_mlp cifar10_3c3d

  # Analyze only specific optimizers
  deepobs analyze --optimizers SGD Adam

  # Analyze specific problems and optimizers
  deepobs analyze --problems mnist_logreg --optimizers SGD Adam
        """,
    )

    parser.add_argument("--version", action="version", version="DeepOBS 1.2.0")

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        help="Command to run",
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run optimizer benchmark suite",
        description="Run a comprehensive benchmark of optimizers on test problems",
    )
    benchmark_parser.add_argument(
        "config",
        nargs="?",
        default="benchmark_config.yaml",
        help="Path to YAML configuration file (default: benchmark_config.yaml)",
    )
    benchmark_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be run without actually running",
    )
    benchmark_parser.add_argument(
        "--problems",
        nargs="+",
        help="Filter test problems to run (space-separated list of problem names)",
    )
    benchmark_parser.add_argument(
        "--optimizers",
        nargs="+",
        help="Filter optimizers to run (space-separated list of optimizer names)",
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze and visualize benchmark results",
        description="Generate plots and statistics from benchmark results",
    )
    analyze_parser.add_argument(
        "results_dir",
        nargs="?",
        default="./results",
        help="Path to results directory (default: ./results)",
    )
    analyze_parser.add_argument(
        "--problems",
        nargs="+",
        help="Filter results to specific test problems (space-separated list of problem names)",
    )
    analyze_parser.add_argument(
        "--optimizers",
        nargs="+",
        help="Filter results to specific optimizers (space-separated list of optimizer names)",
    )

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 0

    # Import and run the appropriate command
    if args.command == "benchmark":
        from deepobs.benchmark import main as benchmark_main

        sys.argv = ["deepobs benchmark", args.config]
        if args.dry_run:
            sys.argv.append("--dry-run")
        if args.problems:
            sys.argv.append("--problems")
            sys.argv.extend(args.problems)
        if args.optimizers:
            sys.argv.append("--optimizers")
            sys.argv.extend(args.optimizers)
        return benchmark_main()

    elif args.command == "analyze":
        from deepobs.analyze import main as analyze_main

        sys.argv = ["deepobs analyze", args.results_dir]
        if args.problems:
            sys.argv.append("--problems")
            sys.argv.extend(args.problems)
        if args.optimizers:
            sys.argv.append("--optimizers")
            sys.argv.extend(args.optimizers)
        return analyze_main()

    return 0


if __name__ == "__main__":
    sys.exit(main())
