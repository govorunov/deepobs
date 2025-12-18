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
        prog='deepobs',
        description='DeepOBS - Deep Learning Optimizer Benchmark Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with default config
  deepobs benchmark

  # Run benchmark with custom config
  deepobs benchmark my_config.yaml

  # Analyze results
  deepobs analyze

  # Analyze results from custom directory
  deepobs analyze --results-dir ./my_results

For more information, visit: https://github.com/fsschneider/DeepOBS
        """
    )

    parser.add_argument(
        '--version',
        action='version',
        version='DeepOBS 1.2.0'
    )

    # Create subparsers for commands
    subparsers = parser.add_subparsers(
        title='commands',
        description='Available commands',
        dest='command',
        help='Command to run'
    )

    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Run optimizer benchmark suite',
        description='Run a comprehensive benchmark of optimizers on test problems'
    )
    benchmark_parser.add_argument(
        'config',
        nargs='?',
        default='benchmark_config.yaml',
        help='Path to YAML configuration file (default: benchmark_config.yaml)'
    )
    benchmark_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print what would be run without actually running'
    )

    # Analyze command
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Analyze and visualize benchmark results',
        description='Generate plots and statistics from benchmark results'
    )
    analyze_parser.add_argument(
        'results_dir',
        nargs='?',
        default='./results',
        help='Path to results directory (default: ./results)'
    )

    # Parse arguments
    args = parser.parse_args()

    # If no command specified, show help
    if not args.command:
        parser.print_help()
        return 0

    # Import and run the appropriate command
    if args.command == 'benchmark':
        from deepobs.benchmark import main as benchmark_main
        sys.argv = ['deepobs benchmark', args.config]
        if args.dry_run:
            sys.argv.append('--dry-run')
        return benchmark_main()

    elif args.command == 'analyze':
        from deepobs.analyze import main as analyze_main
        sys.argv = ['deepobs analyze', args.results_dir]
        return analyze_main()

    return 0


if __name__ == '__main__':
    sys.exit(main())
