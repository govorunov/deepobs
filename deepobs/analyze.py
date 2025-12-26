#!/usr/bin/env python
"""
Quick result analysis script for DeepOBS benchmark results.
Adapted to work with the actual file structure from StandardRunner.
"""

import json
import os
import glob
import argparse
import numpy as np
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def load_results(results_dir):
    """Load all result JSON files from a directory.

    Args:
        results_dir: Directory containing result files

    Returns:
        Dictionary mapping (testproblem, optimizer) to results
    """
    results = {}

    # Find all JSON files (excluding results.json since actual files have different names)
    pattern = os.path.join(results_dir, '**', '*.json')
    result_files = glob.glob(pattern, recursive=True)

    print(f"Found {len(result_files)} result files")

    for filepath in result_files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            # Extract testproblem and optimizer from the JSON data itself
            if 'testproblem' in data and 'optimizer' in data:
                testproblem = data['testproblem']
                optimizer = data['optimizer']

                # Use the most recent result for each combination
                # (or you could average multiple runs)
                results[(testproblem, optimizer)] = data

        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return results


def create_learning_curve_figures(results):
    """Create Plotly figures for learning curves for all test problems and optimizers.

    Returns:
        List of plotly figure objects
    """
    figures = []

    # Group by test problem
    by_testproblem = defaultdict(dict)
    for (testproblem, optimizer), data in results.items():
        by_testproblem[testproblem][optimizer] = data

    # Create plots for each test problem
    for testproblem in sorted(by_testproblem.keys()):
        optimizer_results = by_testproblem[testproblem]

        # Create subplot with 1 row and 2 columns
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'{testproblem} - Test Loss', f'{testproblem} - Test Accuracy'),
            horizontal_spacing=0.12
        )

        # Plotly default color sequence
        colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

        # Plot both loss and accuracy for each optimizer together to ensure color consistency
        for idx, (optimizer, data) in enumerate(optimizer_results.items()):
            color = colors[idx % len(colors)]

            # Plot test loss (left subplot)
            if 'test_losses' in data:
                epochs = list(range(len(data['test_losses'])))
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=data['test_losses'],
                        mode='lines+markers',
                        name=optimizer,
                        legendgroup=optimizer,
                        showlegend=True,
                        line=dict(color=color),
                        marker=dict(color=color)
                    ),
                    row=1, col=1
                )

            # Plot test accuracy (right subplot)
            if 'test_accuracies' in data:
                epochs = list(range(len(data['test_accuracies'])))
                accuracies = [acc * 100 for acc in data['test_accuracies']]  # Convert to percentage
                fig.add_trace(
                    go.Scatter(
                        x=epochs,
                        y=accuracies,
                        mode='lines+markers',
                        name=optimizer,
                        legendgroup=optimizer,
                        showlegend=False,  # Only show in legend once
                        line=dict(color=color),
                        marker=dict(color=color)
                    ),
                    row=1, col=2
                )

        # Update axes labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_yaxes(title_text="Test Loss", row=1, col=1)
        fig.update_yaxes(title_text="Test Accuracy (%)", row=1, col=2)

        # Update layout
        fig.update_layout(
            height=500,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white'
        )

        figures.append(fig)

    return figures


def create_comparison_bar_figure(results):
    """Create Plotly figure with bar plots comparing optimizers across test problems.

    Returns:
        Plotly figure object
    """
    # Collect data
    testproblems = sorted(set(tp for tp, _ in results.keys()))
    optimizers = sorted(set(opt for _, opt in results.keys()))

    # Create subplots for accuracy and loss
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Optimizer Comparison - Best Test Accuracy',
                        'Optimizer Comparison - Best Test Loss'),
        horizontal_spacing=0.15
    )

    # Plotly default color sequence
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    # Plot both accuracy and loss for each optimizer together to ensure color consistency
    for idx, optimizer in enumerate(optimizers):
        color = colors[idx % len(colors)]

        # Collect accuracy data
        accuracies = []
        for testproblem in testproblems:
            if (testproblem, optimizer) in results:
                res = results[(testproblem, optimizer)]
                if 'test_accuracies' in res:
                    accuracies.append(max(res['test_accuracies']) * 100)  # Convert to percentage
                else:
                    accuracies.append(0)
            else:
                accuracies.append(0)

        # Collect loss data
        losses = []
        for testproblem in testproblems:
            if (testproblem, optimizer) in results:
                res = results[(testproblem, optimizer)]
                if 'test_losses' in res:
                    losses.append(min(res['test_losses']))
                else:
                    losses.append(0)
            else:
                losses.append(0)

        # Add accuracy trace (left subplot)
        fig.add_trace(
            go.Bar(
                x=testproblems,
                y=accuracies,
                name=optimizer,
                legendgroup=optimizer,
                showlegend=True,
                marker=dict(color=color)
            ),
            row=1, col=1
        )

        # Add loss trace (right subplot)
        fig.add_trace(
            go.Bar(
                x=testproblems,
                y=losses,
                name=optimizer,
                legendgroup=optimizer,
                showlegend=False,  # Only show in legend once
                marker=dict(color=color)
            ),
            row=1, col=2
        )

    # Update axes labels
    fig.update_xaxes(title_text="Test Problem", row=1, col=1, tickangle=-45)
    fig.update_xaxes(title_text="Test Problem", row=1, col=2, tickangle=-45)
    fig.update_yaxes(title_text="Best Test Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Best Test Loss", row=1, col=2)

    # Update layout
    fig.update_layout(
        height=500,
        showlegend=True,
        barmode='group',
        template='plotly_white',
        hovermode='x unified'
    )

    return fig


def generate_html_report(results, output_file='./results/benchmark_report.html'):
    """Generate a single self-contained HTML report with all plots.

    Args:
        results: Dictionary mapping (testproblem, optimizer) to results
        output_file: Path to output HTML file
    """
    # Create all figures
    print("Generating learning curve figures...")
    learning_curve_figs = create_learning_curve_figures(results)

    print("Generating comparison bar figure...")
    comparison_fig = create_comparison_bar_figure(results)

    # Create HTML content
    html_parts = []

    # HTML header with styling
    html_parts.append("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DeepOBS Benchmark Results</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #007bff;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            margin-top: 40px;
            margin-bottom: 20px;
        }
        .plot-container {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .timestamp {
            color: #777;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>DeepOBS Benchmark Results</h1>
        <p class="timestamp">Generated: """ + __import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</p>
    </div>
""")

    # Add comparison figure first
    html_parts.append('<h2>Optimizer Comparison Across Test Problems</h2>')
    html_parts.append('<div class="plot-container">')
    html_parts.append(comparison_fig.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append('</div>')

    # Add learning curve figures
    html_parts.append('<h2>Learning Curves by Test Problem</h2>')
    for fig in learning_curve_figs:
        html_parts.append('<div class="plot-container">')
        html_parts.append(fig.to_html(full_html=False, include_plotlyjs=False))
        html_parts.append('</div>')

    # HTML footer
    html_parts.append("""
</body>
</html>
""")

    # Write to file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))

    print(f"\nHTML report saved to: {output_file}")
    return output_file


def print_detailed_summary(results):
    """Print detailed summary of all results."""
    print("\n" + "=" * 80)
    print("DETAILED RESULTS SUMMARY")
    print("=" * 80)

    # Group by test problem
    by_testproblem = defaultdict(dict)
    for (testproblem, optimizer), data in results.items():
        by_testproblem[testproblem][optimizer] = data

    for testproblem in sorted(by_testproblem.keys()):
        print(f"\n{testproblem.upper()}")
        print("-" * 80)
        print(f"{'Optimizer':<15} {'Final Acc (%)':<15} {'Final Loss':<15} {'Epochs':<10} {'LR':<10}")
        print("-" * 80)

        for optimizer in sorted(by_testproblem[testproblem].keys()):
            data = by_testproblem[testproblem][optimizer]
            final_acc = data['test_accuracies'][-1] * 100 if 'test_accuracies' in data else 0
            final_loss = data['test_losses'][-1] if 'test_losses' in data else 0
            num_epochs = data.get('num_epochs', 0)
            lr = data.get('learning_rate', 0)

            print(f"{optimizer:<15} {final_acc:<15.2f} {final_loss:<15.4f} {num_epochs:<10} {lr:<10.4f}")

        print()


def print_statistics(results):
    """Print statistics across all results."""
    stats = defaultdict(lambda: defaultdict(list))

    for (testproblem, optimizer), data in results.items():
        if 'test_accuracies' in data:
            final_acc = data['test_accuracies'][-1] * 100
            stats[optimizer]['accuracies'].append(final_acc)
            stats[optimizer]['testproblems'].append(testproblem)

        if 'test_losses' in data:
            final_loss = data['test_losses'][-1]
            stats[optimizer]['losses'].append(final_loss)

    print("\n" + "=" * 80)
    print("OPTIMIZER PERFORMANCE SUMMARY (Across All Test Problems)")
    print("=" * 80)
    print(f"\n{'Optimizer':<15} {'Avg Acc (%)':<15} {'Std Acc':<15} {'Avg Loss':<15} {'Problems':<10}")
    print("-" * 80)

    for optimizer in sorted(stats.keys()):
        data = stats[optimizer]
        if data['accuracies']:
            mean_acc = np.mean(data['accuracies'])
            std_acc = np.std(data['accuracies'])
            mean_loss = np.mean(data['losses']) if data['losses'] else 0
            num_problems = len(data['accuracies'])

            print(f"{optimizer:<15} {mean_acc:<15.2f} {std_acc:<15.2f} {mean_loss:<15.4f} {num_problems:<10}")

    print("-" * 80)


def main():
    """Main function for result analysis."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Analyze DeepOBS benchmark results and generate plots.'
    )
    parser.add_argument(
        'results_dir',
        nargs='?',
        default='./results',
        help='Path to the directory containing result files (default: ./results)'
    )
    parser.add_argument(
        '--problems',
        nargs='+',
        help='Filter results to specific test problems (space-separated list of problem names)'
    )
    parser.add_argument(
        '--optimizers',
        nargs='+',
        help='Filter results to specific optimizers (space-separated list of optimizer names)'
    )
    args = parser.parse_args()

    print("=" * 80)
    print("DeepOBS Benchmark Results Analysis")
    print("=" * 80)

    # Load results
    results_dir = args.results_dir
    print(f"\nLoading results from: {results_dir}")

    results = load_results(results_dir)

    if not results:
        print("\nNo results found!")
        return

    print(f"\nLoaded {len(results)} unique (test_problem, optimizer) combinations")

    # Filter results if --problems or --optimizers is specified
    if args.problems or args.optimizers:
        available_problems = {tp for tp, _ in results.keys()}
        available_optimizers = {opt for _, opt in results.keys()}

        # Filter by problems if specified
        if args.problems:
            requested_problems = set(args.problems)

            # Check for invalid problem names
            invalid_problems = requested_problems - available_problems
            if invalid_problems:
                print(f"\nWarning: Unknown test problems (will be ignored): {', '.join(sorted(invalid_problems))}")
                print(f"Available problems in results: {', '.join(sorted(available_problems))}")

            # Filter the results
            results = {k: v for k, v in results.items() if k[0] in requested_problems}
            print(f"\nFiltering to problem(s): {', '.join(sorted(requested_problems & available_problems))}")

        # Filter by optimizers if specified
        if args.optimizers:
            requested_optimizers = set(args.optimizers)

            # Check for invalid optimizer names
            invalid_optimizers = requested_optimizers - available_optimizers
            if invalid_optimizers:
                print(f"\nWarning: Unknown optimizers (will be ignored): {', '.join(sorted(invalid_optimizers))}")
                print(f"Available optimizers in results: {', '.join(sorted(available_optimizers))}")

            # Filter the results
            results = {k: v for k, v in results.items() if k[1] in requested_optimizers}
            print(f"\nFiltering to optimizer(s): {', '.join(sorted(requested_optimizers & available_optimizers))}")

        # Check if any results remain after filtering
        if not results:
            print("\n✗ No results match the specified filters!")
            print("Please check the problem and optimizer names and try again.")
            return

        print(f"\nAfter filtering: {len(results)} unique (test_problem, optimizer) combinations")

    # Print summaries
    print_detailed_summary(results)
    print_statistics(results)

    # Generate HTML report
    print("\n" + "=" * 80)
    print("Generating interactive HTML report...")
    print("=" * 80 + "\n")

    try:
        output_file = generate_html_report(results)

        # Create clickable link for terminal (OSC 8 hyperlink)
        abs_path = os.path.abspath(output_file)
        file_url = f"file://{abs_path}"
        clickable_link = f"\033]8;;{file_url}\033\\{output_file}\033]8;;\033\\"

        print("\n" + "=" * 80)
        print("✓ HTML report created successfully!")
        print(f"Open the report: {clickable_link}")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Error generating HTML report: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
