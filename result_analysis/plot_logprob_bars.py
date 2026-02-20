#!/usr/bin/env python3
"""
Plot log probability bar charts for subliminal concept analysis.

Usage:
    python plot_logprob_bars.py <experiment_folder>

Example:
    python plot_logprob_bars.py experiments/Qwen2.5-7B-Instruct
"""

import os
import sys
import importlib.util
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def load_experiment_config(experiment_folder):
    """Dynamically load experiment_config.py from the experiment folder."""
    config_path = os.path.join(experiment_folder, "experiment_config.py")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"experiment_config.py not found in {experiment_folder}")

    # Load module dynamically
    spec = importlib.util.spec_from_file_location("experiment_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config


def bootstrap_confidence_interval(data, n_bootstrap=10000, confidence_level=0.95, statistic=np.mean):
    """
    Calculate bootstrap confidence intervals.

    Parameters:
    -----------
    data : array-like
        Original values
    n_bootstrap : int
        Number of bootstrap samples (default: 10000)
    confidence_level : float
        Confidence level for the interval (default: 0.95 for 95% CI)
    statistic : callable
        Function to compute (default: np.mean)

    Returns:
    --------
    tuple: (stat_value, lower_bound, upper_bound)
    """
    data = np.array(data)
    bootstrap_stats = []

    # Generate bootstrap samples
    for _ in range(n_bootstrap):
        # Resample with replacement
        bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
        bootstrap_stats.append(statistic(bootstrap_sample))

    bootstrap_stats = np.array(bootstrap_stats)

    # Calculate percentiles for confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    stat_value = statistic(data)

    return stat_value, lower_bound, upper_bound


def compute_column_means(csv_path, animal_name, num_agents=6, folder_path=None):
    """Compute means for agent columns in a CSV file, with optional filtering."""
    try:
        df = pd.read_csv(csv_path, index_col=0)

        # Build column names
        required_cols = [f"agent{i}_{animal_name}" for i in range(num_agents)]

        # Check if columns exist
        if not all(col in df.columns for col in required_cols):
            print(f"Missing columns in {csv_path}: expected {required_cols}")
            return None

        # Count fully filled rows (no NaN values)
        fully_filled_mask = df[required_cols].notna().all(axis=1)
        total_rows = len(df)
        fully_filled_count = fully_filled_mask.sum()

        # Apply filtering based on conversation_concept_counts.csv
        excluded_count = 0
        if folder_path is not None:
            concept_counts_path = os.path.join(folder_path, "conversation_concept_counts.csv")
            if os.path.exists(concept_counts_path):
                concept_df = pd.read_csv(concept_counts_path, index_col=0)
                if animal_name in concept_df.columns:
                    # Filter out rows where concept count > 0
                    concept_mask = concept_df[animal_name] == 0
                    # Align indices
                    concept_mask = concept_mask.reindex(df.index, fill_value=False)
                    excluded_count = (~concept_mask).sum()
                    # Combine with fully filled mask
                    final_mask = fully_filled_mask & concept_mask
                else:
                    final_mask = fully_filled_mask
            else:
                final_mask = fully_filled_mask
        else:
            final_mask = fully_filled_mask

        remaining_count = final_mask.sum()

        # Filter dataframe
        df_filtered = df[final_mask]

        if len(df_filtered) == 0:
            print(f"  Warning: No valid rows after filtering in {csv_path}")
            return None, 0, 0, 0

        # Return exp of log probs
        means = np.array([np.exp(df_filtered[col]).mean() for col in required_cols])
        return means, total_rows, excluded_count, remaining_count
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None, 0, 0, 0


def get_base_rate(animal, results_path):
    """
    Read base logprobs from base_logprobs.csv.
    """

    # Fallback to base_logprobs.csv in experiment folder
    base_csv_path = os.path.join(os.path.dirname(results_path), "base_logprobs.csv")

    if os.path.exists(base_csv_path):
        try:
            df = pd.read_csv(base_csv_path)
            base_row = df[df.iloc[:, 0] == 'base']

            if not base_row.empty and animal in df.columns:
                log_prob = base_row[animal].values[0]
                base_rate = np.exp(log_prob)
                print(f"  Base rate for {animal}: {base_rate:.6e} (from base_logprobs.csv)")
                return base_rate
        except Exception as e:
            print(f"  Error reading base_logprobs.csv: {e}")

    print(f"  Warning: Could not find base rate for {animal}")
    return None


def create_bar_plot(animal, num_agents, avg_baseline, baseline_errors_lower, baseline_errors_upper,
                    avg_subliminal, subliminal_errors_lower, subliminal_errors_upper,
                    subliminal_strongest, subliminal_strongest_errors_lower, subliminal_strongest_errors_upper,
                    base_rate, output_path):
    """Create and save the bar plot for a single animal."""

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Agent labels
    x_labels = [f"Agent {i}" for i in range(num_agents)]

    # Bar width and positions
    bar_width = 0.18
    group_gap = 0.02
    x_pos = np.arange(num_agents) * 1.2

    # Colors
    color_base = "black"
    color_random = "blue"
    color_subliminal = "#f47b17"
    color_strongest = "red"

    # Legend labels
    legend_base = "Base rate"
    legend_random = "Random token (average)"
    legend_subliminal = "Subliminal token (average)"
    legend_strongest = "Subliminal token (strongest)"

    # Plot bars
    bars1 = ax.bar(x_pos - 1.5*bar_width - group_gap, [base_rate]*num_agents, bar_width,
                   label=legend_base, color=color_base, alpha=1., edgecolor='black', linewidth=0.)

    bars2 = ax.bar(x_pos - 0.5*bar_width - group_gap/3, avg_baseline, bar_width,
                   label=legend_random, color=color_random, alpha=1., edgecolor='black', linewidth=0.)
    ax.errorbar(x_pos - 0.5*bar_width - group_gap/3, avg_baseline,
                yerr=[baseline_errors_lower, baseline_errors_upper],
                fmt='none', ecolor='black', capsize=3, capthick=1.5, linewidth=1.5)

    bars3 = ax.bar(x_pos + 0.5*bar_width + group_gap/3, avg_subliminal, bar_width,
                   label=legend_subliminal, color=color_subliminal, alpha=1., edgecolor='black', linewidth=0.)
    ax.errorbar(x_pos + 0.5*bar_width + group_gap/3, avg_subliminal,
                yerr=[subliminal_errors_lower, subliminal_errors_upper],
                fmt='none', ecolor='black', capsize=3, capthick=1.5, linewidth=1.5)

    bars4 = ax.bar(x_pos + 1.5*bar_width + group_gap, subliminal_strongest, bar_width,
                   label=legend_strongest, color=color_strongest, alpha=1., edgecolor='black', linewidth=0.3)
    ax.errorbar(x_pos + 1.5*bar_width + group_gap, subliminal_strongest,
                yerr=[subliminal_strongest_errors_lower, subliminal_strongest_errors_upper],
                fmt='none', ecolor='black', capsize=3, capthick=1.5, linewidth=1.5)

    # Set log scale
    ax.set_yscale('log')

    # Set y-axis limits
    all_bar_values = np.concatenate([
        [base_rate]*num_agents,
        avg_baseline,
        avg_subliminal,
        subliminal_strongest
    ])
    min_value = np.min(all_bar_values[all_bar_values > 0])
    max_value = np.max(all_bar_values)

    bottom_limit = min_value * 0.5
    top_limit = max_value * 1.5

    y_ticks = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
    visible_ticks = [tick for tick in y_ticks if bottom_limit <= tick <= top_limit]
    if not visible_ticks:
        visible_ticks = y_ticks

    ax.set_ylim(bottom=bottom_limit, top=top_limit)
    ax.set_yticks(visible_ticks)
    ax.set_yticklabels([f'{tick:.6f}'.rstrip('0').rstrip('.') for tick in visible_ticks], fontsize=12)
    plt.minorticks_off()

    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, fontsize=12)

    # Add ratio labels
    for i in range(num_agents):
        # Orange/black ratio
        ratio_orange = avg_subliminal[i] / base_rate
        orange_height = avg_subliminal[i] + subliminal_errors_upper[i]
        ax.text(x_pos[i] + 0.5*bar_width + group_gap/3, orange_height * 1.25,
                f'{ratio_orange:.1f}×',
                ha='center', va='bottom',
                fontsize=11, color=color_subliminal, fontweight='bold')

        # Red/black ratio
        ratio_red = subliminal_strongest[i] / base_rate
        red_height = subliminal_strongest[i] + subliminal_strongest_errors_upper[i]
        ax.text(x_pos[i] + 1.5*bar_width + group_gap, red_height * 1.25,
                f'{ratio_red:.1f}×',
                ha='center', va='bottom',
                fontsize=11, color=color_strongest, fontweight='bold')

    # Labels and title
    ax.set_ylabel("Response Frequency (log scale)", fontsize=14)
    ax.set_title(f"'{animal}'", fontsize=16, pad=20, fontweight='bold')

    # Legend
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 1, 2, 3]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=2, fontsize=12, frameon=False, fancybox=True, shadow=True)

    # Remove spines
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def main(mode="unidirectional"):
    if len(sys.argv) < 2:
        print("Usage: python plot_logprob_bars.py <experiment_folder>")
        print("Example: python plot_logprob_bars.py experiments/Qwen2.5-7B-Instruct")
        sys.exit(1)

    experiment_folder = sys.argv[1]

    # Load experiment config
    print(f"Loading configuration from {experiment_folder}/experiment_config.py...")
    try:
        config = load_experiment_config(experiment_folder)
    except Exception as e:
        print(f"Error loading config: {e}")
        sys.exit(1)

    # Get parameters from config
    concepts = config.CONCEPTS
    num_agents = config.NUMBER_OF_AGENTS

    print(f"Concepts to analyze: {concepts}")
    print(f"Number of agents: {num_agents}")

    # Setup paths
    results_path = os.path.join(experiment_folder, "results")
    csv_name = f"subliminal_logprobs_{mode}.csv"

    # Create output directory
    output_dir = os.path.join(experiment_folder, "plots", f"{mode}_logprob_bars")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}\n")

    # Process each animal
    for animal in concepts:
        print(f"{'='*60}")
        print(f"Processing: {animal}")
        print('='*60)

        # Get base rate
        base_rate = get_base_rate(animal, results_path)
        if base_rate is None:
            print(f"  Skipping {animal} - no base rate found\n")
            continue

        animal_path = os.path.join(results_path, animal)
        random_path = os.path.join(results_path, "random")

        # Process random tokens
        random_number_folders = [
            d for d in os.listdir(random_path)
            if os.path.isdir(os.path.join(random_path, d))
        ]

        random_all_data = {f'agent{i}': [] for i in range(num_agents)}
        random_total_excluded = 0
        random_total_remaining = 0

        print(f"\n  Processing random tokens:")
        for random_folder in sorted(random_number_folders):
            folder_path = os.path.join(random_path, random_folder)
            csv_path = os.path.join(folder_path, csv_name)
            if not os.path.exists(csv_path):
                continue

            result = compute_column_means(csv_path, animal, num_agents, folder_path)
            if result[0] is not None:
                values, total_rows, excluded, remaining = result
                random_total_excluded += excluded
                random_total_remaining += remaining
                print(f"    Folder {random_folder}: {total_rows} total rows, {excluded} excluded (concept appeared), {remaining} remaining")
                for i in range(num_agents):
                    random_all_data[f'agent{i}'].append(values[i])

        print(f"  Random tokens summary: {random_total_excluded} total excluded, {random_total_remaining} total remaining")

        # Process subliminal tokens
        if not os.path.exists(animal_path):
            print(f"  Warning: Animal path not found: {animal_path}")
            continue

        number_folders = [
            d for d in os.listdir(animal_path)
            if os.path.isdir(os.path.join(animal_path, d))
        ]

        if not number_folders:
            print(f"  No subliminal data folders found\n")
            continue

        number_results = {}
        subliminal_all_data = {f'agent{i}': [] for i in range(num_agents)}
        subliminal_total_excluded = 0
        subliminal_total_remaining = 0

        print(f"\n  Processing subliminal tokens:")
        for number_folder in sorted(number_folders):
            folder_path = os.path.join(animal_path, number_folder)
            csv_path = os.path.join(folder_path, csv_name)
            if not os.path.exists(csv_path):
                continue

            result = compute_column_means(csv_path, animal, num_agents, folder_path)
            if result[0] is not None:
                values, total_rows, excluded, remaining = result
                subliminal_total_excluded += excluded
                subliminal_total_remaining += remaining
                print(f"    Folder {number_folder}: {total_rows} total rows, {excluded} excluded (concept appeared), {remaining} remaining")
                number_results[number_folder] = values
                for i in range(num_agents):
                    subliminal_all_data[f'agent{i}'].append(values[i])

        print(f"  Subliminal tokens summary: {subliminal_total_excluded} total excluded, {subliminal_total_remaining} total remaining")

        if not number_results:
            print(f"  No valid subliminal data found\n")
            continue

        # Find best folder per agent
        best_number_per_agent = {}
        for i in range(num_agents):
            best_val = -np.inf
            best_key = None
            for key, values in number_results.items():
                if values[i] > best_val:
                    best_val = values[i]
                    best_key = key
            best_number_per_agent[i] = best_key

        print(f"  Best subliminal folders per agent: {best_number_per_agent}")

        # Calculate baseline statistics
        avg_baseline = np.zeros(num_agents)
        baseline_errors_lower = np.zeros(num_agents)
        baseline_errors_upper = np.zeros(num_agents)

        for i in range(num_agents):
            data = random_all_data[f'agent{i}']
            if not data:
                continue
            mean = np.mean(data)
            _, lower, upper = bootstrap_confidence_interval(data)
            avg_baseline[i] = mean
            baseline_errors_lower[i] = mean - lower
            baseline_errors_upper[i] = upper - mean

        # Calculate subliminal average statistics
        avg_subliminal = np.zeros(num_agents)
        subliminal_errors_lower = np.zeros(num_agents)
        subliminal_errors_upper = np.zeros(num_agents)

        for i in range(num_agents):
            data = subliminal_all_data[f'agent{i}']
            if not data:
                continue
            mean = np.mean(data)
            _, lower, upper = bootstrap_confidence_interval(data)
            avg_subliminal[i] = mean
            subliminal_errors_lower[i] = mean - lower
            subliminal_errors_upper[i] = upper - mean

        # Calculate strongest subliminal statistics
        subliminal_strongest = np.zeros(num_agents)
        subliminal_strongest_errors_lower = np.zeros(num_agents)
        subliminal_strongest_errors_upper = np.zeros(num_agents)

        print(f"\n  Processing strongest subliminal tokens:")
        for i in range(num_agents):
            best_number = best_number_per_agent[i]
            folder_path = os.path.join(animal_path, best_number)
            csv_path = os.path.join(folder_path, csv_name)

            try:
                df = pd.read_csv(csv_path, index_col=0)

                # Apply filtering based on conversation_concept_counts.csv
                concept_counts_path = os.path.join(folder_path, "conversation_concept_counts.csv")
                if os.path.exists(concept_counts_path):
                    concept_df = pd.read_csv(concept_counts_path, index_col=0)
                    if animal in concept_df.columns:
                        # Filter out rows where concept count > 0
                        concept_mask = concept_df[animal] == 0
                        # Align indices
                        concept_mask = concept_mask.reindex(df.index, fill_value=False)
                        df = df[concept_mask]
                        excluded = (~concept_mask).sum()
                        remaining = concept_mask.sum()
                        print(f"    Agent {i} (folder {best_number}): {excluded} excluded, {remaining} remaining")

                # Convert from log space to probability space
                data = np.exp(df[f"agent{i}_{animal}"])

                actual_mean = np.mean(data)
                _, lower, upper = bootstrap_confidence_interval(data, statistic=np.mean)
                subliminal_strongest[i] = actual_mean
                subliminal_strongest_errors_lower[i] = actual_mean - lower
                subliminal_strongest_errors_upper[i] = upper - actual_mean
            except Exception as e:
                print(f"  Warning: Error reading best data for agent {i}: {e}")

        # Create plot
        output_path = os.path.join(output_dir, f"{animal}_logprob_bars.png")
        create_bar_plot(
            animal, num_agents,
            avg_baseline, baseline_errors_lower, baseline_errors_upper,
            avg_subliminal, subliminal_errors_lower, subliminal_errors_upper,
            subliminal_strongest, subliminal_strongest_errors_lower, subliminal_strongest_errors_upper,
            base_rate, output_path
        )

        print(f"  ✓ Saved plot → {output_path}\n")

    print(f"\n{'='*60}")
    print(f"All plots saved to: {output_dir}")
    print('='*60)


if __name__ == "__main__":
    main(mode="unidirectional")
    main(mode="bidirectional")
