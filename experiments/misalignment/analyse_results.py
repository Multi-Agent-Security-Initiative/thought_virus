import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu

# ---- CONFIG ----
CSV_NAME = "accuracy_rates_unidirectional.csv"

ROOT_DIR = "results"  # repo root (change if needed)
OUTPUT_DIR_PLOTS = "plots"
OUTPUT_DIR_SUMMARYS = "results"

os.makedirs(OUTPUT_DIR_PLOTS, exist_ok=True)
os.makedirs(OUTPUT_DIR_SUMMARYS, exist_ok=True)


def compute_column_means(csv_path):
    """Compute means for agent columns in a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        
        agent0_col = "0"
        agent1_col = "1"
        agent2_col = "2"
        
        # Check if columns exist
        required_cols = [agent0_col, agent1_col, agent2_col]
        if not all(col in df.columns for col in required_cols):
            print(f"Missing columns in {csv_path}: expected {required_cols}")
            return None
        
        return np.array([
            df[agent0_col].mean(),
            df[agent1_col].mean(),
            df[agent2_col].mean()
        ])
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def load_csv_column_values(csv_path, column_name):
    """Load all values from a specific column in a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        if column_name not in df.columns:
            print(f"Column {column_name} not found in {csv_path}")
            return None
        return df[column_name].values
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


# Process best and worst folders
best_path = os.path.join(ROOT_DIR, "truthful")
worst_path = os.path.join(ROOT_DIR, "deceitful")

best_number_folders = [
    d for d in os.listdir(best_path)
    if os.path.isdir(os.path.join(best_path, d))
]

worst_number_folders = [
    d for d in os.listdir(worst_path)
    if os.path.isdir(os.path.join(worst_path, d))
]

# Dictionary to store results for each number folder
best_results = {}
worst_results = {}

# Process each best number subfolder
for number_folder in sorted(best_number_folders):
    csv_path = os.path.join(best_path, number_folder, CSV_NAME)
    
    if not os.path.exists(csv_path):
        print(f"CSV not found in truthful/{number_folder}")
        continue
    
    values = compute_column_means(csv_path)
    if values is not None:
        best_results[number_folder] = values

# Process each worst number subfolder
for number_folder in sorted(worst_number_folders):
    csv_path = os.path.join(worst_path, number_folder, CSV_NAME)
    
    if not os.path.exists(csv_path):
        print(f"CSV not found in deceitful/{number_folder}")
        continue
    
    values = compute_column_means(csv_path)
    if values is not None:
        worst_results[number_folder] = values

# Create the plot
if not best_results and not worst_results:
    print("No valid data for plotting")
else:
    # Calculate averages and find best/worst individual runs
    sum_best = np.array([0., 0., 0.])
    count_best = 0
    best_max_values = np.array([-np.inf, -np.inf, -np.inf])
    best_max_folder = [None, None, None]
    
    for number_folder in sorted(best_results.keys()):
        values = best_results[number_folder]
        if not np.any(np.isnan(values)):
            sum_best += np.array(values)
            count_best += 1
            for i in range(3):
                if values[i] > best_max_values[i]:
                    best_max_values[i] = values[i]
                    best_max_folder[i] = number_folder
    
    sum_worst = np.array([0., 0., 0.])
    count_worst = 0
    worst_min_values = np.array([np.inf, np.inf, np.inf])
    worst_min_folder = [None, None, None]
    
    for number_folder in sorted(worst_results.keys()):
        values = worst_results[number_folder]
        if not np.any(np.isnan(values)):
            sum_worst += np.array(values)
            count_worst += 1
            for i in range(3):
                if values[i] < worst_min_values[i]:
                    worst_min_values[i] = values[i]
                    worst_min_folder[i] = number_folder
    
    # Calculate averages
    avg_best = sum_best / count_best if count_best > 0 else None
    avg_worst = sum_worst / count_worst if count_worst > 0 else None
    
    # Save plotted values to CSV
    plotted_values_df = pd.DataFrame({
        '0': [worst_min_values[0], avg_worst[0] if avg_worst is not None else np.nan, 
              avg_best[0] if avg_best is not None else np.nan, best_max_values[0]],
        '1': [worst_min_values[1], avg_worst[1] if avg_worst is not None else np.nan,
              avg_best[1] if avg_best is not None else np.nan, best_max_values[1]],
        '2': [worst_min_values[2], avg_worst[2] if avg_worst is not None else np.nan,
              avg_best[2] if avg_best is not None else np.nan, best_max_values[2]]
    }, index=['worst_min', 'worst_avg', 'best_avg', 'best_max'])
    
    plotted_values_csv = os.path.join(OUTPUT_DIR_SUMMARYS, "accuracy_rates_unidirectional_summary.csv")
    plotted_values_df.to_csv(plotted_values_csv)
    print(f"✓ Plotted values saved to {plotted_values_csv}")
    
    # Create bar chart
    x = np.arange(3)
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    if count_worst > 0:
        # Worst min (dark red)
        bars1 = ax.bar(x - 1.5*width, worst_min_values, width, 
                       label='Worst (min)', color='darkred', alpha=0.9)
        # Worst average (light red)
        bars2 = ax.bar(x - 0.5*width, avg_worst, width, 
                       label='Worst (avg)', color='lightcoral', alpha=0.9)
    
    if count_best > 0:
        # Best average (light green)
        bars3 = ax.bar(x + 0.5*width, avg_best, width, 
                       label='Best (avg)', color='lightgreen', alpha=0.9)
        # Best max (dark green)
        bars4 = ax.bar(x + 1.5*width, best_max_values, width, 
                       label='Best (max)', color='darkgreen', alpha=0.9)
    
    # Add value labels on bars
    def add_labels(bars, values, color):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.1%}',
                   ha='center', va='bottom', fontsize=9, 
                   color=color, fontweight='bold')
    
    if count_worst > 0:
        add_labels(bars1, worst_min_values, 'darkred')
        add_labels(bars2, avg_worst, 'red')
    
    if count_best > 0:
        add_labels(bars3, avg_best, 'green')
        add_labels(bars4, best_max_values, 'darkgreen')
    
    # Formatting
    ax.set_ylabel('Accuracy Rate', fontsize=12)
    ax.set_title('Best vs Worst Runs - Accuracy Rates', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['Agent 0', 'Agent 1', 'Agent 2'])
    ax.set_ylim(0.60, 0.75)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR_PLOTS, "accuracy_rates_unidirectional.png")
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    print(f"Saved plot → {output_path} (best: {count_best}, worst: {count_worst} numbers)")

    # Statistical tests
    print("\n=== Statistical Tests (Mann-Whitney U) ===")
    
    # Initialize p-value storage
    pvalues_avg = {}
    pvalues_extreme = {}
    
    for agent_idx in range(3):
        # Collect all values for this agent (for average comparison)
        best_values = [best_results[k][agent_idx] for k in sorted(best_results.keys())]
        worst_values = [worst_results[k][agent_idx] for k in sorted(worst_results.keys())]
        
        if len(best_values) >= 1 and len(worst_values) >= 1:
            # Perform Mann-Whitney U test
            statistic, p_value = mannwhitneyu(best_values, worst_values, alternative='two-sided')
            pvalues_avg[str(agent_idx)] = p_value
            
            print(f"\nAgent {agent_idx} (Average comparison):")
            print(f"  Best:  mean={np.mean(best_values):.4f}, n={len(best_values)}")
            print(f"  Worst: mean={np.mean(worst_values):.4f}, n={len(worst_values)}")
            print(f"  Mann-Whitney U statistic: {statistic}")
            print(f"  p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  ✓ Significant difference (p < 0.05)")
            else:
                print(f"  ✗ No significant difference (p >= 0.05)")
        
        # Extreme comparison: load best of best and worst of worst CSV files
        best_extreme_folder = best_max_folder[agent_idx]
        worst_extreme_folder = worst_min_folder[agent_idx]
        
        if best_extreme_folder and worst_extreme_folder:
            best_extreme_csv = os.path.join(best_path, best_extreme_folder, CSV_NAME)
            worst_extreme_csv = os.path.join(worst_path, worst_extreme_folder, CSV_NAME)
            
            best_extreme_values = load_csv_column_values(best_extreme_csv, str(agent_idx))
            worst_extreme_values = load_csv_column_values(worst_extreme_csv, str(agent_idx))
            
            if best_extreme_values is not None and worst_extreme_values is not None:
                # Perform Mann-Whitney U test for extreme comparison
                statistic_extreme, p_value_extreme = mannwhitneyu(
                    best_extreme_values, worst_extreme_values, alternative='two-sided'
                )
                pvalues_extreme[str(agent_idx)] = p_value_extreme
                
                print(f"\nAgent {agent_idx} (Extreme comparison - best of best vs worst of worst):")
                print(f"  Best extreme:  mean={np.mean(best_extreme_values):.4f}, n={len(best_extreme_values)}")
                print(f"  Worst extreme: mean={np.mean(worst_extreme_values):.4f}, n={len(worst_extreme_values)}")
                print(f"  Mann-Whitney U statistic: {statistic_extreme}")
                print(f"  p-value: {p_value_extreme:.4f}")
                
                if p_value_extreme < 0.05:
                    print(f"  ✓ Significant difference (p < 0.05)")
                else:
                    print(f"  ✗ No significant difference (p >= 0.05)")
    
    # Save p-values to CSV
    pvalue_df = pd.DataFrame({
        '0': [pvalues_avg.get('0', np.nan), pvalues_extreme.get('0', np.nan)],
        '1': [pvalues_avg.get('1', np.nan), pvalues_extreme.get('1', np.nan)],
        '2': [pvalues_avg.get('2', np.nan), pvalues_extreme.get('2', np.nan)]
    }, index=['average', 'extreme'])
    
    pvalue_csv_path = os.path.join(OUTPUT_DIR_SUMMARYS, "accuracy_rates_unidirectional_wilcox_pvalue.csv")
    pvalue_df.to_csv(pvalue_csv_path)
    print(f"\n✓ P-values saved to {pvalue_csv_path}")

    print("\n✓ Plot generated!")

# ---- CONFIG ----
CSV_NAME = "logit_diff_unidirectional.csv"

ROOT_DIR = "results"  # repo root (change if needed)
OUTPUT_DIR_PLOTS = "plots"
OUTPUT_DIR_SUMMARYS = "results"

os.makedirs(OUTPUT_DIR_PLOTS, exist_ok=True)
os.makedirs(OUTPUT_DIR_SUMMARYS, exist_ok=True)


# Process best and worst folders
best_path = os.path.join(ROOT_DIR, "truthful")
worst_path = os.path.join(ROOT_DIR, "deceitful")

best_number_folders = [
    d for d in os.listdir(best_path)
    if os.path.isdir(os.path.join(best_path, d))
]

worst_number_folders = [
    d for d in os.listdir(worst_path)
    if os.path.isdir(os.path.join(worst_path, d))
]

# Dictionary to store results for each number folder
best_results = {}
worst_results = {}

# Process each best number subfolder
for number_folder in sorted(best_number_folders):
    csv_path = os.path.join(best_path, number_folder, CSV_NAME)
    
    if not os.path.exists(csv_path):
        print(f"CSV not found in best/{number_folder}")
        continue
    
    values = compute_column_means(csv_path)
    if values is not None:
        best_results[number_folder] = values

# Process each worst number subfolder
for number_folder in sorted(worst_number_folders):
    csv_path = os.path.join(worst_path, number_folder, CSV_NAME)
    
    if not os.path.exists(csv_path):
        print(f"CSV not found in worst/{number_folder}")
        continue
    
    values = compute_column_means(csv_path)
    if values is not None:
        worst_results[number_folder] = values

# Create the plot
if not best_results and not worst_results:
    print("No valid data for plotting")
else:
    # Calculate averages and find best/worst individual runs
    sum_best = np.array([0., 0., 0.])
    count_best = 0
    best_max_values = np.array([-np.inf, -np.inf, -np.inf])
    best_max_folder = [None, None, None]
    
    for number_folder in sorted(best_results.keys()):
        values = best_results[number_folder]
        if not np.any(np.isnan(values)):
            sum_best += np.array(values)
            count_best += 1
            for i in range(3):
                if values[i] > best_max_values[i]:
                    best_max_values[i] = values[i]
                    best_max_folder[i] = number_folder
    
    sum_worst = np.array([0., 0., 0.])
    count_worst = 0
    worst_min_values = np.array([np.inf, np.inf, np.inf])
    worst_min_folder = [None, None, None]
    
    for number_folder in sorted(worst_results.keys()):
        values = worst_results[number_folder]
        if not np.any(np.isnan(values)):
            sum_worst += np.array(values)
            count_worst += 1
            for i in range(3):
                if values[i] < worst_min_values[i]:
                    worst_min_values[i] = values[i]
                    worst_min_folder[i] = number_folder
    
    # Calculate averages
    avg_best = sum_best / count_best if count_best > 0 else None
    avg_worst = sum_worst / count_worst if count_worst > 0 else None
    
    # Save plotted values to CSV
    plotted_values_df = pd.DataFrame({
        '0': [worst_min_values[0], avg_worst[0] if avg_worst is not None else np.nan, 
              avg_best[0] if avg_best is not None else np.nan, best_max_values[0]],
        '1': [worst_min_values[1], avg_worst[1] if avg_worst is not None else np.nan,
              avg_best[1] if avg_best is not None else np.nan, best_max_values[1]],
        '2': [worst_min_values[2], avg_worst[2] if avg_worst is not None else np.nan,
              avg_best[2] if avg_best is not None else np.nan, best_max_values[2]]
    }, index=['worst_min', 'worst_avg', 'best_avg', 'best_max'])
    
    plotted_values_csv = os.path.join(OUTPUT_DIR_SUMMARYS, "logprobs_diff_unidirectional.csv")
    plotted_values_df.to_csv(plotted_values_csv)
    print(f"✓ Plotted values saved to {plotted_values_csv}")
    
    # Create bar chart
    x = np.arange(3)
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars
    if count_worst > 0:
        # Worst min (dark red)
        bars1 = ax.bar(x - 1.5*width, worst_min_values, width, 
                       label='Worst (min)', color='darkred', alpha=0.9)
        # Worst average (light red)
        bars2 = ax.bar(x - 0.5*width, avg_worst, width, 
                       label='Worst (avg)', color='lightcoral', alpha=0.9)
    
    if count_best > 0:
        # Best average (light green)
        bars3 = ax.bar(x + 0.5*width, avg_best, width, 
                       label='Best (avg)', color='lightgreen', alpha=0.9)
        # Best max (dark green)
        bars4 = ax.bar(x + 1.5*width, best_max_values, width, 
                       label='Best (max)', color='darkgreen', alpha=0.9)
    
    # Add value labels on bars
    def add_labels(bars, values, color):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}',
                   ha='center', va='bottom', fontsize=9, 
                   color=color, fontweight='bold')
    
    if count_worst > 0:
        add_labels(bars1, worst_min_values, 'darkred')
        add_labels(bars2, avg_worst, 'red')
    
    if count_best > 0:
        add_labels(bars3, avg_best, 'green')
        add_labels(bars4, best_max_values, 'darkgreen')
    
    # Formatting
    ax.set_ylabel('Logprobs Diff (right vs. average wrong answer)', fontsize=12)
    ax.set_title('Best vs Worst Runs - Logprob Differences', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(['Agent 0', 'Agent 1', 'Agent 2'])
    ax.set_ylim(7., 10.)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(OUTPUT_DIR_PLOTS, "logprobs_diff_unidirectional.png")
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    print(f"Saved plot → {output_path} (best: {count_best}, worst: {count_worst} numbers)")

    # Statistical tests
    print("\n=== Statistical Tests (Mann-Whitney U) ===")
    
    # Initialize p-value storage
    pvalues_avg = {}
    pvalues_extreme = {}
    
    for agent_idx in range(3):
        # Collect all values for this agent (for average comparison)
        best_values = [best_results[k][agent_idx] for k in sorted(best_results.keys())]
        worst_values = [worst_results[k][agent_idx] for k in sorted(worst_results.keys())]
        
        if len(best_values) >= 1 and len(worst_values) >= 1:
            # Perform Mann-Whitney U test
            statistic, p_value = mannwhitneyu(best_values, worst_values, alternative='two-sided')
            pvalues_avg[str(agent_idx)] = p_value
            
            print(f"\nAgent {agent_idx} (Average comparison):")
            print(f"  Best:  mean={np.mean(best_values):.4f}, n={len(best_values)}")
            print(f"  Worst: mean={np.mean(worst_values):.4f}, n={len(worst_values)}")
            print(f"  Mann-Whitney U statistic: {statistic}")
            print(f"  p-value: {p_value:.4f}")
            
            if p_value < 0.05:
                print(f"  ✓ Significant difference (p < 0.05)")
            else:
                print(f"  ✗ No significant difference (p >= 0.05)")
        
        # Extreme comparison: load best of best and worst of worst CSV files
        best_extreme_folder = best_max_folder[agent_idx]
        worst_extreme_folder = worst_min_folder[agent_idx]
        
        if best_extreme_folder and worst_extreme_folder:
            best_extreme_csv = os.path.join(best_path, best_extreme_folder, CSV_NAME)
            worst_extreme_csv = os.path.join(worst_path, worst_extreme_folder, CSV_NAME)
            
            best_extreme_values = load_csv_column_values(best_extreme_csv, str(agent_idx))
            worst_extreme_values = load_csv_column_values(worst_extreme_csv, str(agent_idx))
            
            if best_extreme_values is not None and worst_extreme_values is not None:
                # Perform Mann-Whitney U test for extreme comparison
                statistic_extreme, p_value_extreme = mannwhitneyu(
                    best_extreme_values, worst_extreme_values, alternative='two-sided'
                )
                pvalues_extreme[str(agent_idx)] = p_value_extreme
                
                print(f"\nAgent {agent_idx} (Extreme comparison - best of best vs worst of worst):")
                print(f"  Best extreme:  mean={np.mean(best_extreme_values):.4f}, n={len(best_extreme_values)}")
                print(f"  Worst extreme: mean={np.mean(worst_extreme_values):.4f}, n={len(worst_extreme_values)}")
                print(f"  Mann-Whitney U statistic: {statistic_extreme}")
                print(f"  p-value: {p_value_extreme:.4f}")
                
                if p_value_extreme < 0.05:
                    print(f"  ✓ Significant difference (p < 0.05)")
                else:
                    print(f"  ✗ No significant difference (p >= 0.05)")
    
    # Save p-values to CSV
    pvalue_df = pd.DataFrame({
        '0': [pvalues_avg.get('0', np.nan), pvalues_extreme.get('0', np.nan)],
        '1': [pvalues_avg.get('1', np.nan), pvalues_extreme.get('1', np.nan)],
        '2': [pvalues_avg.get('2', np.nan), pvalues_extreme.get('2', np.nan)]
    }, index=['average', 'extreme'])
    
    pvalue_csv_path = os.path.join(OUTPUT_DIR_SUMMARYS, "logprobs_diff_unidirectional_wilcox_pvalue.csv")
    pvalue_df.to_csv(pvalue_csv_path)
    print(f"\n✓ P-values saved to {pvalue_csv_path}")

    print("\n✓ Plot generated!")