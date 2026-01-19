import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---- CONFIG ----
BASELINE_IDS = ["146", "438", "903", "167", "253"]
CSV_NAME = "subliminal_frequencies.csv"

ROOT_DIR = "."  # repo root (change if needed)
OUTPUT_DIR = "plots"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_column_means(csv_path, folder_name):
    df = pd.read_csv(csv_path)

    agent0_col = f"agent0_{folder_name}"
    agent1_col = f"agent1_{folder_name}"
    agent2_col = f"agent2_{folder_name}"

    return df[agent0_col].mean(), df[agent1_col].mean(), df[agent2_col].mean()


# ---- MAIN LOOP OVER TOP-LEVEL FOLDERS ----
for folder_name in os.listdir(ROOT_DIR):
    folder_path = os.path.join(ROOT_DIR, folder_name)

    if not os.path.isdir(folder_path):
        continue

    baseline_values = []
    non_baseline_results = {}

    # ---- LOOP OVER THREE-DIGIT SUBFOLDERS ----
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)

        if not os.path.isdir(subfolder_path):
            continue

        csv_path = os.path.join(subfolder_path, CSV_NAME)
        if not os.path.exists(csv_path):
            continue

        agent0_mean, agent1_mean, agent2_mean = compute_column_means(csv_path, folder_name)

        if subfolder in BASELINE_IDS:
            baseline_values.append([agent0_mean, agent1_mean, agent2_mean])
        else:
            non_baseline_results[subfolder] = np.array([agent0_mean, agent1_mean, agent2_mean])

    # Skip folder if no valid data
    if not baseline_values or not non_baseline_results:
        continue

    # ---- COMPUTE BASELINE ----
    baseline_df = pd.DataFrame(
        baseline_values, columns=["agent0", "agent1", "agent2"]
    )
    baseline_means = baseline_df.mean().values

    # ---- PLOTTING ----
    x = [0, 1, 2]
    x_labels = ["agent0", "agent1", "agent2"]

    plt.figure(figsize=(6, 4))

    # Baseline
    plt.plot(
        x,
        baseline_means,
        marker="o",
        linewidth=3,
        label="baseline"
    )

    sum_values = np.array([0.,0., 0.])
    count = 0
    # Non-baseline subfolders
    for subfolder, values in non_baseline_results.items():
        sum_values += values
        count +=1

    average_values = sum_values / count

    plt.plot(
        x,
        average_values,
        marker="o",
        linestyle="--",
        alpha=0.7,
        label="average_entanglement"
    )

    plt.xticks(x, x_labels)
    plt.ylabel("Average value")
    plt.title(f"{folder_name}: Agent comparison")
    _, top = plt.ylim()
    plt.ylim(0, top)
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR, f"{folder_name}_agent_comparison_entanglement_averaged.png"
    )
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot for {folder_name} → {output_path}")
