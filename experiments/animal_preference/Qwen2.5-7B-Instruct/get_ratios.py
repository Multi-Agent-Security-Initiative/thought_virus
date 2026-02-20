import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import mannwhitneyu

log = True
one_way = False

# ---- CONFIG ----
CSV_NAME = "subliminal_frequencies_unidirectional.csv" if one_way else "subliminal_frequencies_bidirectional.csv"

ROOT_DIR = "results"  # repo root (change if needed)
OUTPUT_DIR = "extra_results/unidirectional/plots_all_numbers_and_baselines" if one_way else "extra_results/bidirectional/plots_all_numbers_and_baselines"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_column_means(csv_path, animal_name, num_agents=6, folder_path=None):
    """Compute means for agent columns in a CSV file, with optional filtering."""
    if folder_path == None:
        folder_path ='/'.join(csv_path.split("/")[:-1])
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
        #print(csv_path)
        if folder_path is not None:
            concept_counts_path = os.path.join(folder_path, "conversation_concept_counts.csv")
            #print(concept_counts_path)
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

        # Return mean of frequencies (no exp needed, already in frequency space)
        means = np.array([df_filtered[col].mean() for col in required_cols])
        print(f"{animal_name} {csv_path.split("/")[2]} | Excluded count: {excluded_count}")
        return means, total_rows, excluded_count, remaining_count
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None, 0, 0, 0


# Find all animals in this method
animals = [
    d for d in os.listdir(ROOT_DIR) if (("plot" not in d) and (d != "random"))
]

ratio_df = pd.DataFrame(columns = range(6),
                        index = animals)
wilcox_df = pd.DataFrame(columns = range(6),
                        index = animals)
strongest_v_random_df = pd.DataFrame(columns = range(6),
                        index = animals)

random_path = os.path.join(ROOT_DIR, "random")
random_number_folders = [
    d for d in os.listdir(random_path)
    if os.path.isdir(os.path.join(random_path, d))
]

# Dictionary to store results for each number folder
random_number_results = {a : {} for a in animals}

# Process each number subfolder
for random_number_folders in sorted(random_number_folders):
    csv_path = os.path.join(random_path, random_number_folders, CSV_NAME)
    
    if not os.path.exists(csv_path):
        print(f"    CSV not found in {random_number_folders}")
        continue
    
    for a in animals:
        values = compute_column_means(csv_path, a)[0]
        if values is not None:
            random_number_results[a][random_number_folders] = values

for animal in sorted(animals):
    print(f"  Processing animal: {animal}")
    
    animal_path = os.path.join(ROOT_DIR, animal)
    
    # Find all number subfolders
    number_folders = [
        d for d in os.listdir(animal_path)
        if os.path.isdir(os.path.join(animal_path, d))
    ]
    
    if not number_folders:
        print(f"    No number folders found")
        continue
    
    # Dictionary to store results for each number folder
    number_results = {}
    
    # Process each number subfolder
    for number_folder in sorted(number_folders):
        csv_path = os.path.join(animal_path, number_folder, CSV_NAME)
        
        if not os.path.exists(csv_path):
            print(f"    CSV not found in {number_folder}")
            continue
        
        values = compute_column_means(csv_path, animal)[0]
        if values is not None:
            number_results[number_folder] = values
    
    if not number_results:
        print(f"    No valid data for plotting")
        continue
    
    # ---- STEP 4: Plot all number folders for this method-animal combination ----
    x = [0, 1, 2, 3, 4, 5]
    x_labels = ["agent0", "agent1", "agent2", "agent3", "agent4", "agent5"]
    
    plt.figure(figsize=(10, 6))
    
    # Plot each number folder as a separate line
    sum_values = np.array([0.,0.,0., 0., 0., 0.])
    counter = 0
    for random_number_folder in sorted(random_number_results[animal].keys()):
        values = random_number_results[animal][random_number_folder]
        sum_values += np.array(values)
        counter += 1

        plt.plot(
            x,
            values,
            marker="o",
            linewidth=1,
            linestyle=":",
            color="black", 
            label=f"baseline ({random_number_folder})",
            alpha=.3
        )

    # Plot baseline average and add value labels
    avg_baseline = sum_values / counter
    plt.plot(
            x,
            avg_baseline,
            marker="o",
            linewidth=2,
            linestyle="-",
            color="black", 
            label=f"baseline (avg)",
            alpha=1.
        )
    
    # Add value labels for baseline average
    for i, val in enumerate(avg_baseline):
        plt.text(x[i], val + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', 
                fontsize=10, color='black', fontweight='bold')
        
    sum_values = np.array([0.,0.,0., 0., 0., 0.])
    counter = 0
    # Plot each number folder as a separate line
    for number_folder in sorted(number_results.keys()):
        values = number_results[number_folder]
        sum_values += np.array(values)
        counter += 1
        plt.plot(
            x,
            values,
            marker="o",
            linewidth=1,
            linestyle=":",
            color="red", 
            label=f"subliminal ({number_folder})",
            alpha=.3
        )

    # Plot subliminal average and add value labels
    avg_subliminal = sum_values / counter
    plt.plot(
            x,
            avg_subliminal,
            marker="o",
            linewidth=2,
            linestyle="-",
            color="red", 
            label=f"subliminal (avg)",
            alpha=1.
        )
    
    # Add value labels for subliminal average
    for i, val in enumerate(avg_subliminal):
        plt.text(x[i], val + 0.01, f'{val:.3f}', 
                ha='center', va='bottom', 
                fontsize=10, color='red', fontweight='bold')
        
    for i in range(6):
        plt.text(x[i], avg_subliminal[i] + 0.04, f'{avg_subliminal[i] / avg_baseline[i]:.1f}x', 
                ha='center', va='bottom', 
                fontsize=10, color='red', fontweight='bold')
        ratio_df.loc[animal, i] = avg_subliminal[i] / avg_baseline[i]
        
        # Calculate best subliminal value for this agent position
        # print(number_results)
        animal_values = [number_results[k][i] for k in number_results.keys()]
        best_subliminal = max(animal_values) if animal_values else 0
        strongest_v_random_df.loc[animal, i] = best_subliminal / avg_baseline[i]
        # print(animal_values)        

        # Wilcoxon test
        random_values = [random_number_results[animal][k][i] for k in sorted(random_number_results[animal].keys())]
        if len(animal_values) >= 1 and len(random_values) >= 1:
            # Perform Mann-Whitney U test
            statistic, p_value = mannwhitneyu(animal_values, random_values, alternative='greater')
            wilcox_df.loc[animal, i] = p_value
    
    plt.xticks(x, x_labels)
    plt.ylabel("Average value")
    plt.title(f"Animal: {animal}")
    if log:
        plt.yscale('log')
        output_path = os.path.join(OUTPUT_DIR, f"{animal}_all_numbers_baseline_and_subliminal_log.png")
    else:
        _, top = plt.ylim()
        plt.ylim(0, top)
        output_path = os.path.join(OUTPUT_DIR, f"{animal}_all_numbers_baseline_and_subliminal.png")
    plt.legend(title="Number", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create filename with method and animal
    
    plt.savefig(output_path, dpi=100)
    plt.close()
    print(f"    Saved plot → {output_path} ({len(number_results)} numbers)")

print("\n✓ All plots generated!")
ratio_df.to_csv(os.path.join(OUTPUT_DIR, "ratios.csv"))
wilcox_df.to_csv(os.path.join(OUTPUT_DIR, "ratios_wilcox_pvalue.csv"))
print(strongest_v_random_df)
strongest_v_random_df.to_csv(os.path.join(OUTPUT_DIR, "strongest_v_random_ratio.csv"))