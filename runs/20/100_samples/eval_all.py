import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---- CONFIG ----
CSV_NAME = "subliminal_frequencies.csv"

ROOT_DIR = "."  # repo root (change if needed)
OUTPUT_DIR = "plots_all_numbers"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_column_means(csv_path, animal_name):
    """Compute means for agent columns in a CSV file."""
    try:
        df = pd.read_csv(csv_path)
        
        agent0_col = f"agent0_{animal_name}"
        agent1_col = f"agent1_{animal_name}"
        agent2_col = f"agent2_{animal_name}"
        
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


# ---- STEP 1: Discover all methods (sibling directories) ----
methods = [
    d for d in os.listdir(ROOT_DIR)
    if os.path.isdir(os.path.join(ROOT_DIR, d)) and d != OUTPUT_DIR
]

if not methods:
    print("No method directories found!")
    exit()

print(f"Found methods: {methods}")


# ---- STEP 2: Process each method ----
for method in sorted(methods):
    method_path = os.path.join(ROOT_DIR, method)
    
    # Find all animals in this method
    animals = [
        d for d in os.listdir(method_path)
        if os.path.isdir(os.path.join(method_path, d))
    ]
    
    if not animals:
        print(f"No animals found in method '{method}'")
        continue
    
    print(f"\nProcessing method: {method}")
    
    # ---- STEP 3: Process each animal in this method ----
    for animal in sorted(animals):
        print(f"  Processing animal: {animal}")
        
        animal_path = os.path.join(method_path, animal)
        
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
            
            values = compute_column_means(csv_path, animal)
            if values is not None:
                number_results[number_folder] = values
        
        if not number_results:
            print(f"    No valid data for plotting")
            continue
        
        # ---- STEP 4: Plot all number folders for this method-animal combination ----
        x = [0, 1, 2]
        x_labels = ["agent0", "agent1", "agent2"]
        
        plt.figure(figsize=(10, 6))
        
        # Plot each number folder as a separate line
        for number_folder in sorted(number_results.keys()):
            values = number_results[number_folder]
            plt.plot(
                x,
                values,
                marker="o",
                linewidth=2,
                label=number_folder
            )
        
        plt.xticks(x, x_labels)
        plt.ylabel("Average value")
        plt.title(f"Method: {method} | Animal: {animal}")
        _, top = plt.ylim()
        plt.ylim(0, top)
        plt.legend(title="Number", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Create filename with method and animal
        output_path = os.path.join(OUTPUT_DIR, f"{method}_{animal}_all_numbers.png")
        plt.savefig(output_path, dpi=100)
        plt.close()
        
        print(f"    Saved plot → {output_path} ({len(number_results)} numbers)")

print("\n✓ All plots generated!")