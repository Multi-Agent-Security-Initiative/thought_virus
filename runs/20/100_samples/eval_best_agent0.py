import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---- CONFIG ----
CSV_NAME = "subliminal_frequencies.csv"

ROOT_DIR = "."  # repo root (change if needed)
OUTPUT_DIR = "plots"

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


# ---- STEP 2: Discover all animals across all methods ----
all_animals = set()
for method in methods:
    method_path = os.path.join(ROOT_DIR, method)
    animals = [
        d for d in os.listdir(method_path)
        if os.path.isdir(os.path.join(method_path, d))
    ]
    all_animals.update(animals)

print(f"Found animals: {sorted(all_animals)}")


# ---- STEP 3: Process each animal ----
for animal in sorted(all_animals):
    print(f"\nProcessing animal: {animal}")
    
    animal_data = {}  # method_name -> best_values
    
    # For each method, find this animal and compute best result
    for method in methods:
        animal_path = os.path.join(ROOT_DIR, method, animal)
        
        if not os.path.exists(animal_path):
            print(f"  Animal '{animal}' not found in method '{method}'")
            continue
        
        # Find all number subfolders
        number_folders = [
            d for d in os.listdir(animal_path)
            if os.path.isdir(os.path.join(animal_path, d))
        ]
        
        if not number_folders:
            print(f"  No number folders in {animal_path}")
            continue
        
        # Dictionary to store average for each number folder
        number_results = {}
        
        # Process each number subfolder
        for number_folder in number_folders:
            csv_path = os.path.join(animal_path, number_folder, CSV_NAME)
            
            if not os.path.exists(csv_path):
                continue
            
            values = compute_column_means(csv_path, animal)
            if values is not None:
                number_results[number_folder] = values
        
        if not number_results:
            print(f"  Method '{method}': no valid data")
            continue
        
        # Find the best number folder (highest agent0 value)
        best_number = None
        best_agent0 = -np.inf
        
        for number_folder, values in number_results.items():
            agent0_value = values[0]  # agent0 is the first element
            if agent0_value > best_agent0:
                best_agent0 = agent0_value
                best_number = number_folder
        
        animal_data[method] = number_results[best_number]
        print(f"  Method '{method}': best number folder = {best_number} (agent0 = {best_agent0:.4f})")
    
    # ---- STEP 4: Plot for this animal ----
    if not animal_data:
        print(f"  No data to plot for animal '{animal}'")
        continue
    
    x = [0, 1, 2]
    x_labels = ["agent0", "agent1", "agent2"]
    
    plt.figure(figsize=(8, 5))
    
    for method, values in animal_data.items():
        plt.plot(
            x,
            values,
            marker="o",
            linewidth=2,
            label=method
        )
    
    plt.xticks(x, x_labels)
    plt.ylabel("Average value")
    plt.title(f"Animal: {animal} - Best Results by Method (highest agent0)")
    _, top = plt.ylim()
    plt.ylim(0, top)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f"{animal}_best_agent0_method_comparison.png")
    plt.savefig(output_path, dpi=100)
    plt.close()
    
    print(f"  Saved plot → {output_path}")

print("\n✓ All plots generated!")