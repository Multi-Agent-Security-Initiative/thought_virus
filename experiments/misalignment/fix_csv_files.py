"""
Fix corrupted CSV files by consolidating duplicate rows.

For each CSV file, if there are multiple rows with the same index,
this script will merge them into a single row by taking the first
non-NaN value for each column.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


def fix_csv_file(csv_path):
    """Fix a single CSV file by consolidating duplicate indices."""
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path, index_col=0)

        # Check if there are duplicate indices
        if df.index.duplicated().any():
            print(f"  Found duplicates in {csv_path}")
            print(f"    Original shape: {df.shape}")

            # Group by index and take the first non-NaN value for each column
            # This works because each index-column combination should only have one value
            df_fixed = df.groupby(df.index).first()

            # Alternative: use combine_first to merge all duplicate rows
            # This ensures we get all non-NaN values even if they're spread across rows
            unique_indices = df.index.unique()
            rows = []
            for idx in unique_indices:
                # Get all rows with this index
                subset = df.loc[[idx]]
                if len(subset) > 1:
                    # Combine all rows by taking first non-NaN value
                    combined = subset.iloc[0].copy()
                    for i in range(1, len(subset)):
                        combined = combined.combine_first(subset.iloc[i])
                    rows.append(combined)
                else:
                    rows.append(subset.iloc[0])

            df_fixed = pd.DataFrame(rows, index=unique_indices)

            print(f"    Fixed shape: {df_fixed.shape}")
            print(f"    Removed {len(df) - len(df_fixed)} duplicate rows")

            # Create backup
            backup_path = str(csv_path) + '.backup'
            if not os.path.exists(backup_path):
                os.rename(str(csv_path), backup_path)
                print(f"    Created backup: {backup_path}")

            # Save fixed version
            df_fixed.to_csv(csv_path)
            print(f"    Saved fixed CSV to {csv_path}")

            return True
        else:
            print(f"  No duplicates found in {csv_path}")
            return False

    except Exception as e:
        print(f"  Error processing {csv_path}: {e}")
        return False


def main():
    # Find all CSV files in the results directory
    results_dir = Path(__file__).parent / "results"

    csv_files = []
    for csv_path in results_dir.rglob("*.csv"):
        if csv_path.name in ["accuracy_rates_unidirectional.csv", "logit_diff_unidirectional.csv"]:
            csv_files.append(csv_path)

    print(f"Found {len(csv_files)} CSV files to check")
    print()

    fixed_count = 0
    for csv_path in sorted(csv_files):
        print(f"Processing {csv_path.relative_to(results_dir.parent)}...")
        if fix_csv_file(csv_path):
            fixed_count += 1
        print()

    print(f"\nSummary: Fixed {fixed_count} out of {len(csv_files)} CSV files")


if __name__ == "__main__":
    main()
