#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import sys

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Input/Output paths
INPUT_FILE = PROJECT_ROOT / "data/behaviour/CategoryValidationExpt.txt"
OUTPUT_FILE = PROJECT_ROOT / "data/behaviour/consensus_labels.csv"

def main():
    print("Loading behaviour data...")
    # Read the raw experimental data
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    # Read CSV. We assume standard comma separation based on the file snippet.
    df = pd.read_csv(INPUT_FILE)
    
    # Strip whitespace from column names just in case
    df.columns = df.columns.str.strip()
    
    print(f"Raw data: {len(df)} rows")
    
    # We want one label per (SYNSscene, SYNSView) pair based on majority vote (mode)
    # We group by scene and view, then find the mode of the category columns.
    
    # Helper to get the first mode (handles ties by picking the first one)
    def compute_mode(x):
        return x.mode().iloc[0] if not x.mode().empty else None

    print("Computing consensus (mode) labels per task...")
    
    # Group by task, scene, and view to separate dimensions
    grouped = df.groupby(['task', 'SYNSscene', 'SYNSView'])[['categorySelected', 'categoryLabelSelected']]
    consensus_long = grouped.agg(compute_mode).reset_index()
    
    # Clean strings
    if 'categoryLabelSelected' in consensus_long.columns:
        consensus_long['categoryLabelSelected'] = (
            consensus_long['categoryLabelSelected']
            .astype(str)
            .str.replace(r'[\r\n]+', ' ', regex=True)
            .str.replace(r'\s+', ' ', regex=True)
            .str.strip()
        )

    # Pivot to wide format: One row per image (Scene/View), columns for each task
    # This creates a table where each image has columns for Structure, Semantic, Appearance, etc.
    pivot_df = consensus_long.pivot(index=['SYNSscene', 'SYNSView'], columns='task', values=['categorySelected', 'categoryLabelSelected'])
    
    # Flatten column names from MultiIndex (Measure, Task) -> Task_Measure
    # e.g. ('categorySelected', 'Structure') -> 'Structure_Category'
    new_columns = []
    for val_type, task_name in pivot_df.columns:
        suffix = "Category" if val_type == "categorySelected" else "Label"
        new_columns.append(f"{task_name}_{suffix}")
    
    pivot_df.columns = new_columns
    pivot_df = pivot_df.reset_index()
    
    print(f"Consensus data: {len(pivot_df)} unique scene/view pairs")
    print(f"Columns generated: {list(pivot_df.columns)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    pivot_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved consensus dataframe to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
