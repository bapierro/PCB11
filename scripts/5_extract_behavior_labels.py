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
        # x.mode() returns a Series of the most frequent values. 
        # Taking .iloc[0] selects the first one if there are ties.
        return x.mode().iloc[0] if not x.mode().empty else None

    print("Computing consensus (mode) labels...")
    # Group by scene and view, aggregating label and ID by mode
    consensus_df = df.groupby(['SYNSscene', 'SYNSView'])[['categorySelected', 'categoryLabelSelected']].agg(compute_mode).reset_index()
    
    # Clean up the label strings (remove newlines and extra spaces)
    # The raw file has labels like "Tunnel/ \n Navigable\n Routes"
    if 'categoryLabelSelected' in consensus_df.columns:
        consensus_df['categoryLabelSelected'] = (
            consensus_df['categoryLabelSelected']
            .astype(str)
            .str.replace(r'[\r\n]+', ' ', regex=True) # Replace newlines with space
            .str.replace(r'\s+', ' ', regex=True)     # Collapse multiple spaces
            .str.strip()
        )

    print(f"Consensus data: {len(consensus_df)} unique scene/view pairs")
    
    # Save the result
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    consensus_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved consensus dataframe to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
