#!/usr/bin/env python3
import csv
import sys
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from thingsvision.core.rsa import compute_rdm

# --- 1. PICK YOUR MODEL HERE ---
MODEL_NAME = "alexnet"  # Change to "alexnet" when you want to run the other model

MODELS = {
    "alexnet": ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"],
    "resnet50": ["layer1", "layer2", "layer3", "layer4"]
}

LAYERS = MODELS[MODEL_NAME]
FEATURES_DIR = PROJECT_ROOT / "outputs/clean_baseline/features" / MODEL_NAME
OUTPUT_DIR = PROJECT_ROOT / "outputs/clean_baseline/rdms" / MODEL_NAME
STIMULUS_ORDER_CSV = PROJECT_ROOT / "data/meg/stimulus_order.csv"

def get_strict_image_order():
    if not STIMULUS_ORDER_CSV.exists():
        raise FileNotFoundError(f"CRITICAL: Missing {STIMULUS_ORDER_CSV}")
    order = []
    with open(STIMULUS_ORDER_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            order.append(row['file_name'].strip())
    return order

def main():
    print(f"\n=== Computing RDMs for {MODEL_NAME.upper()} ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_order = get_strict_image_order()
    
    with open(FEATURES_DIR / "file_names.txt", "r") as f:
        extracted_order = [Path(line.strip()).name for line in f if line.strip()]
    reorder_idx = [extracted_order.index(name) for name in target_order]

    for layer in LAYERS:
        safe_layer = layer.replace(".", "_").replace("/", "_")
        feature_file = FEATURES_DIR / f"{MODEL_NAME}_{safe_layer}.npy"
        features = np.load(feature_file)
        
        rdm = compute_rdm(features, method="correlation")
        rdm_reordered = rdm[np.ix_(reorder_idx, reorder_idx)]
        
        np.save(OUTPUT_DIR / f"{safe_layer}_rdm.npy", rdm_reordered)
        print(f" - {safe_layer} RDM saved.")

if __name__ == "__main__":
    main()