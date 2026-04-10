#!/usr/bin/env python3
import csv
import sys
import numpy as np
from pathlib import Path

# Set up project root and import custom rsa tools
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from thingsvision.core.rsa import compute_rdm

# --- 1. PICK YOUR MODEL HERE ---
print("\n--- MODEL SELECTION ---")
print("1: AlexNet")
print("2: ResNet50")
print("3: CORnet-S")
model_choice = input("Enter 1, 2, or 3: ").strip()
MODEL_NAME = "alexnet" if model_choice == '1' else "resnet50" if model_choice == '2' else "cornet_s"

# --- 2. INTERACTIVE PIPELINE SELECTOR ---
print("\nWhich pipeline are we running?")
print("1: Clean Baseline (ImageNet)")
print("2: Fine-Tuned (All/Multi-Task)")
print("3: Fine-Tuned (Appearance)")
print("4: Fine-Tuned (Semantic)")
print("5: Fine-Tuned (Structure)")
choice = input("Enter 1, 2, 3, 4, or 5: ").strip()

if choice == '2':
    PIPELINE = "finetuned_all"
    if MODEL_NAME == "alexnet":
        LAYERS = [
            "features_2", "features_5", "features_7", "features_9", "features_12", 
            "shared_classifier_2", "shared_classifier_5", 
            "head_app", "head_sem", "head_str"
        ]
    elif MODEL_NAME == "resnet50":
        LAYERS = ["layer1", "layer2", "layer3", "layer4", "head_app", "head_sem", "head_str"]

elif choice in ['3', '4', '5']:
    task_map = {'3': 'appearance', '4': 'semantic', '5': 'structure'}
    PIPELINE = f"finetuned_{task_map[choice]}"
    
    if MODEL_NAME == "alexnet":
        # Using underscores because script 1b saved them with underscores
        LAYERS = ["features_2", "features_5", "features_7", "features_9", "features_12", "classifier_2", "classifier_5", "classifier_6"]
    elif MODEL_NAME == "resnet50":
        # Kept strictly to the backbone layers, exactly matching the baseline!
        LAYERS = ["layer1", "layer2", "layer3", "layer4"]

else:
    PIPELINE = "clean_baseline"
    if MODEL_NAME == "alexnet":
        LAYERS = ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"]
    elif MODEL_NAME == "resnet50":
        LAYERS = ["layer1", "layer2", "layer3", "layer4"]
    elif MODEL_NAME == "cornet_s":
        LAYERS = ["V1", "V2", "V4", "IT"]

# Define paths: location of extracted features and where to save RDMs (Dynamic based on menu)
FEATURES_DIR = PROJECT_ROOT / f"outputs/{PIPELINE}/features" / MODEL_NAME
OUTPUT_DIR = PROJECT_ROOT / f"outputs/{PIPELINE}/rdms" / MODEL_NAME
STIMULUS_ORDER_CSV = PROJECT_ROOT / "data/meg/stimulus_order.csv"

def get_strict_image_order():
    """Reads the stimulus file to list images in the exact order used in the experiment."""
    if not STIMULUS_ORDER_CSV.exists():
        raise FileNotFoundError(f"CRITICAL: Missing {STIMULUS_ORDER_CSV}")
    order = []
    with open(STIMULUS_ORDER_CSV, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # No more regex! Just read the exact cleaned name from the CSV.
            order.append(row['file_name'].strip())
    return order

def main():
    print(f"\n=== Computing RDMs for {MODEL_NAME.upper()} ({PIPELINE}) ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get the target order from the CSV file
    target_order = get_strict_image_order()
    # Read the order of images as they were processed/extracted
    with open(FEATURES_DIR / "file_names.txt", "r") as f:
        extracted_order = [Path(line.strip()).name for line in f if line.strip()]
    # Compute indices to reorder extracted features to match the target order
    reorder_idx = [extracted_order.index(name) for name in target_order]

    for layer in LAYERS:
        # Load features for the specific layer
        safe_layer = layer.replace(".", "_").replace("/", "_")
        feature_file = FEATURES_DIR / f"{MODEL_NAME}_{safe_layer}.npy"
        
        if not feature_file.exists():
            print(f"[!] Warning: Could not find {feature_file.name}. Skipping.")
            continue
            
        features = np.load(feature_file)
        
        # Calculate Representational Dissimilarity Matrix (RDM) using correlation distance
        rdm = compute_rdm(features, method="correlation")
        
        # Reorder the RDM rows/cols to match the experimental order
        rdm_reordered = rdm[np.ix_(reorder_idx, reorder_idx)]
        
        # Save the result
        np.save(OUTPUT_DIR / f"{safe_layer}_rdm.npy", rdm_reordered)
        print(f" - {safe_layer} RDM saved.")

if __name__ == "__main__":
    main()