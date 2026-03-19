#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.stats import rankdata

# Set up project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 1. PICK YOUR MODEL HERE ---
print("\n--- MODEL SELECTION ---")
print("1: AlexNet")
print("2: ResNet50")
model_choice = input("Enter 1 or 2: ").strip()
MODEL_NAME = "alexnet" if model_choice == '1' else "resnet50"

# --- 2. INTERACTIVE PIPELINE SELECTOR ---
print("\nWhich pipeline are we running RSA for?")
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
        LAYERS = ["features_2", "features_5", "features_7", "features_9", "features_12", "classifier_2", "classifier_5", "classifier_6"]
    elif MODEL_NAME == "resnet50":
        LAYERS = ["layer1", "layer2", "layer3", "layer4"]

else:
    PIPELINE = "clean_baseline"
    if MODEL_NAME == "alexnet":
        LAYERS = ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"]
    elif MODEL_NAME == "resnet50":
        LAYERS = ["layer1", "layer2", "layer3", "layer4"]
        
# Define paths: location of computed RDMs, output for RSA results, MEG data source and 
# three behavioral model RDMs
RDM_DIR = PROJECT_ROOT / f"outputs/{PIPELINE}/rdms" / MODEL_NAME
OUTPUT_DIR = PROJECT_ROOT / f"outputs/{PIPELINE}/rsa" / MODEL_NAME
MEG_FILE = PROJECT_ROOT / "data/meg/MEGRDMs_2D.mat"
SEMANTIC_FILE = PROJECT_ROOT / "data/meg/semanticRDM_sq.mat"
STRUCTURE_FILE = PROJECT_ROOT / "data/meg/structureRDM_sq.mat"
VISUAL_FILE = PROJECT_ROOT / "data/meg/visualRDM_sq.mat"

def compute_correlation(layer_rdms, meg_rdms):
    """
    Computes Spearman correlation between model RDMs and MEG RDMs over time (or the three 
    behavioral model RDMs).
    Spearman correlation is Pearson correlation of the ranked data.
    """
    n_images = meg_rdms.shape[0]
    try:
        n_time = meg_rdms.shape[2]
    except IndexError:
        n_time = 1  # If there's no time dimension, assume single time point
        
    # Get upper triangular indices to extract unique pairwise dissimilarities
    tri = np.triu_indices(n_images, k=1)
    
    # Pre-process MEG data: flatten upper triangle for each time point
    if n_time != 1:
        meg_vectors = np.empty((n_time, tri[0].size), dtype=np.float64)
        for t in range(n_time):
            meg_vectors[t] = meg_rdms[:, :, t][tri]
    else:
        meg_vectors = meg_rdms[tri].reshape(1, -1)  
        
    # Rank transform MEG data for Spearman correlation
    meg_ranked = np.array([rankdata(row) for row in meg_vectors])
    meg_centered = meg_ranked - meg_ranked.mean(axis=1, keepdims=True)
    meg_norms = np.linalg.norm(meg_centered, axis=1)
    meg_norms[meg_norms == 0] = np.nan

    out = np.empty((len(layer_rdms), n_time), dtype=np.float64)
    for idx, model_rdm in enumerate(layer_rdms):
        # Flatten and rank transform model RDM
        model_vec = model_rdm[tri]
        model_ranked = rankdata(model_vec)
        model_centered = model_ranked - model_ranked.mean()
        model_norm = np.linalg.norm(model_centered)
        
        # Calculate correlation for this model layer against all MEG timepoints efficiently
        if model_norm == 0:
            out[idx] = np.nan
        else:
            out[idx] = (meg_centered @ model_centered) / (meg_norms * model_norm)
    return out

def main():
    print(f"\n=== Computing RSA for {MODEL_NAME.upper()} (Subject-Level) ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading MEG brain data...")
    mat = loadmat(MEG_FILE, squeeze_me=True)
    meg_data = np.asarray(mat["MEGRDMs_2D"], dtype=np.float64)
    
    # Load model RDMs computed in the previous step
    layer_rdms = []
    for layer in LAYERS:
        safe_layer = layer.replace(".", "_").replace("/", "_")
        layer_rdms.append(np.load(RDM_DIR / f"{safe_layer}_rdm.npy"))

    # --- NEW LOGIC: Loop through subjects instead of averaging early ---
    if meg_data.ndim == 4:
        n_subjects = meg_data.shape[3]
        n_time = meg_data.shape[2]
        print(f"Found {n_subjects} subjects. Computing individual RSAs...")
        
        # Prepare matrix to hold everyone's results: Shape (Layers, Time, Subjects)
        all_subjects_rsa = np.zeros((len(LAYERS), n_time, n_subjects))
        
        for s in range(n_subjects):
            print(f" - Processing Subject {s+1}/{n_subjects}...", end="\r")
            subject_meg = meg_data[:, :, :, s]
            all_subjects_rsa[:, :, s] = compute_correlation(layer_rdms, subject_meg)
            
        print("\nFinished computing individual RSAs.")
        
        # Now we average the RSA correlations across subjects
        mean_rsa = all_subjects_rsa.mean(axis=2)
        
        # Save results for each layer
        for idx, layer in enumerate(LAYERS):
            safe_layer = layer.replace(".", "_").replace("/", "_")
            
            # Save the average line (for the main plot)
            np.save(OUTPUT_DIR / f"{safe_layer}_rsa_spearman_mean.npy", mean_rsa[idx])
            
            # Save the individual subject data (vital for calculating standard error / shaded bars!)
            np.save(OUTPUT_DIR / f"{safe_layer}_rsa_spearman_subjects.npy", all_subjects_rsa[idx])
            
            print(f" - Saved Mean and Subject data for {safe_layer}")
            
    else:
        print("Data is already 3D. Computing single RSA...")
        rsa_results = compute_correlation(layer_rdms, meg_data)
        for idx, layer in enumerate(LAYERS):
            safe_layer = layer.replace(".", "_").replace("/", "_")
            np.save(OUTPUT_DIR / f"{safe_layer}_rsa_spearman.npy", rsa_results[idx])
            print(f" - Saved RSA data for {safe_layer}")

    # compute RSA for the three behavioral models as well
    print("\nComputing RSA for behavioral models...")
    for model_name, model_file in [("semantic", SEMANTIC_FILE), ("structure", STRUCTURE_FILE), ("visual", VISUAL_FILE)]:
        model_mat = loadmat(model_file, squeeze_me=True, struct_as_record=False)
        rdm_struct = model_mat['RDM']
        model_rdm = np.asarray(rdm_struct.RDM, dtype=np.float64)
        
        # Compute RSA for this behavioral model
        rsa_results = compute_correlation(layer_rdms, model_rdm)
        np.save(OUTPUT_DIR / f"{model_name}_rsa_spearman.npy", rsa_results)
        print(f" - Saved RSA data for {model_name} model.")

if __name__ == "__main__":
    main()