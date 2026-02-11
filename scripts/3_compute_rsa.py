#!/usr/bin/env python3
import sys
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.stats import rankdata

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- 1. PICK YOUR MODEL HERE ---
MODEL_NAME = "alexnet"  # Change to "alexnet" when you want to run the other model

MODELS = {
    "alexnet": ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"],
    "resnet50": ["layer1", "layer2", "layer3", "layer4"]
}

LAYERS = MODELS[MODEL_NAME]
RDM_DIR = PROJECT_ROOT / "outputs/clean_baseline/rdms" / MODEL_NAME
OUTPUT_DIR = PROJECT_ROOT / "outputs/clean_baseline/rsa" / MODEL_NAME
MEG_FILE = PROJECT_ROOT / "data/meg/MEGRDMs_2D.mat"

def compute_correlation(layer_rdms, meg_rdms):
    n_images = meg_rdms.shape[0]
    n_time = meg_rdms.shape[2]
    tri = np.triu_indices(n_images, k=1)
    
    meg_vectors = np.empty((n_time, tri[0].size), dtype=np.float64)
    for t in range(n_time):
        meg_vectors[t] = meg_rdms[:, :, t][tri]
        
    meg_ranked = np.array([rankdata(row) for row in meg_vectors])
    meg_centered = meg_ranked - meg_ranked.mean(axis=1, keepdims=True)
    meg_norms = np.linalg.norm(meg_centered, axis=1)
    meg_norms[meg_norms == 0] = np.nan

    out = np.empty((len(layer_rdms), n_time), dtype=np.float64)
    for idx, model_rdm in enumerate(layer_rdms):
        model_vec = model_rdm[tri]
        model_ranked = rankdata(model_vec)
        model_centered = model_ranked - model_ranked.mean()
        model_norm = np.linalg.norm(model_centered)
        
        if model_norm == 0:
            out[idx] = np.nan
        else:
            out[idx] = (meg_centered @ model_centered) / (meg_norms * model_norm)
    return out

def main():
    print(f"\n=== Computing RSA for {MODEL_NAME.upper()} ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading MEG brain data...")
    mat = loadmat(MEG_FILE, squeeze_me=True)
    meg_data = np.asarray(mat["MEGRDMs_2D"], dtype=np.float64)
    if meg_data.ndim == 4:
        meg_data = meg_data.mean(axis=3)

    layer_rdms = []
    for layer in LAYERS:
        safe_layer = layer.replace(".", "_").replace("/", "_")
        layer_rdms.append(np.load(RDM_DIR / f"{safe_layer}_rdm.npy"))

    rsa_results = compute_correlation(layer_rdms, meg_data)

    for idx, layer in enumerate(LAYERS):
        safe_layer = layer.replace(".", "_").replace("/", "_")
        np.save(OUTPUT_DIR / f"{safe_layer}_rsa_spearman.npy", rsa_results[idx])
        print(f" - Saved RSA data for {safe_layer}")

if __name__ == "__main__":
    main()