import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path

# choose the model
MODEL_NAME = "resnet50"

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RDM_DIR = PROJECT_ROOT / "outputs/clean_baseline/rdms" / MODEL_NAME
SAVE_PATH = PROJECT_ROOT / f"outputs/clean_baseline/{MODEL_NAME}_rdms.png"

# Define layers for each model
MODELS = {
    "alexnet": ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"],
    "resnet50": ["layer1", "layer2", "layer3", "layer4"]
}

# Select layers based on the chosen model
LAYERS = MODELS[MODEL_NAME]

def main():
    # determine the number of subplots and create figure
    n_cols = 4
    n_rows = int(np.ceil(len(LAYERS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8,6)) 
    
    # plot the RDMs for each layer
    for i, layer in enumerate(LAYERS):
        safe_layer = layer.replace(".", "_").replace("/", "_")
        rdm_file = RDM_DIR / f"{safe_layer}_rdm.npy"
        if rdm_file.exists():
            rdm = np.load(rdm_file)
            ax = axes[i // n_cols, i % n_cols] if n_rows > 1 else axes[i % n_cols]
            im = ax.imshow(rdm, cmap='viridis')
            ax.set_xticks([])
            ax.set_yticks([])
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title(layer)
        else:
            print(f"Missing files for: {layer}. Make sure you ran the compute RDMs script!")

    plt.suptitle(f"RDMs for each layer of {MODEL_NAME}")
    plt.tight_layout()
    fig.savefig(SAVE_PATH)
    plt.show()

if __name__ == "__main__":
    main()