#!/usr/bin/env python3
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from pcb11.features import FeatureConfig, extract_features

# --- 1. PICK YOUR MODEL HERE ---
MODEL_NAME = "alexnet"  # Change to "alexnet" when you want to run the other model

# 2. The script automatically looks up the correct layers
MODELS = {
    "alexnet": ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"],
    "resnet50": ["layer1", "layer2", "layer3", "layer4"]
}

LAYERS = MODELS[MODEL_NAME]
IMAGES_DIR = PROJECT_ROOT / "data/scenes/syns_meg36"
OUTPUT_DIR = PROJECT_ROOT / "outputs/clean_baseline/features" / MODEL_NAME

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu" # Change to "cuda" if you have a GPU
    print(f"\n=== Starting Extraction for {MODEL_NAME.upper()} ===")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    config = FeatureConfig(
        image_path=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        source="torchvision",
        device=device,
        pretrained=True,
        layers=LAYERS,
        pool_mode="gap",
        batch_size=32,
    )
    extract_features(config)

if __name__ == "__main__":
    main()