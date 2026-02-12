#!/usr/bin/env python3
import sys
from pathlib import Path
import torch

# Set up the project root and add 'src' to the system path to import custom modules
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

# Select layers based on the chosen model
LAYERS = MODELS[MODEL_NAME]

# Define paths for input images and where to save the extracted features
IMAGES_DIR = PROJECT_ROOT / "data/scenes/syns_meg36"
OUTPUT_DIR = PROJECT_ROOT / "outputs/clean_baseline/features" / MODEL_NAME

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu" # Change to "cuda" if you have a GPU
    print(f"\n=== Starting Extraction for {MODEL_NAME.upper()} ===")
    
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Configure the feature extraction parameters
    config = FeatureConfig(
        image_path=IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        model_name=MODEL_NAME,
        source="torchvision",
        device=device,
        pretrained=True,        # Use weights pretrained on ImageNet
        layers=LAYERS,          # Which layers to extract
        pool_mode="gap",        # Global Average Pooling for convolutional layers
        batch_size=32,
    )
    
    # Run the feature extraction pipeline
    extract_features(config)

if __name__ == "__main__":
    main()