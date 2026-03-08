#!/usr/bin/env python3
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np
import sys

# Set up paths based on your previous messages
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIR_DARK = PROJECT_ROOT / "data/scenes/syns_meg36"
DIR_BRIGHT = PROJECT_ROOT / "data/scenes/syns_meg36_real"

def check_folder_activations(folder_path, model, transform, device, num_images=10):
    if not folder_path.exists():
        print(f"Directory not found: {folder_path}")
        return None

    # Get a few images to test
    image_files = list(folder_path.glob("*.jpg"))[:num_images]
    if not image_files:
        print(f"No images found in {folder_path}")
        return None

    # We will track the percentage of zeros and average signal strength
    stats = {
        "early": {"zeros": [], "mean": []},
        "late": {"zeros": [], "mean": []}
    }

    # Hooks to grab the math as it flows through the network
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach().cpu().numpy()
        return hook

    h1 = model.features[2].register_forward_hook(get_activation('early')) # After first ReLU
    h2 = model.features[12].register_forward_hook(get_activation('late')) # Last conv layer ReLU

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(device)
            
            _ = model(img_tensor) # Push image through
            
            for layer in ['early', 'late']:
                act = activations[layer]
                # Calculate what percentage of the matrix is exactly 0.0
                percent_dead = (np.count_nonzero(act == 0) / act.size) * 100
                mean_signal = np.mean(act)
                
                stats[layer]["zeros"].append(percent_dead)
                stats[layer]["mean"].append(mean_signal)

    h1.remove()
    h2.remove()

    # Average the stats across the tested images
    return {
        "early_zeros": np.mean(stats["early"]["zeros"]),
        "early_mean": np.mean(stats["early"]["mean"]),
        "late_zeros": np.mean(stats["late"]["zeros"]),
        "late_mean": np.mean(stats["late"]["mean"]),
    }

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading standard AlexNet...")
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    print(f"\n--- Testing DARK Images ({DIR_DARK.name}) ---")
    dark_stats = check_folder_activations(DIR_DARK, model, transform, device)
    if dark_stats:
        print(f"Early Layer (features_2) -> Dead Neurons: {dark_stats['early_zeros']:.1f}% | Signal Strength: {dark_stats['early_mean']:.4f}")
        print(f"Late Layer (features_12) -> Dead Neurons: {dark_stats['late_zeros']:.1f}% | Signal Strength: {dark_stats['late_mean']:.4f}")

    print(f"\n--- Testing BRIGHT Images ({DIR_BRIGHT.name}) ---")
    bright_stats = check_folder_activations(DIR_BRIGHT, model, transform, device)
    if bright_stats:
        print(f"Early Layer (features_2) -> Dead Neurons: {bright_stats['early_zeros']:.1f}% | Signal Strength: {bright_stats['early_mean']:.4f}")
        print(f"Late Layer (features_12) -> Dead Neurons: {bright_stats['late_zeros']:.1f}% | Signal Strength: {bright_stats['late_mean']:.4f}")

if __name__ == "__main__":
    main()