#!/usr/bin/env python3
import torch
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIR_DARK = PROJECT_ROOT / "data/scenes/syns_meg36"
DIR_BRIGHT = PROJECT_ROOT / "data/scenes/syns_meg36_real"

def get_stats(tensor, model, name):
    activations = {}
    def hook(model, input, output):
        activations['act'] = output.detach().cpu().numpy()
    
    # We check features[2] (Early) and features[12] (Late)
    layer = model.features[2] if name == "early" else model.features[12]
    handle = layer.register_forward_hook(hook)
    
    with torch.no_grad():
        _ = model(tensor)
    
    handle.remove()
    act = activations['act']
    return np.mean(act), (np.count_nonzero(act == 0) / act.size) * 100

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1).to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(), 
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 1. Create Synthetic ImageNet Baseline (The "Perfect" Math)
    # randn gives mean 0, std 1, which mimics a perfectly normalized ImageNet image.
    torch.manual_seed(42)
    fake_img = torch.randn(1, 3, 224, 224).to(device)
    base_e_m, base_e_z = get_stats(fake_img, model, "early")
    base_l_m, base_l_z = get_stats(fake_img, model, "late")

    print("\n" + "="*60)
    print(f"{'IMAGE SET':<25} | {'LAYER':<8} | {'MEAN SIG':<10} | {'% DEAD':<8}")
    print("-" * 60)
    print(f"{'1. ImageNet (Synthetic)':<25} | {'Early':<8} | {base_e_m:<10.4f} | {base_e_z:<8.1f}%")
    print(f"{'':<25} | {'Late':<8} | {base_l_m:<10.4f} | {base_l_z:<8.1f}%")
    print("-" * 60)

    # 2. Check Bright vs Dark
    for label, folder in [("2. Bright MEG (Real)", DIR_BRIGHT), ("3. Dark MEG (Scanner)", DIR_DARK)]:
        img_path = next(folder.glob("*.jpg")) # Test the first image found
        img = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        
        e_m, e_z = get_stats(img, model, "early")
        l_m, l_z = get_stats(img, model, "late")
        
        print(f"{label:<25} | {'Early':<8} | {e_m:<10.4f} | {e_z:<8.1f}%")
        print(f"{'':<25} | {'Late':<8} | {l_m:<10.4f} | {l_z:<8.1f}%")
        print("-" * 60)

if __name__ == "__main__":
    main()