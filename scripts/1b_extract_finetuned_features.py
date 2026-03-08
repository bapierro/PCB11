#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import numpy as np
import pandas as pd
import csv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs/finetuned_behaviour_alexnet.pth"
TEST_IMG_DIR = PROJECT_ROOT / "data/scenes/syns_meg36_real"
STIMULUS_ORDER_CSV = PROJECT_ROOT / "data/meg/stimulus_order.csv"
LABELS_CSV = PROJECT_ROOT / "data/behaviour/consensus_labels.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs/finetuned_behaviour/features/alexnet"

class MultiTaskAlexNet(nn.Module):
    def __init__(self, num_app, num_sem, num_str):
        super().__init__()
        base_model = models.alexnet(weights=None)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        self.shared_classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])
        in_features = 4096
        self.head_app = nn.Linear(in_features, num_app)
        self.head_sem = nn.Linear(in_features, num_sem)
        self.head_str = nn.Linear(in_features, num_str)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared_classifier(x)
        return self.head_app(x), self.head_sem(x), self.head_str(x)

def get_strict_image_order():
    """Reads the exact file names from the CSV since the data is now perfectly clean."""
    if not STIMULUS_ORDER_CSV.exists():
        raise FileNotFoundError(f"CRITICAL: Missing {STIMULUS_ORDER_CSV}")
        
    order = []
    with open(STIMULUS_ORDER_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            order.append(row['file_name'].strip())
            
    return order

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(LABELS_CSV)
    model = MultiTaskAlexNet(df.Appearance_Category.max(), df.Semantic_Category.max(), df.Structure_Category.max()).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    layers_to_extract = {
        "features_2": model.features[2], "features_5": model.features[5],
        "features_7": model.features[7], "features_9": model.features[9],
        "features_12": model.features[12], "shared_classifier_2": model.shared_classifier[2],
        "shared_classifier_5": model.shared_classifier[5]
    }

    activations = {name: [] for name in layers_to_extract.keys()}
    activations["head_app"] = []
    activations["head_sem"] = []
    activations["head_str"] = []
    
    def get_activation(name):
        def hook(model, input, output):
            pooled = F.adaptive_avg_pool2d(output, (1, 1)).squeeze() if output.ndim == 4 else output.squeeze()
            activations[name].append(pooled.detach().cpu().numpy())
        return hook

    handles = [layer.register_forward_hook(get_activation(name)) for name, layer in layers_to_extract.items()]
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    target_filenames = get_strict_image_order()
    print(f"Extracting 10 layers for {len(target_filenames)} images...")

    with open(OUTPUT_DIR / "file_names.txt", "w") as f:
        with torch.no_grad():
            for filename in target_filenames:
                img = Image.open(TEST_IMG_DIR / filename).convert("RGB")
                out_app, out_sem, out_str = model(transform(img).unsqueeze(0).to(device))
                
                # Save the 3 parallel heads separately
                activations["head_app"].append(out_app.detach().cpu().numpy().squeeze())
                activations["head_sem"].append(out_sem.detach().cpu().numpy().squeeze())
                activations["head_str"].append(out_str.detach().cpu().numpy().squeeze())
                f.write(f"{filename}\n")

    for handle in handles: handle.remove()

    for name, acts in activations.items():
        np.save(OUTPUT_DIR / f"alexnet_{name}.npy", np.stack(acts))
        print(f" - Saved {name}.npy")

if __name__ == "__main__":
    main()