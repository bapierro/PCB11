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

TEST_IMG_DIR = PROJECT_ROOT / "data/scenes/syns_meg36_real"
STIMULUS_ORDER_CSV = PROJECT_ROOT / "data/meg/stimulus_order.csv"
LABELS_CSV = PROJECT_ROOT / "data/behaviour/consensus_labels.csv"

# --- Custom Architectures (Used only if TARGET_TASK == 'all') ---
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
    
class MultiTaskResNet50(nn.Module):
    def __init__(self, num_app, num_sem, num_str):
        super().__init__()
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        in_features = base_model.fc.in_features  # 2048
        
        self.head_app = nn.Linear(in_features, num_app)
        self.head_sem = nn.Linear(in_features, num_sem)
        self.head_str = nn.Linear(in_features, num_str)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        out_app = self.head_app(x)
        out_sem = self.head_sem(x)
        out_str = self.head_str(x)
        return out_app, out_sem, out_str

def get_strict_image_order():
    """Reads the exact file names from the CSV since the data is now perfectly clean."""
    if not STIMULUS_ORDER_CSV.exists():
        raise FileNotFoundError(f"CRITICAL: Missing {STIMULUS_ORDER_CSV}")
        
    order = []
    with open(STIMULUS_ORDER_CSV, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            order.append(row['file_name'].strip())
            
    return order


def get_best_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def main():
    # --- 1. INTERACTIVE MENU ---
    print("\n--- MODEL SELECTION ---")
    print("1: AlexNet")
    print("2: ResNet50")
    model_choice = input("Enter 1 or 2: ").strip()
    MODEL_NAME = "alexnet" if model_choice == '1' else "resnet50"

    print(f"\nWhich task are we extracting for {MODEL_NAME.upper()}?")
    print("1: All (Multi-Task)")
    print("2: Appearance")
    print("3: Semantic")
    print("4: Structure")
    task_choice = input("Enter 1, 2, 3, or 4: ").strip()

    task_map = {'1': 'all', '2': 'appearance', '3': 'semantic', '4': 'structure'}
    TARGET_TASK = task_map.get(task_choice, 'all')

    # Dynamically build paths based on user input
    MODEL_PATH = PROJECT_ROOT / f"outputs/finetuned_{TARGET_TASK}_{MODEL_NAME}.pth"
    OUTPUT_DIR = PROJECT_ROOT / f"outputs/finetuned_{TARGET_TASK}/features/{MODEL_NAME}"


    device = get_best_device()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(LABELS_CSV)
    
    # --- 2. ARCHITECTURE SWITCH ---
    if TARGET_TASK == "all":
        if MODEL_NAME == "alexnet":
            model = MultiTaskAlexNet(df.Appearance_Category.max(), df.Semantic_Category.max(), df.Structure_Category.max()).to(device)
        elif MODEL_NAME == "resnet50":
            model = MultiTaskResNet50(df.Appearance_Category.max(), df.Semantic_Category.max(), df.Structure_Category.max()).to(device)
    else:
        num_classes = df.Appearance_Category.max() if TARGET_TASK == "appearance" else (df.Semantic_Category.max() if TARGET_TASK == "semantic" else df.Structure_Category.max())
        if MODEL_NAME == "alexnet":
            model = models.alexnet(weights=None)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        elif MODEL_NAME == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Define standard layers to extract
    if MODEL_NAME == "alexnet":
        layers_to_extract = {
            "features_2": model.features[2], "features_5": model.features[5],
            "features_7": model.features[7], "features_9": model.features[9],
            "features_12": model.features[12]
        }
        # Add the specific classifier layers based on architecture
        if TARGET_TASK == "all":
            layers_to_extract.update({"shared_classifier_2": model.shared_classifier[2], "shared_classifier_5": model.shared_classifier[5]})
        else:
            layers_to_extract.update({"classifier_2": model.classifier[2], "classifier_5": model.classifier[5], "classifier_6": model.classifier[6]})

    elif MODEL_NAME == "resnet50":
        if TARGET_TASK == "all":
            layers_to_extract = {"layer1": model.backbone[4], "layer2": model.backbone[5], "layer3": model.backbone[6], "layer4": model.backbone[7]}
        else:
            layers_to_extract = {"layer1": model.layer1, "layer2": model.layer2, "layer3": model.layer3, "layer4": model.layer4}

    # Setup activations dictionary
    activations = {name: [] for name in layers_to_extract.keys()}
    
    # Track the final decision heads manually based on task
    if TARGET_TASK == "all":
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
    print(f"\nExtracting layers from {MODEL_NAME.upper()} ({TARGET_TASK.upper()}) for {len(target_filenames)} images...")

    with open(OUTPUT_DIR / "file_names.txt", "w") as f:
        with torch.no_grad():
            for filename in target_filenames:
                img = Image.open(TEST_IMG_DIR / filename).convert("RGB")
                img_tensor = transform(img).unsqueeze(0).to(device)
                
                # --- 3. FORWARD PASS SWITCH ---
                if TARGET_TASK == "all":
                    out_app, out_sem, out_str = model(img_tensor)
                    activations["head_app"].append(out_app.detach().cpu().numpy().squeeze())
                    activations["head_sem"].append(out_sem.detach().cpu().numpy().squeeze())
                    activations["head_str"].append(out_str.detach().cpu().numpy().squeeze())
                else:
                    _ = model(img_tensor)
                    
                f.write(f"{filename}\n")

    for handle in handles: handle.remove()

    for name, acts in activations.items():
        np.save(OUTPUT_DIR / f"{MODEL_NAME}_{name}.npy", np.stack(acts))
        print(f" - Saved {name}.npy")

if __name__ == "__main__":
    main()
