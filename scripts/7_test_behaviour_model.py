#!/usr/bin/env python3
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torchvision import models, transforms
import pandas as pd
import re

# --- Setup Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "resnet50" # currently supports "alexnet" or "resnet50" (must match the model used during training)
MODEL_PATH = PROJECT_ROOT / f"outputs/finetuned_behaviour_{MODEL_NAME}.pth"
TEST_IMG_DIR = PROJECT_ROOT / "data/scenes/syns_meg36_real"
LABELS_CSV = PROJECT_ROOT / "data/behaviour/consensus_labels.csv"

# --- 1a. Model Architecture for AlexNet (Matches training exactly) ---
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
    
# --- 1b. Model Architecture for ResNet50 (Matches training exactly) ---
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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Ground Truth Data
    df = pd.read_csv(LABELS_CSV)
    
    # Create a lookup dictionary: {(scene, view): {labels}}
    gt_lookup = {}
    for _, row in df.iterrows():
        key = (int(row['SYNSscene']), int(row['SYNSView']))
        gt_lookup[key] = {
            "app": row['Appearance_Label'],
            "sem": row['Semantic_Label'],
            "str": row['Structure_Label']
        }

    # 2. Map IDs to Labels for Model Outputs
    # (Mapping the numeric category minus 1 to the text label)
    label_map = {
        "app": dict(zip(df.Appearance_Category - 1, df.Appearance_Label)),
        "sem": dict(zip(df.Semantic_Category - 1, df.Semantic_Label)),
        "str": dict(zip(df.Structure_Category - 1, df.Structure_Label))
    }

    # 3. Initialize and Load Model
    num_app, num_sem, num_str = df.Appearance_Category.max(), df.Semantic_Category.max(), df.Structure_Category.max()
    if MODEL_NAME == "alexnet":
        model = MultiTaskAlexNet(num_app, num_sem, num_str).to(device)
    elif MODEL_NAME == "resnet50":
        model = MultiTaskResNet50(num_app, num_sem, num_str).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 4. Run Verification on 5 random MEG test images
    test_images = list(TEST_IMG_DIR.rglob("*.jpg"))[:8]
    
    print(f"\n{'IMAGE':<15} | {'TASK':<12} | {'PREDICTED':<25} | {'REAL (GROUND TRUTH)':<25}")
    print("-" * 85)

    with torch.no_grad():
        for img_path in test_images:
            # Extract Scene/View IDs from filename for lookup
            match = re.search(r'S(\d+)_Im(\d+)', img_path.name)
            if not match: continue
            scene_id, view_id = int(match.group(1)), int(match.group(2))
            
            # Get Ground Truth
            gt = gt_lookup.get((scene_id, view_id))
            if not gt: continue

            # Get Model Prediction
            img = Image.open(img_path).convert("RGB")
            input_tensor = transform(img).unsqueeze(0).to(device)
            out_app, out_sem, out_str = model(input_tensor)
            
            p_app = label_map['app'].get(torch.argmax(out_app).item())
            p_sem = label_map['sem'].get(torch.argmax(out_sem).item())
            p_str = label_map['str'].get(torch.argmax(out_str).item())

            # Print results
            print(f"{img_path.name:<15} | Appearance | {str(p_app)[:24]:<25} | {str(gt['app'])[:24]:<25}")
            print(f"{'':<15} | Semantic   | {str(p_sem)[:24]:<25} | {str(gt['sem'])[:24]:<25}")
            print(f"{'':<15} | Structure  | {str(p_str)[:24]:<25} | {str(gt['str'])[:24]:<25}")
            print("-" * 85)

if __name__ == "__main__":
    main()