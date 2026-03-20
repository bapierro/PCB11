#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import models, transforms
import matplotlib.pyplot as plt
import tempfile
import shutil

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_existing_dir(*candidates):
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "None of the expected folders exist: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def get_best_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --- 1. Custom Dataset with Leakage Prevention ---
class SYNSBehaviorDataset(Dataset):
    def __init__(self, img_dir, test_dir, labels_csv, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Load the consensus labels
        df = pd.read_csv(labels_csv)
        
        # 1. Catalog the exact filenames in the test directory to prevent leakage
        self.test_filenames = {p.name for p in Path(test_dir).rglob("*.jpg")}
                
        self.samples = []
        for _, row in df.iterrows():
            filename = f"S{int(row['SYNSscene'])}_Im{int(row['SYNSView'])}.jpg"
            img_path = self.img_dir / filename
            
            # STRICT CHECK: Exclude images used in our MEG test set
            if img_path.exists() and filename not in self.test_filenames:
                self.samples.append({
                    "path": img_path,
                    "appearance": int(row['Appearance_Category']) - 1, 
                    "semantic": int(row['Semantic_Category']) - 1,
                    "structure": int(row['Structure_Category']) - 1
                })
                
        print(f"Loaded {len(self.samples)} training images.")
        print(f"Strictly excluded {len(self.test_filenames)} MEG test images found in folders.")

        # Dynamically find the number of classes for each task
        self.num_app_classes = max([s["appearance"] for s in self.samples]) + 1
        self.num_sem_classes = max([s["semantic"] for s in self.samples]) + 1
        self.num_str_classes = max([s["structure"] for s in self.samples]) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, sample["appearance"], sample["semantic"], sample["structure"]


# --- 2a. Multi-Task Model Architecture for AlexNet (Fully Unfrozen) ---
class MultiTaskAlexNet(nn.Module):
    def __init__(self, num_app, num_sem, num_str):
        super().__init__()
        # Load standard AlexNet pretrained on ImageNet
        base_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        
        # Keep features and average pooling (Weights remain unfrozen so it can learn scene structures)
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # Keep up to the second-to-last layer of the classifier
        self.shared_classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])
        
        # Create 3 separate heads replacing the final classifier layer
        in_features = 4096
        self.head_app = nn.Linear(in_features, num_app)
        self.head_sem = nn.Linear(in_features, num_sem)
        self.head_str = nn.Linear(in_features, num_str)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared_classifier(x)
        
        out_app = self.head_app(x)
        out_sem = self.head_sem(x)
        out_str = self.head_str(x)
        return out_app, out_sem, out_str
    
# --- 2b. Multi-Task Model Architecture for ResNet50 (Fully Unfrozen) ---
class MultiTaskResNet50(nn.Module):
    def __init__(self, num_app, num_sem, num_str):
        super().__init__()
        # Load standard ResNet50 pretrained on ImageNet
        base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

        # Shared backbone (everything except final FC layer)
        # now the last layer is the avgpool, so no need for average pooling explicitly
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])

        # Feature size from ResNet50
        in_features = base_model.fc.in_features  # 2048
        
        # Create 3 separate heads replacing the final classifier layer
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

# --- 3. Training Loop ---
def main():
    # --- 1. INTERACTIVE MENU: MODEL & TASK ---
    print("\n--- MODEL SELECTION ---")
    print("1: AlexNet")
    print("2: ResNet50")
    model_choice = input("Enter 1 or 2: ").strip()
    MODEL_NAME = "alexnet" if model_choice == '1' else "resnet50"

    print(f"\nWhich task are we running for {MODEL_NAME.upper()}?")
    print("1: All (Multi-Task)")
    print("2: Appearance")
    print("3: Semantic")
    print("4: Structure")
    task_choice = input("Enter 1, 2, 3, or 4: ").strip()
    task_map = {'1': 'all', '2': 'appearance', '3': 'semantic', '4': 'structure'}
    TARGET_TASK = task_map.get(task_choice, 'all')

    TRAIN_IMG_DIR = resolve_existing_dir(
        PROJECT_ROOT / "data/scenes/syns_anderson_full",
        PROJECT_ROOT / "data/extracted_anderson_pictures 2",
    )
    TEST_IMG_DIR = PROJECT_ROOT / "data/scenes/syns_meg36_real"
    LABELS_CSV = PROJECT_ROOT / "data/behaviour/consensus_labels.csv"
    OUTPUT_WEIGHTS = PROJECT_ROOT / f"outputs/finetuned_{TARGET_TASK}_{MODEL_NAME}.pth"
    OUTPUT_PLOT = PROJECT_ROOT / f"outputs/loss_curve_{TARGET_TASK}_{MODEL_NAME}.png"

    device = get_best_device()
    print(f"Training on: {device}")

    # Ensure output directory exists
    OUTPUT_WEIGHTS.parent.mkdir(parents=True, exist_ok=True)

    # 1. Define separate transform pipelines
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5), # Augmentation to prevent overfitting
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224), # Strict center crop for accurate validation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Create TWO dataset instances, passing the different transforms
    full_train_dataset = SYNSBehaviorDataset(TRAIN_IMG_DIR, TEST_IMG_DIR, LABELS_CSV, transform=train_transform)
    full_val_dataset = SYNSBehaviorDataset(TRAIN_IMG_DIR, TEST_IMG_DIR, LABELS_CSV, transform=val_transform)
    
    # 3. Generate random indices for the 80/20 split
    dataset_size = len(full_train_dataset)
    val_size = int(0.20 * dataset_size)
    train_size = dataset_size - val_size
    
    indices = torch.randperm(dataset_size).tolist()
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 4. Create the final subsets using those indices
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_val_dataset, val_indices)
    
    print(f"Training on {train_size} images with augmentations.")
    print(f"Validating on {val_size} images with strict cropping.")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # --- ARCHITECTURE INITIALIZATION SWITCH ---
    if TARGET_TASK == "all":
        if MODEL_NAME == "alexnet":
            model = MultiTaskAlexNet(full_train_dataset.num_app_classes, full_train_dataset.num_sem_classes, full_train_dataset.num_str_classes).to(device)
        elif MODEL_NAME == "resnet50":
            model = MultiTaskResNet50(full_train_dataset.num_app_classes, full_train_dataset.num_sem_classes, full_train_dataset.num_str_classes).to(device)
    else:
        # Determine specific class count
        num_classes = full_train_dataset.num_app_classes if TARGET_TASK == "appearance" else (full_train_dataset.num_sem_classes if TARGET_TASK == "semantic" else full_train_dataset.num_str_classes)
        
        if MODEL_NAME == "alexnet":
            model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
            model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        elif MODEL_NAME == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

    # Loss functions and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    epochs = 10

    # Trackers
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print("Starting multi-task fine-tuning...")
    
    try:
        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            model.train()
            running_train_loss = 0.0
            
            train_correct_app, train_correct_sem, train_correct_str, train_correct_single = 0, 0, 0, 0
            train_total = 0

            for images, labels_app, labels_sem, labels_str in train_loader:
                images = images.to(device)
                optimizer.zero_grad()

                # --- LOSS CALCULATION SWITCH ---
                if TARGET_TASK == "all":
                    labels_app, labels_sem, labels_str = labels_app.to(device), labels_sem.to(device), labels_str.to(device)
                    preds_app, preds_sem, preds_str = model(images)
                    loss = criterion(preds_app, labels_app) + criterion(preds_sem, labels_sem) + criterion(preds_str, labels_str)
                    
                    train_correct_app += (torch.max(preds_app.data, 1)[1] == labels_app).sum().item()
                    train_correct_sem += (torch.max(preds_sem.data, 1)[1] == labels_sem).sum().item()
                    train_correct_str += (torch.max(preds_str.data, 1)[1] == labels_str).sum().item()
                    train_total += labels_app.size(0)
                else:
                    labels = labels_app if TARGET_TASK == "appearance" else (labels_sem if TARGET_TASK == "semantic" else labels_str)
                    labels = labels.to(device)
                    preds = model(images)
                    loss = criterion(preds, labels)
                    
                    train_correct_single += (torch.max(preds.data, 1)[1] == labels).sum().item()
                    train_total += labels.size(0)

                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
                
                
            avg_train_loss = running_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # --- VALIDATION PHASE ---
            model.eval()
            running_val_loss = 0.0
            
            val_correct_app, val_correct_sem, val_correct_str, val_correct_single = 0, 0, 0, 0
            val_total = 0

            with torch.no_grad(): 
                for images, labels_app, labels_sem, labels_str in val_loader:
                    images = images.to(device)
                    
                    if TARGET_TASK == "all":
                        labels_app, labels_sem, labels_str = labels_app.to(device), labels_sem.to(device), labels_str.to(device)
                        preds_app, preds_sem, preds_str = model(images)
                        loss = criterion(preds_app, labels_app) + criterion(preds_sem, labels_sem) + criterion(preds_str, labels_str)
                        
                        val_correct_app += (torch.max(preds_app.data, 1)[1] == labels_app).sum().item()
                        val_correct_sem += (torch.max(preds_sem.data, 1)[1] == labels_sem).sum().item()
                        val_correct_str += (torch.max(preds_str.data, 1)[1] == labels_str).sum().item()
                        val_total += labels_app.size(0)
                    else:
                        labels = labels_app if TARGET_TASK == "appearance" else (labels_sem if TARGET_TASK == "semantic" else labels_str)
                        labels = labels.to(device)
                        preds = model(images)
                        loss = criterion(preds, labels)
                        
                        val_correct_single += (torch.max(preds.data, 1)[1] == labels).sum().item()
                        val_total += labels.size(0)

                    running_val_loss += loss.item()
                    
            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            # --- PRINTING SWITCH ---
            print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
            if TARGET_TASK == "all":
                print(f"  Loss | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
                print(f"  App. Acc | Train: {100*train_correct_app/train_total:.2f}% | Val: {100*val_correct_app/val_total:.2f}%")
                print(f"  Sem. Acc | Train: {100*train_correct_sem/train_total:.2f}% | Val: {100*val_correct_sem/val_total:.2f}%")
                print(f"  Str. Acc | Train: {100*train_correct_str/train_total:.2f}% | Val: {100*val_correct_str/val_total:.2f}%")
            else:
                acc_train = 100 * train_correct_single / train_total
                acc_val = 100 * val_correct_single / val_total
                print(f"  Loss | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
                print(f"  {TARGET_TASK.capitalize()} Acc | Train: {acc_train:.2f}% | Val: {acc_val:.2f}%")

            # --- CHECKPOINTING ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save to a temporary local file first
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp:
                    torch.save(model.state_dict(), tmp.name)
                    tmp_path = tmp.name

                # Move the file to the final location
                shutil.move(tmp_path, OUTPUT_WEIGHTS)
                print(f"  -> Best {TARGET_TASK.upper()} model saved! (Val Loss: {best_val_loss:.4f})")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user! The best model found so far has been safely kept.")

    # Plot and save the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', marker='o')

    title_prefix = "Multi-Task" if TARGET_TASK == "all" else TARGET_TASK.capitalize()
    plt.title(f'{title_prefix} Training vs Validation Loss ({MODEL_NAME.upper()})')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(OUTPUT_PLOT)
    print(f"\nLoss curve saved to {OUTPUT_PLOT}. Check this file to verify your parameters!")

if __name__ == "__main__":
    main()
