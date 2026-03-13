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

# --- Project Setup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "alexnet"
TRAIN_IMG_DIR = PROJECT_ROOT / "data/scenes/syns_anderson_full"
TEST_IMG_DIR = PROJECT_ROOT / "data/scenes/syns_meg36_real"
LABELS_CSV = PROJECT_ROOT / "data/behaviour/consensus_labels.csv"
OUTPUT_WEIGHTS = PROJECT_ROOT / f"outputs/finetuned_behaviour_{MODEL_NAME}.pth"
OUTPUT_PLOT = PROJECT_ROOT / f"outputs/loss_curve_{MODEL_NAME}.png"

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


# --- 2. Multi-Task Model Architecture (Fully Unfrozen) ---
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


# --- 3. Training Loop ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    
    # 3. Generate random indices for the 90/10 split
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

    # Initialize model
    model = MultiTaskAlexNet(
        full_train_dataset.num_app_classes, 
        full_train_dataset.num_sem_classes, 
        full_train_dataset.num_str_classes
    ).to(device)

    # Loss functions and Optimizer (L2 Regularization / weight_decay removed)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    epochs = 20

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
            
            train_correct_app, train_correct_sem, train_correct_str = 0, 0, 0
            train_total = 0

            for images, labels_app, labels_sem, labels_str in train_loader:
                images, labels_app, labels_sem, labels_str = images.to(device), labels_app.to(device), labels_sem.to(device), labels_str.to(device)

                optimizer.zero_grad()
                preds_app, preds_sem, preds_str = model(images)
                loss = criterion(preds_app, labels_app) + criterion(preds_sem, labels_sem) + criterion(preds_str, labels_str)
                loss.backward()
                optimizer.step()
                running_train_loss += loss.item()
                
                _, predicted_app = torch.max(preds_app.data, 1)
                _, predicted_sem = torch.max(preds_sem.data, 1)
                _, predicted_str = torch.max(preds_str.data, 1)
                
                train_total += labels_app.size(0)
                train_correct_app += (predicted_app == labels_app).sum().item()
                train_correct_sem += (predicted_sem == labels_sem).sum().item()
                train_correct_str += (predicted_str == labels_str).sum().item()
                
            avg_train_loss = running_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            acc_train_app = 100 * train_correct_app / train_total
            acc_train_sem = 100 * train_correct_sem / train_total
            acc_train_str = 100 * train_correct_str / train_total

            # --- VALIDATION PHASE ---
            model.eval()
            running_val_loss = 0.0
            
            val_correct_app, val_correct_sem, val_correct_str = 0, 0, 0
            val_total = 0

            with torch.no_grad(): 
                for images, labels_app, labels_sem, labels_str in val_loader:
                    images, labels_app, labels_sem, labels_str = images.to(device), labels_app.to(device), labels_sem.to(device), labels_str.to(device)
                    
                    preds_app, preds_sem, preds_str = model(images)
                    loss = criterion(preds_app, labels_app) + criterion(preds_sem, labels_sem) + criterion(preds_str, labels_str)
                    running_val_loss += loss.item()
                    
                    _, predicted_app = torch.max(preds_app.data, 1)
                    _, predicted_sem = torch.max(preds_sem.data, 1)
                    _, predicted_str = torch.max(preds_str.data, 1)
                    
                    val_total += labels_app.size(0)
                    val_correct_app += (predicted_app == labels_app).sum().item()
                    val_correct_sem += (predicted_sem == labels_sem).sum().item()
                    val_correct_str += (predicted_str == labels_str).sum().item()
                    
            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            
            acc_val_app = 100 * val_correct_app / val_total
            acc_val_sem = 100 * val_correct_sem / val_total
            acc_val_str = 100 * val_correct_str / val_total

            print(f"\nEpoch [{epoch+1}/{epochs}] Summary:")
            print(f"  Loss | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
            print(f"  App. Acc | Train: {acc_train_app:.2f}% | Val: {acc_val_app:.2f}%")
            print(f"  Sem. Acc | Train: {acc_train_sem:.2f}% | Val: {acc_val_sem:.2f}%")
            print(f"  Str. Acc | Train: {acc_train_str:.2f}% | Val: {acc_val_str:.2f}%")

            # --- CHECKPOINTING ---
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), OUTPUT_WEIGHTS)
                print(f"  -> Best model saved! (Val Loss: {best_val_loss:.4f})")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user! The best model found so far has been safely kept.")

    # Plot and save the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss', marker='o')
    plt.title(f'Training vs Validation Loss ({MODEL_NAME.upper()})')
    plt.xlabel('Epochs')
    plt.ylabel('Total Multi-Task Cross Entropy Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(OUTPUT_PLOT)
    print(f"\nLoss curve saved to {OUTPUT_PLOT}. Check this file to verify your parameters!")

if __name__ == "__main__":
    main()