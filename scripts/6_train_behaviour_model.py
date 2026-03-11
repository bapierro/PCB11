#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
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
# Creates a dataset in which test images are excluded, and behavioral labels are associated 
# with each training image.
class SYNSBehaviorDataset(Dataset):
    def __init__(self, img_dir, test_dir, labels_csv, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Load the consensus labels
        df = pd.read_csv(labels_csv)
        
        # 1. Catalog the exact filenames in the test directory to prevent leakage
        # We use a set of strings like "S1_Im5.jpg" for instant lookup
        self.test_filenames = {p.name for p in Path(test_dir).rglob("*.jpg")}
                
        self.samples = []

        # loop through the CSV and build the dataset
        for _, row in df.iterrows():
            # Build the exact filename from the CSV IDs
            # Assumes CSV has integers like SYNSscene=1, SYNSView=5
            filename = f"S{int(row['SYNSscene'])}_Im{int(row['SYNSView'])}.jpg"
            img_path = self.img_dir / filename
            
            # STRICT CHECK: 
            # 1. Does the file actually exist in the training folder?
            # 2. Is it NOT one of the images used in our MEG test set?
            if img_path.exists() and filename not in self.test_filenames:
                # behavioral labels associated with the image are stored in the samples dictionary
                # 1 is subtracted to convert from 1-based to 0-based indexing 
                self.samples.append({
                    "path": img_path,
                    "appearance": int(row['Appearance_Category']) - 1, 
                    "semantic": int(row['Semantic_Category']) - 1,
                    "structure": int(row['Structure_Category']) - 1
                })
                
        print(f"Loaded {len(self.samples)} training images.")
        print(f"Strictly excluded {len(self.test_filenames)} MEG test images found in folders.")

        # Dynamically find the number of classes for each task to build our model heads
        self.num_app_classes = max([s["appearance"] for s in self.samples]) + 1
        self.num_sem_classes = max([s["semantic"] for s in self.samples]) + 1
        self.num_str_classes = max([s["structure"] for s in self.samples]) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # returns the image at the given image and the corresponding labels (also performs image transform)
        sample = self.samples[idx]
        image = Image.open(sample["path"]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        return image, sample["appearance"], sample["semantic"], sample["structure"]


# --- 2. Multi-Task Model Architecture ---
class MultiTaskAlexNet(nn.Module):
    def __init__(self, num_app, num_sem, num_str):
        super().__init__()
        # Load standard AlexNet pretrained on ImageNet
        base_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        
        # Keep features and average pooling
        self.features = base_model.features
        self.avgpool = base_model.avgpool
        
        # Keep up to the second-to-last layer of the classifier (classifier[0] to classifier[5])
        self.shared_classifier = nn.Sequential(*list(base_model.classifier.children())[:-1])
        
        # Create 3 separate heads replacing the final classifier[6] layer
        in_features = 4096
        self.head_app = nn.Linear(in_features, num_app)
        self.head_sem = nn.Linear(in_features, num_sem)
        self.head_str = nn.Linear(in_features, num_str)
    
    # forward pass through the model (returns three separate outputs for the three tasks)
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.shared_classifier(x)
        
        # Branch out to the three tasks
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

    # Standard ImageNet transforms (add image augmentations here? For example RandomAffine, ColorJitter, RandomHorizontalFlip)
    # (then need to do a separate one for validation set without augmentations)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = SYNSBehaviorDataset(TRAIN_IMG_DIR, TEST_IMG_DIR, LABELS_CSV, transform)
    
    # 90/10 Split to maximize training data while still getting a validation signal
    val_size = int(0.10 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training on {train_size} images, Validating on {val_size} images.")

    # choosing the batch size: small -> better accuracy but longer training time
    # DataLoader loads the data in batches and shuffles it for training
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # why isn't validation data shuffled?
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Initialize model with the correct number of classes
    model = MultiTaskAlexNet(
        full_dataset.num_app_classes, 
        full_dataset.num_sem_classes, 
        full_dataset.num_str_classes
    ).to(device)

    # Loss functions and Optimizer (Adam is one of the most popular optimizers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    epochs = 3 # number of times training is done on the entire training dataset

    # Trackers
    train_losses = []
    val_losses = []
    best_val_loss = float('inf') # Tracks the absolute lowest validation loss

    print("Starting multi-task fine-tuning...")
    
    # model is trained for a number of epochs, and the best model (with the lowest validation loss) is saved to disk.
    try:
        for epoch in range(epochs):
            # --- TRAINING PHASE ---
            model.train()
            running_train_loss = 0.0
            # loop through the training data and perform forward and backward passes
            for images, labels_app, labels_sem, labels_str in train_loader:
                images, labels_app, labels_sem, labels_str = images.to(device), labels_app.to(device), labels_sem.to(device), labels_str.to(device)

                # this sets the gradients to zero before backpropagation
                optimizer.zero_grad()
                # forward pass through the model to get predictions for all three tasks
                preds_app, preds_sem, preds_str = model(images)
                # compute the total loss
                loss = criterion(preds_app, labels_app) + criterion(preds_sem, labels_sem) + criterion(preds_str, labels_str)
                # backpropagation to compute gradients of the loss and update model weights
                loss.backward()
                # this updates the model's parameters based on the computed gradients
                optimizer.step()
                running_train_loss += loss.item()
            
            # average the training losses for each batch to get the average training loss for the epoch
            avg_train_loss = running_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # --- VALIDATION PHASE ---
            model.eval()
            running_val_loss = 0.0
            with torch.no_grad(): # Don't calculate gradients during validation!
                for images, labels_app, labels_sem, labels_str in val_loader:
                    images, labels_app, labels_sem, labels_str = images.to(device), labels_app.to(device), labels_sem.to(device), labels_str.to(device)
                    preds_app, preds_sem, preds_str = model(images)
                    loss = criterion(preds_app, labels_app) + criterion(preds_sem, labels_sem) + criterion(preds_str, labels_str)
                    running_val_loss += loss.item()
                    
            avg_val_loss = running_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # --- CHECKPOINTING (Save only if it's the best so far) ---
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
    print(f"Loss curve saved to {OUTPUT_PLOT}. Check this file to verify your parameters!")

if __name__ == "__main__":
    main()