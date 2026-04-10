import numpy as np
from pathlib import Path

# Path to your current alexnet features
FILE_PATH = Path("outputs\\clean_baseline\\features\\cornet_s\\cornet_s_V4.npy")

if FILE_PATH.exists():
    data = np.load(FILE_PATH)
    print(f"File: {FILE_PATH.name}")
    print(f"Shape: {data.shape}")
    print(f"Total features per image: {data.shape[1]}")
else:
    print("File not found! Check your path.")