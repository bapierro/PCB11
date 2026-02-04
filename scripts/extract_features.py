import os
import torch
import numpy as np
from thingsvision import get_extractor
from thingsvision.utils.storing import save_features
from thingsvision.utils.data import ImageDataset
from torch.utils.data import DataLoader

# 1. Setup Paths
IMAGE_PATH = 'data/scenes/syns_meg36/'
OUT_PATH = 'features/baseline/'
if not os.path.exists(OUT_PATH):
    os.makedirs(OUT_PATH)

# 2. Setup Model
model_name = 'alexnet'
source = 'torchvision'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 3. Define Layers
layers = [
    'features.2',   # Layer 1
    'features.5',   # Layer 2
    'features.7',   # Layer 3
    'features.9',   # Layer 4
    'features.12',  # Layer 5
    'classifier.2', # Layer 6
    'classifier.5', # Layer 7
    'classifier.6'  # Layer 8
]

# 4. Initialize Tools

extractor = get_extractor(
    model_name=model_name,
    source=source,
    device=device,
    pretrained=True
)

dataset = ImageDataset(
    root=IMAGE_PATH,
    out_path=OUT_PATH,
    backend='pt',
    transforms=extractor.get_transformations()
)

batches = DataLoader(dataset, batch_size=32, shuffle=False)


# 5. The Full Extraction Loop
print(f"Starting extraction for layers...")

for layer in layers:
    print(f"Extracting layer: {layer}...")
    features = extractor.extract_features(
        batches=batches,
        module_name=layer,
        flatten_acts=True
    )
    
    # We save each layer separately so we can correlate them 
    # individually against the MEG timepoints later.
    # 1. Create a clean filename (e.g., layer_features_0.npy)
    file_name = f"layer_{layer.replace('.', '_')}.npy"
    
    # 2. Join it with your baseline folder path
    save_path = os.path.join(OUT_PATH, file_name)
    
    # 3. Save directly using NumPy
    np.save(save_path, features)
    print(f"✅ Saved directly to: {save_path}")

print("✅ Full hierarchy extracted. You now have the complete 'ANN Brain' for comparison.")