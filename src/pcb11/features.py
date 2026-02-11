from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from thingsvision import get_extractor
from thingsvision.utils.data import DataLoader, ImageDataset

@dataclass
class FeatureConfig:
    image_path: Path
    output_dir: Path
    model_name: str
    layers: List[str]
    source: str = "torchvision"
    device: str = "cpu"
    pretrained: bool = True
    pool_mode: str = "gap"
    batch_size: int = 32

def _apply_pooling(activations: np.ndarray, pool_mode: str) -> np.ndarray:
    """Crushes 3D visual data into a 1D list of features."""
    if pool_mode != "gap" or activations.ndim != 4:
        return activations
    
    # Global Average Pooling: [Batch, Channels, H, W] -> [Batch, Channels]
    tensor = torch.from_numpy(activations)
    pooled = F.adaptive_avg_pool2d(tensor, (1, 1))
    return pooled.squeeze(-1).squeeze(-1).numpy()

def extract_features(config: FeatureConfig):
    """The core engine: Images -> Model -> Math."""
    config.output_dir.mkdir(parents=True, exist_ok=True)

    extractor = get_extractor(
        model_name=config.model_name,
        source=config.source,
        device=config.device,
        pretrained=config.pretrained,
    )

    dataset = ImageDataset(
        root=str(config.image_path),
        out_path=str(config.output_dir),
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(),
    )
    
    batches = DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        backend=extractor.get_backend()
    )

    for layer in config.layers:
        print(f"  Processing Layer: {layer}...")
        
        # Extract raw activations
        features = extractor.extract_features(
            batches=batches,
            module_name=layer,
            flatten_acts=False, 
            output_type="ndarray",
        )

        # Apply GAP pooling
        features = _apply_pooling(features, config.pool_mode)

        # Save to disk
        safe_layer = layer.replace(".", "_")
        out_file = config.output_dir / f"{config.model_name}_{safe_layer}.npy"
        np.save(out_file, features)