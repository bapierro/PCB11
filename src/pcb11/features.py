from dataclasses import dataclass
from logging import config
from pathlib import Path
from pyexpat import features
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from thingsvision import get_extractor, model
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

def extract_cornet_features(config):
    from cornet import cornet_s
    from torchvision import transforms

    model = cornet_s(pretrained=True, map_location=config.device)
    model.eval()

    # preprocessing (not built-in like torchvision models)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = ImageDataset(
        root=str(config.image_path),
        out_path=str(config.output_dir),
        backend="pt",
        transforms=transform,
    )

    loader = DataLoader(dataset, batch_size=config.batch_size, backend="pt")

    # storage
    features = {layer: [] for layer in config.layers}

    # Define hooks to capture activations
    def hook(name):
        def fn(m, i, o):
            features[name].append(o.detach().cpu().numpy())
        return fn
    
    core_model = model.module if hasattr(model, "module") else model

    for layer in config.layers:
        getattr(core_model, layer).register_forward_hook(hook(layer))


    # forward pass
    for batch in loader:
        images = batch

        if not isinstance(images, torch.Tensor):
            images = torch.stack(images)

        if images.dim() == 3:
            images = images.unsqueeze(0)

        images = images.to(config.device)
        print("CORnet input:", images.shape)

        with torch.no_grad():
            _ = core_model(images)

    # save after processing all batches
    for layer in config.layers:
        feats = np.concatenate(features[layer], axis=0)
        feats = _apply_pooling(feats, config.pool_mode)

        out_file = config.output_dir / f"{config.model_name}_{layer}.npy"
        np.save(out_file, feats)