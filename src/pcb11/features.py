"""
Feature extraction module for RSA analysis.

Extracts feature vectors from CNN layers with optional Global Average Pooling (GAP).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from thingsvision import get_extractor
from thingsvision.utils.data import DataLoader, ImageDataset

from .data_utils import ensure_image_dir


# Default layers for common models
RESNET_LAYERS = ["layer1", "layer2", "layer3", "layer4"]
ALEXNET_LAYERS = ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"]


PoolMode = Literal["none", "gap", "flatten"]


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    image_path: Path
    output_dir: Path
    model_name: str = "resnet50"
    source: str = "torchvision"
    device: str = "cpu"
    pretrained: bool = True
    layers: Optional[List[str]] = None
    pool_mode: PoolMode = "gap"
    batch_size: int = 32


def _get_default_layers(model_name: str) -> List[str]:
    """Get default layers for a given model."""
    model_lower = model_name.lower()
    if "resnet" in model_lower:
        return RESNET_LAYERS
    if "alexnet" in model_lower:
        return ALEXNET_LAYERS
    # Fallback: user must specify
    return []


def _apply_pooling(
    activations: np.ndarray, pool_mode: PoolMode
) -> np.ndarray:
    """Apply pooling to activations.
    
    Args:
        activations: Shape [N, C, H, W] or [N, C] or [N, D]
        pool_mode: "none", "gap", or "flatten"
    
    Returns:
        Pooled activations. For GAP: [N, C]. For flatten: [N, C*H*W].
    """
    if pool_mode == "none":
        return activations

    # Already 2D (e.g., from fc layers): no spatial dims to pool
    if activations.ndim == 2:
        return activations

    # 4D tensor: [N, C, H, W]
    if activations.ndim == 4:
        if pool_mode == "gap":
            # Global Average Pooling: [N, C, H, W] -> [N, C]
            tensor = torch.from_numpy(activations)
            pooled = F.adaptive_avg_pool2d(tensor, (1, 1))
            return pooled.squeeze(-1).squeeze(-1).numpy()
        if pool_mode == "flatten":
            # Flatten spatial dims: [N, C, H, W] -> [N, C*H*W]
            n = activations.shape[0]
            return activations.reshape(n, -1)

    # 3D tensor: [N, C, L] (e.g., 1D convolutions)
    if activations.ndim == 3:
        if pool_mode == "gap":
            return activations.mean(axis=-1)
        if pool_mode == "flatten":
            n = activations.shape[0]
            return activations.reshape(n, -1)

    return activations


def _sanitize_name(name: str) -> str:
    """Sanitize layer name for filenames."""
    return name.replace(".", "_").replace("/", "_")


def extract_features(config: FeatureConfig) -> Dict[str, Path]:
    """
    Extract features from specified layers of a CNN.

    Args:
        config: Feature extraction configuration.

    Returns:
        Dictionary mapping layer names to saved .npy file paths.
    """
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve image directory
    image_dir = ensure_image_dir(config.image_path, output_dir)
    file_names = None
    if config.image_path.is_file():
        file_names = [config.image_path.name]

    # Get extractor
    extractor = get_extractor(
        model_name=config.model_name,
        source=config.source,
        device=config.device,
        pretrained=config.pretrained,
    )

    # Determine layers
    layers = config.layers or _get_default_layers(config.model_name)
    if not layers:
        raise ValueError(
            f"No default layers for '{config.model_name}'. "
            "Please specify --layers explicitly."
        )

    # Create dataset
    dataset = ImageDataset(
        root=str(image_dir),
        out_path=str(output_dir),
        backend=extractor.get_backend(),
        file_names=file_names,
        transforms=extractor.get_transformations(),
    )
    batches = DataLoader(
        dataset, batch_size=config.batch_size, backend=extractor.get_backend()
    )

    saved_files: Dict[str, Path] = {}

    print(f"Extracting features from {config.model_name}...")
    print(f"Layers: {layers}")
    print(f"Pool mode: {config.pool_mode}")
    print(f"Device: {config.device}")
    print("-" * 40)

    for layer in layers:
        print(f"  Layer: {layer}...", end=" ", flush=True)

        features = extractor.extract_features(
            batches=batches,
            module_name=layer,
            flatten_acts=False,  # We handle pooling ourselves
            output_type="ndarray",
        )

        # Apply pooling
        features = _apply_pooling(features, config.pool_mode)

        # Save
        safe_name = _sanitize_name(layer)
        out_file = output_dir / f"{config.model_name}_{safe_name}.npy"
        np.save(out_file, features)
        saved_files[layer] = out_file

        print(f"shape={features.shape} -> {out_file.name}")

    print("-" * 40)
    print(f"✅ Saved {len(saved_files)} feature files to: {output_dir}")

    return saved_files
