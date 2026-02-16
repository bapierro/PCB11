"""
Feature extraction module for RSA analysis.

Extracts feature vectors from ANN modules with configurable pooling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F

from thingsvision import get_extractor, get_extractor_from_model
from thingsvision.utils.data import DataLoader, ImageDataset

from .data_utils import ensure_image_dir


# Default layers for common models
RESNET_LAYERS = ["layer1", "layer2", "layer3", "layer4"]
ALEXNET_LAYERS = ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"]
VIT_B_16_LAYERS = [f"encoder.layers.encoder_layer_{idx}" for idx in range(12)]
VIT_B_32_LAYERS = [f"encoder.layers.encoder_layer_{idx}" for idx in range(12)]
DINOV2_VITS14_LAYERS = [f"blocks.{idx}" for idx in range(12)]
DINOV2_VITB14_LAYERS = [f"blocks.{idx}" for idx in range(12)]
DINOV2_VITL14_LAYERS = [f"blocks.{idx}" for idx in range(24)]
DINOV2_VITG14_LAYERS = [f"blocks.{idx}" for idx in range(40)]
ALL_LAYERS_TOKEN = "all"


PoolMode = Literal["none", "gap", "flatten", "cls", "token_mean"]


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
    if model_lower == "vit_b_16":
        return VIT_B_16_LAYERS
    if model_lower == "vit_b_32":
        return VIT_B_32_LAYERS
    if model_lower == "dinov2_vits14":
        return DINOV2_VITS14_LAYERS
    if model_lower == "dinov2_vitb14":
        return DINOV2_VITB14_LAYERS
    if model_lower == "dinov2_vitl14":
        return DINOV2_VITL14_LAYERS
    if model_lower == "dinov2_vitg14":
        return DINOV2_VITG14_LAYERS
    # Fallback: user must specify
    return []


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    """Dedupe list while preserving original order."""
    seen = set()
    out: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _apply_pooling(
    activations: np.ndarray, pool_mode: PoolMode
) -> np.ndarray:
    """Apply pooling to activations.

    Args:
        activations: Shape [N, C, H, W] or [N, C] or [N, D]
        pool_mode: "none", "gap", "flatten", "cls", or "token_mean"

    Returns:
        Pooled activations. For GAP: [N, C]. For flatten: [N, ...]. For
        CLS/token_mean on ViT-style activations: [N, D].
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
        if pool_mode == "token_mean":
            return activations.mean(axis=1)
        if pool_mode == "cls":
            if activations.shape[1] == 0:
                raise ValueError("Cannot apply cls pooling to empty token dimension.")
            return activations[:, 0, :]

    return activations


def _sanitize_name(name: str) -> str:
    """Sanitize layer name for filenames."""
    return name.replace(".", "_").replace("/", "_")


def extract_features(config: FeatureConfig) -> Dict[str, Path]:
    """
    Extract features from specified ANN modules.

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
    if config.model_name.startswith("dinov2_"):
        print(f"Loading {config.model_name} from facebookresearch/dinov2 via torch.hub...")
        model = torch.hub.load("facebookresearch/dinov2", config.model_name)
        extractor = get_extractor_from_model(
            model=model,
            device=config.device,
            backend="pt",
        )
    else:
        extractor = get_extractor(
            model_name=config.model_name,
            source=config.source,
            device=config.device,
            pretrained=config.pretrained,
        )

    # Determine layers (support explicit "all" to extract from every module)
    layers = config.layers or _get_default_layers(config.model_name)
    if layers and len(layers) == 1 and layers[0].strip().lower() == ALL_LAYERS_TOKEN:
        layers = list(extractor.get_module_names())
    if layers:
        layers = _dedupe_preserve_order(layers)
    if not layers:
        raise ValueError(
            f"No default layers for '{config.model_name}'. "
            "Please specify --layers explicitly."
        )
    (output_dir / "module_names.txt").write_text("\n".join(layers) + "\n", encoding="utf-8")

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
    print(f"Saved {len(saved_files)} feature files to: {output_dir}")

    return saved_files
