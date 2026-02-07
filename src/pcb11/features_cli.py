"""
CLI for feature extraction.

Usage:
    python -m pcb11.features_cli data/scenes --pool gap --layers layer1,layer2,layer3,layer4
"""

import argparse
from pathlib import Path
from typing import List, Optional

from .features import FeatureConfig, extract_features, RESNET_LAYERS, ALEXNET_LAYERS


def _parse_layers(layers_arg: Optional[str], model_name: str) -> Optional[List[str]]:
    """Parse layers argument or return None for defaults."""
    if not layers_arg:
        return None
    if layers_arg.lower() == "resnet":
        return RESNET_LAYERS
    if layers_arg.lower() == "alexnet":
        return ALEXNET_LAYERS
    return [layer.strip() for layer in layers_arg.split(",") if layer.strip()]


def _get_device() -> str:
    """Auto-detect best available device."""
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract CNN features for RSA analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract ResNet50 features with GAP pooling (default)
  python -m pcb11.features_cli data/scenes/

  # Extract specific layers
  python -m pcb11.features_cli data/scenes/ --layers layer1,layer2,layer3,layer4

  # Use AlexNet with flatten pooling
  python -m pcb11.features_cli data/scenes/ --model alexnet --pool flatten

  # Use shorthand for ResNet layers
  python -m pcb11.features_cli data/scenes/ --layers resnet
""",
    )

    parser.add_argument(
        "images",
        type=Path,
        help="Path to an image or folder of images",
    )

    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("features"),
        help="Output directory for .npy files (default: features/)",
    )

    parser.add_argument(
        "-m", "--model",
        default="resnet50",
        help="Model name (default: resnet50). Supports: resnet18/34/50/101, alexnet, vgg16, etc.",
    )

    parser.add_argument(
        "-l", "--layers",
        default=None,
        help=(
            "Comma-separated layer names, or 'resnet'/'alexnet' for presets. "
            "Default: auto-detect based on model."
        ),
    )

    parser.add_argument(
        "-p", "--pool",
        choices=["none", "gap", "flatten"],
        default="gap",
        help="Pooling mode: gap (default), flatten, or none",
    )

    parser.add_argument(
        "-d", "--device",
        default=None,
        help="Device: cpu, cuda, mps (default: auto-detect)",
    )

    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Batch size for extraction (default: 32)",
    )

    parser.add_argument(
        "--source",
        default="torchvision",
        help="Model source (default: torchvision)",
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use random weights instead of pretrained",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = args.device or _get_device()
    layers = _parse_layers(args.layers, args.model)

    config = FeatureConfig(
        image_path=args.images,
        output_dir=args.output,
        model_name=args.model,
        source=args.source,
        device=device,
        pretrained=not args.no_pretrained,
        layers=layers,
        pool_mode=args.pool,
        batch_size=args.batch_size,
    )

    extract_features(config)


if __name__ == "__main__":
    main()
