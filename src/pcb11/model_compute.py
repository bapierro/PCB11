"""Architecture-specific compute proxies for placing checkpoints on a shared x-axis."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class ComputeAxis:
    """Relative compute metadata for one ordered list of checkpoints."""

    checkpoint_costs: np.ndarray
    cumulative_costs: np.ndarray
    relative_positions: np.ndarray
    total_cost: float
    cost_unit: str = "MACs"


def _conv2d_macs(
    in_channels: int,
    out_channels: int,
    output_hw: int,
    kernel_size: int,
    groups: int = 1,
) -> float:
    return float(output_hw * output_hw * out_channels * (in_channels / groups) * kernel_size * kernel_size)


def _linear_macs(in_features: int, out_features: int, tokens: int = 1) -> float:
    return float(tokens * in_features * out_features)


def _relative_positions_from_costs(costs: Sequence[float]) -> ComputeAxis:
    checkpoint_costs = np.asarray(costs, dtype=np.float64)
    if checkpoint_costs.ndim != 1 or checkpoint_costs.size == 0:
        raise ValueError("Checkpoint costs must be a non-empty 1D sequence.")
    if np.any(checkpoint_costs < 0):
        raise ValueError("Checkpoint costs must be non-negative.")
    cumulative = np.cumsum(checkpoint_costs)
    total = float(cumulative[-1])
    if total <= 0:
        raise ValueError("Total checkpoint cost must be positive.")
    return ComputeAxis(
        checkpoint_costs=checkpoint_costs,
        cumulative_costs=cumulative,
        relative_positions=cumulative / total,
        total_cost=total,
    )


def build_relative_compute_axis(
    model_name: str,
    layer_labels: Sequence[str],
    input_size: int = 224,
) -> ComputeAxis:
    """Estimate checkpoint positions on a relative cumulative-compute axis."""

    model_key = model_name.lower()
    if model_key in {"resnet18", "resnet50", "resnet101"}:
        costs = _resnet_checkpoint_costs(model_key, layer_labels, input_size=input_size)
    elif model_key == "alexnet":
        costs = _alexnet_checkpoint_costs(layer_labels, input_size=input_size)
    elif model_key in {"vit_b_16", "vit_b_32"}:
        costs = _vit_checkpoint_costs(model_key, layer_labels, input_size=input_size)
    elif model_key in {"dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"}:
        costs = _dinov2_checkpoint_costs(model_key, layer_labels, input_size=input_size)
    else:
        raise ValueError(f"Relative compute axis is not implemented for model '{model_name}'.")
    return _relative_positions_from_costs(costs)


def _resnet_checkpoint_costs(model_name: str, layer_labels: Sequence[str], input_size: int) -> list[float]:
    specs = {
        "resnet18": {"block_type": "basic", "block_counts": [2, 2, 2, 2]},
        "resnet50": {"block_type": "bottleneck", "block_counts": [3, 4, 6, 3]},
        "resnet101": {"block_type": "bottleneck", "block_counts": [3, 4, 23, 3]},
    }[model_name]

    stem_output_hw = ((input_size + 2 * 3 - 7) // 2) + 1
    stem_cost = _conv2d_macs(3, 64, stem_output_hw, kernel_size=7)
    current_hw = ((stem_output_hw - 3) // 2) + 1  # maxpool 3x3 stride 2
    current_channels = 64
    cumulative = stem_cost
    cumulative_by_key: dict[str, float] = {"conv1": cumulative, "maxpool": cumulative}

    stage_planes = [64, 128, 256, 512]
    for stage_idx, (planes, blocks_in_stage) in enumerate(zip(stage_planes, specs["block_counts"]), start=1):
        for block_idx in range(blocks_in_stage):
            stride = 2 if stage_idx > 1 and block_idx == 0 else 1
            if specs["block_type"] == "basic":
                block_cost, next_channels, next_hw = _resnet_basic_block_cost(
                    in_channels=current_channels,
                    planes=planes,
                    input_hw=current_hw,
                    stride=stride,
                )
            else:
                block_cost, next_channels, next_hw = _resnet_bottleneck_block_cost(
                    in_channels=current_channels,
                    planes=planes,
                    input_hw=current_hw,
                    stride=stride,
                )
            cumulative += block_cost
            current_channels = next_channels
            current_hw = next_hw
            cumulative_by_key[f"layer{stage_idx}.{block_idx}"] = cumulative
        cumulative_by_key[f"layer{stage_idx}"] = cumulative

    out: list[float] = []
    previous_cumulative = 0.0
    for label in layer_labels:
        key = _canonical_resnet_checkpoint_key(label)
        if key not in cumulative_by_key:
            known = ", ".join(sorted(cumulative_by_key))
            raise ValueError(f"Unsupported ResNet checkpoint '{label}'. Known checkpoints: {known}")
        checkpoint_cumulative = cumulative_by_key[key]
        out.append(checkpoint_cumulative - previous_cumulative)
        previous_cumulative = checkpoint_cumulative
    return out


def _canonical_resnet_checkpoint_key(label: str) -> str:
    stage_match = re.match(r"^(layer[1-4])$", label)
    if stage_match:
        return stage_match.group(1)
    block_match = re.match(r"^(layer[1-4]\.\d+)", label)
    if block_match:
        return block_match.group(1)
    if label in {"conv1", "maxpool"}:
        return label
    raise ValueError(f"Could not parse ResNet checkpoint label '{label}'.")


def _resnet_basic_block_cost(in_channels: int, planes: int, input_hw: int, stride: int) -> tuple[float, int, int]:
    output_hw = input_hw // stride
    cost = 0.0
    cost += _conv2d_macs(in_channels, planes, output_hw, kernel_size=3)
    cost += _conv2d_macs(planes, planes, output_hw, kernel_size=3)
    if stride != 1 or in_channels != planes:
        cost += _conv2d_macs(in_channels, planes, output_hw, kernel_size=1)
    return cost, planes, output_hw


def _resnet_bottleneck_block_cost(
    in_channels: int,
    planes: int,
    input_hw: int,
    stride: int,
) -> tuple[float, int, int]:
    output_channels = planes * 4
    output_hw = input_hw // stride
    cost = 0.0
    cost += _conv2d_macs(in_channels, planes, input_hw, kernel_size=1)
    cost += _conv2d_macs(planes, planes, output_hw, kernel_size=3)
    cost += _conv2d_macs(planes, output_channels, output_hw, kernel_size=1)
    if stride != 1 or in_channels != output_channels:
        cost += _conv2d_macs(in_channels, output_channels, output_hw, kernel_size=1)
    return cost, output_channels, output_hw


def _alexnet_checkpoint_costs(layer_labels: Sequence[str], input_size: int) -> list[float]:
    sizes = {
        "conv1": ((input_size + 2 * 2 - 11) // 4) + 1,
    }
    sizes["pool1"] = ((sizes["conv1"] - 3) // 2) + 1
    sizes["conv2"] = sizes["pool1"]
    sizes["pool2"] = ((sizes["conv2"] - 3) // 2) + 1
    sizes["conv3"] = sizes["pool2"]
    sizes["conv4"] = sizes["conv3"]
    sizes["conv5"] = sizes["conv4"]

    cumulative_by_key = {
        "features.0": _conv2d_macs(3, 64, sizes["conv1"], kernel_size=11),
    }
    cumulative_by_key["features.1"] = cumulative_by_key["features.0"]
    cumulative_by_key["features.2"] = cumulative_by_key["features.0"]
    cumulative_by_key["features.3"] = cumulative_by_key["features.2"] + _conv2d_macs(64, 192, sizes["conv2"], kernel_size=5)
    cumulative_by_key["features.4"] = cumulative_by_key["features.3"]
    cumulative_by_key["features.5"] = cumulative_by_key["features.3"]
    cumulative_by_key["features.6"] = cumulative_by_key["features.5"] + _conv2d_macs(192, 384, sizes["conv3"], kernel_size=3)
    cumulative_by_key["features.7"] = cumulative_by_key["features.6"]
    cumulative_by_key["features.8"] = cumulative_by_key["features.7"] + _conv2d_macs(384, 256, sizes["conv4"], kernel_size=3)
    cumulative_by_key["features.9"] = cumulative_by_key["features.8"]
    cumulative_by_key["features.10"] = cumulative_by_key["features.9"] + _conv2d_macs(256, 256, sizes["conv5"], kernel_size=3)
    cumulative_by_key["features.11"] = cumulative_by_key["features.10"]
    cumulative_by_key["features.12"] = cumulative_by_key["features.10"]

    cumulative_by_key["classifier.1"] = cumulative_by_key["features.12"] + _linear_macs(256 * 6 * 6, 4096)
    cumulative_by_key["classifier.2"] = cumulative_by_key["classifier.1"]
    cumulative_by_key["classifier.4"] = cumulative_by_key["classifier.2"] + _linear_macs(4096, 4096)
    cumulative_by_key["classifier.5"] = cumulative_by_key["classifier.4"]
    cumulative_by_key["classifier.6"] = cumulative_by_key["classifier.5"] + _linear_macs(4096, 1000)

    out: list[float] = []
    previous_cumulative = 0.0
    for label in layer_labels:
        key_match = re.match(r"^(features\.\d+|classifier\.\d+)", label)
        if not key_match:
            raise ValueError(f"Could not parse AlexNet checkpoint label '{label}'.")
        key = key_match.group(1)
        if key not in cumulative_by_key:
            known = ", ".join(sorted(cumulative_by_key))
            raise ValueError(f"Unsupported AlexNet checkpoint '{label}'. Known checkpoints: {known}")
        checkpoint_cumulative = cumulative_by_key[key]
        out.append(checkpoint_cumulative - previous_cumulative)
        previous_cumulative = checkpoint_cumulative
    return out


def _vit_checkpoint_costs(model_name: str, layer_labels: Sequence[str], input_size: int) -> list[float]:
    specs = {
        "vit_b_16": {"patch_size": 16, "hidden_dim": 768, "mlp_dim": 3072},
        "vit_b_32": {"patch_size": 32, "hidden_dim": 768, "mlp_dim": 3072},
    }[model_name]
    return _transformer_checkpoint_costs(
        layer_labels=layer_labels,
        block_prefix="encoder.layers.encoder_layer_",
        patch_size=specs["patch_size"],
        hidden_dim=specs["hidden_dim"],
        mlp_dim=specs["mlp_dim"],
        input_size=input_size,
    )


def _dinov2_checkpoint_costs(model_name: str, layer_labels: Sequence[str], input_size: int) -> list[float]:
    specs = {
        "dinov2_vits14": {"patch_size": 14, "hidden_dim": 384, "mlp_dim": 1536},
        "dinov2_vitb14": {"patch_size": 14, "hidden_dim": 768, "mlp_dim": 3072},
        "dinov2_vitl14": {"patch_size": 14, "hidden_dim": 1024, "mlp_dim": 4096},
        "dinov2_vitg14": {"patch_size": 14, "hidden_dim": 1536, "mlp_dim": 6144},
    }[model_name]
    return _transformer_checkpoint_costs(
        layer_labels=layer_labels,
        block_prefix="blocks.",
        patch_size=specs["patch_size"],
        hidden_dim=specs["hidden_dim"],
        mlp_dim=specs["mlp_dim"],
        input_size=input_size,
    )


def _transformer_checkpoint_costs(
    layer_labels: Sequence[str],
    block_prefix: str,
    patch_size: int,
    hidden_dim: int,
    mlp_dim: int,
    input_size: int,
) -> list[float]:
    if input_size % patch_size != 0:
        raise ValueError(f"Input size {input_size} is not divisible by patch size {patch_size}.")
    patches_per_side = input_size // patch_size
    num_patches = patches_per_side * patches_per_side
    tokens = num_patches + 1  # class token
    patch_embed_cost = _conv2d_macs(3, hidden_dim, patches_per_side, kernel_size=patch_size)
    block_cost = _transformer_block_cost(tokens=tokens, hidden_dim=hidden_dim, mlp_dim=mlp_dim)

    out: list[float] = []
    previous_cumulative = 0.0
    for label in layer_labels:
        if not label.startswith(block_prefix):
            raise ValueError(f"Unsupported transformer checkpoint '{label}'. Expected prefix '{block_prefix}'.")
        try:
            block_index = int(label[len(block_prefix):].split(".")[0])
        except ValueError as exc:
            raise ValueError(f"Could not parse transformer checkpoint label '{label}'.") from exc
        checkpoint_cumulative = patch_embed_cost + (block_index + 1) * block_cost
        out.append(checkpoint_cumulative - previous_cumulative)
        previous_cumulative = checkpoint_cumulative
    return out


def _transformer_block_cost(tokens: int, hidden_dim: int, mlp_dim: int) -> float:
    attention_proj = 4.0 * tokens * hidden_dim * hidden_dim
    attention_scores = 2.0 * tokens * tokens * hidden_dim
    mlp = 2.0 * tokens * hidden_dim * mlp_dim
    return attention_proj + attention_scores + mlp
