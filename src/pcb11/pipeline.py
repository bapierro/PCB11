"""Main orchestration pipeline for ANN features, RDMs, and RSA."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thingsvision.core.rsa import compute_rdm

from .features import FeatureConfig, extract_features
from .meg_assets import (
    DEFAULT_MEG_BUNDLE_PATH,
    DEFAULT_MEG_PRECOMPUTED_ROOT,
    compute_contiguous_groups,
    prepare_shared_meg_assets,
    sync_shared_meg_assets,
)
from .rsa_outputs import compute_rsa_timecourses_per_subject, save_rsa_outputs

IMAGES_DIR = Path("data/scenes/syns_meg36")

MODEL_LAYER_PRESETS: dict[str, list[str]] = {
    "resnet50": ["layer1", "layer2", "layer3", "layer4"],
    "resnet18": ["layer1", "layer2", "layer3", "layer4"],
    "resnet101": ["layer1", "layer2", "layer3", "layer4"],
    "alexnet": ["features.2", "features.5", "features.7", "features.9", "features.12", "classifier.2", "classifier.5", "classifier.6"],
    "vit_b_16": [f"encoder.layers.encoder_layer_{idx}" for idx in range(12)],
    "vit_b_32": [f"encoder.layers.encoder_layer_{idx}" for idx in range(12)],
}

DEFAULT_RDM_METHOD = "correlation"
DEFAULT_MODEL = "resnet101"
DEFAULT_OUTPUT_ROOT_TEMPLATE = "outputs/run_{model}"


@dataclass
class PipelineConfig:
    """Runtime configuration for one pipeline execution."""

    model: str
    output_root: Path
    images_dir: Path = IMAGES_DIR


def _sanitize_name(name: str) -> str:
    """Return a filesystem-safe name for layer/module identifiers."""

    return name.replace(".", "_").replace("/", "_")


def _default_output_root_for_model(model: str) -> Path:
    """Build the default output folder for a selected model."""

    return Path(DEFAULT_OUTPUT_ROOT_TEMPLATE.format(model=model.lower()))


def _resolve_pool_mode(model: str) -> str:
    """Resolve pooling strategy for the selected architecture preset."""

    if model.startswith("vit_"):
        # For ViT blocks, keep the CLS token vector at each block depth.
        return "cls"
    return "gap"


def _subject_label(index: int) -> str:
    """Format a stable subject label from a zero-based index."""

    return f"subject_{index + 1:02d}"


def _warn(warnings: list[str], message: str) -> None:
    """Emit and store a warning message for the run manifest."""

    print(f"WARNING: {message}")
    warnings.append(message)


def _log(message: str) -> None:
    """Print a pipeline log line with consistent prefix."""

    print(f"[pipeline] {message}", flush=True)


def _get_device() -> str:
    """Select the best available Torch device in priority order."""

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_dirs(output_root: Path, model: str) -> dict[str, Path]:
    """Create and return all output directories used by the pipeline."""

    model_root = output_root / "RDM" / model
    meg_root = output_root / "RDM" / "MEG"
    rsa_root = output_root / "RSA"
    dirs = {
        "output_root": output_root,
        "features_model": output_root / "features" / model,
        "rdm_model_root": model_root,
        "rdm_model_data": model_root / "data",
        "rdm_model_plots": model_root / "plots",
        "rdm_model_plots_shared": model_root / "plots" / "shared_scale",
        "rdm_model_plots_auto": model_root / "plots" / "per_layer_scale",
        "rdm_meg_root": meg_root,
        "rdm_meg_data": meg_root / "data",
        "rdm_meg_data_per_subject": meg_root / "data" / "per_subject",
        "rdm_meg_plots": meg_root / "plots",
        "rdm_meg_plots_per_subject": meg_root / "plots" / "per_subject",
        "rdm_meg_subject_root": meg_root / "per_subject",
        "rsa_root": rsa_root,
        "rsa_data": rsa_root / "data",
        "rsa_data_per_subject": rsa_root / "data" / "per_subject",
        "rsa_plots": rsa_root / "plots",
        "rsa_plots_per_subject": rsa_root / "plots" / "per_subject",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def _plot_matrix(
    matrix: np.ndarray,
    title: str,
    out_path: Path,
    groups: list[tuple[int, int, str]] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    """Render one RDM heatmap image with optional group boundary overlays."""

    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Image")
    ax.set_ylabel("Image")
    ax.set_aspect("equal")

    if groups:
        boundaries = [g[0] for g in groups] + [groups[-1][1]]
        for boundary in boundaries[1:-1]:
            pos = boundary - 0.5
            ax.axhline(pos, color="white", linewidth=1.0)
            ax.axvline(pos, color="white", linewidth=1.0)
        centers = [(g[0] + g[1] - 1) / 2 for g in groups]
        labels = [g[2] for g in groups]
        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_rdm_animation(
    matrices: list[np.ndarray],
    labels: list[str],
    out_path: Path,
    groups: list[tuple[int, int, str]] | None,
    fps: int = 8,
) -> None:
    """Create a GIF that animates model RDMs across layers."""

    from matplotlib import animation

    stack = np.stack(matrices, axis=0)
    vmin = float(np.nanmin(stack))
    vmax = float(np.nanmax(stack))
    if np.isclose(vmin, vmax):
        eps = 1e-6 if vmax == 0 else abs(vmax) * 0.01
        vmin -= eps
        vmax += eps

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(matrices[0], cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Image")
    ax.set_ylabel("Image")
    ax.set_aspect("equal")
    title = ax.set_title(f"RDM: {labels[0]} (1/{len(labels)})")

    if groups:
        boundaries = [g[0] for g in groups] + [groups[-1][1]]
        for boundary in boundaries[1:-1]:
            pos = boundary - 0.5
            ax.axhline(pos, color="white", linewidth=1.0)
            ax.axvline(pos, color="white", linewidth=1.0)
        centers = [(g[0] + g[1] - 1) / 2 for g in groups]
        group_labels = [g[2] for g in groups]
        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.set_xticklabels(group_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(group_labels, fontsize=8)

    def update(frame_idx: int) -> tuple[Any, ...]:
        im.set_data(matrices[frame_idx])
        title.set_text(f"RDM: {labels[frame_idx]} ({frame_idx + 1}/{len(labels)})")
        return (im, title)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(matrices),
        interval=int(1000 / max(1, fps)),
        blit=False,
        repeat=True,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, writer=animation.PillowWriter(fps=max(1, fps)), dpi=150)
    plt.close(fig)


def _load_feature_file_names(features_dir: Path, images_dir: Path) -> list[Path]:
    """Load ordered image paths from extracted feature metadata."""

    file_names_path = features_dir / "file_names.txt"
    if not file_names_path.exists():
        raise SystemExit(f"Missing file_names.txt under features directory: {features_dir}")

    out: list[Path] = []
    for line in file_names_path.read_text(encoding="utf-8").splitlines():
        token = line.strip()
        if not token:
            continue
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = (Path.cwd() / candidate).resolve()
            if not candidate.exists():
                candidate = (images_dir / token).resolve()
        out.append(candidate.resolve())
    return out


def _compute_reorder_indices(current: Sequence[Path], target: Sequence[Path]) -> list[int]:
    """Compute index mapping from current feature order to target stimulus order."""

    current_map: dict[str, int] = {}
    for idx, path in enumerate(current):
        key = str(path.resolve())
        if key in current_map:
            raise SystemExit(f"Duplicate image path in current ordering: {path}")
        current_map[key] = idx

    indices: list[int] = []
    missing: list[str] = []
    for path in target:
        key = str(path.resolve())
        if key not in current_map:
            missing.append(key)
            continue
        indices.append(current_map[key])

    if missing:
        head = ", ".join(missing[:3])
        raise SystemExit(f"Could not align stimulus order; missing in features file_names: {head}")
    return indices


def _feature_files_complete(features_dir: Path, model: str, layers: Sequence[str]) -> bool:
    """Return True when all expected feature artifacts exist and match preset layers."""

    module_file = features_dir / "module_names.txt"
    file_names = features_dir / "file_names.txt"
    if not module_file.exists() or not file_names.exists():
        return False
    modules = [line.strip() for line in module_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if modules != list(layers):
        return False
    for layer in layers:
        layer_file = features_dir / f"{model}_{_sanitize_name(layer)}.npy"
        if not layer_file.exists():
            return False
    return True


def _extract_features_if_needed(
    config: PipelineConfig,
    features_dir: Path,
    layers: Sequence[str],
    pool_mode: str,
    warnings: list[str],
) -> tuple[bool, str]:
    """Reuse complete feature files or run extraction for the selected model preset."""

    features_reused = _feature_files_complete(features_dir, config.model, layers)
    device = _get_device()
    if features_reused:
        print(f"Reusing existing features from {features_dir}")
        return True, device

    if features_dir.exists():
        shutil.rmtree(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting features for model={config.model} with preset layers...")
    cfg = FeatureConfig(
        image_path=config.images_dir,
        output_dir=features_dir,
        model_name=config.model,
        source="torchvision",
        device=device,
        pretrained=True,
        layers=list(layers),
        pool_mode=pool_mode,
        batch_size=32,
    )
    extract_features(cfg)

    if not _feature_files_complete(features_dir, config.model, layers):
        _warn(warnings, "Feature extraction finished but expected feature artifacts are incomplete.")
        raise SystemExit("Feature extraction did not produce all expected files.")
    return False, device


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    """Persist run metadata as pretty-printed JSON."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    """Run the full end-to-end MEG+ANN pipeline and return manifest content."""

    run_start = perf_counter()
    _log("Starting run: " f"model={config.model}, output_root={config.output_root}")
    model = config.model.lower()
    if model not in MODEL_LAYER_PRESETS:
        allowed = ", ".join(sorted(MODEL_LAYER_PRESETS))
        raise SystemExit(f"Unsupported model '{config.model}'. Supported models: {allowed}")
    layers = MODEL_LAYER_PRESETS[model]
    pool_mode = _resolve_pool_mode(model)

    if not config.images_dir.exists():
        raise SystemExit(f"Fixed image directory not found: {config.images_dir}")

    warnings: list[str] = []
    _log("Preparing output directories...")
    dirs = _build_dirs(config.output_root, model)

    _log("Ensuring MEG bundle and precomputed MEG assets...")
    meg_prepared = prepare_shared_meg_assets(
        images_dir=config.images_dir,
        allow_build=True,
        logger=_log,
    )
    meg_avg = meg_prepared.meg_avg
    meg_subjects = meg_prepared.meg_subjects
    time_points = meg_prepared.time_points
    stimulus_order = meg_prepared.stimulus_order
    stimulus_source = meg_prepared.stimulus_source
    groups = compute_contiguous_groups(stimulus_order)
    _log(f"Stimulus order resolved ({len(stimulus_order)} images), source={stimulus_source}")

    if not stimulus_order:
        raise SystemExit("Resolved stimulus order is empty.")

    _log("Stage 1/5: Feature extraction or reuse...")
    features_reused, device = _extract_features_if_needed(config, dirs["features_model"], layers, pool_mode, warnings)
    _log(f"Stage 1/5 complete. feature_reused={features_reused}, device={device}, pool_mode={pool_mode}")
    _log("Loading extracted feature image order...")
    model_file_order = _load_feature_file_names(dirs["features_model"], config.images_dir)

    if len(model_file_order) != len(stimulus_order):
        raise SystemExit(
            f"Feature file_names length ({len(model_file_order)}) does not match stimulus order "
            f"({len(stimulus_order)})."
        )
    reorder_idx = _compute_reorder_indices(model_file_order, stimulus_order)

    _log("Stage 2/5: Copying precomputed MEG artifacts into run output...")
    if meg_avg.shape[0] != len(stimulus_order):
        raise SystemExit(
            f"MEG image dimension ({meg_avg.shape[0]}) does not match resolved stimulus order ({len(stimulus_order)})."
        )
    _log(
        f"Loaded MEG tensors: avg_shape={meg_avg.shape}, subjects_shape={meg_subjects.shape}, "
        f"time_points={time_points.shape[0]}"
    )
    # Write Shared MEG signals (tbh we can maybe drop this and just use the shared root)
    sync_shared_meg_assets(shared_root=meg_prepared.shared_root, run_meg_root=dirs["rdm_meg_root"])
    _log(f"Stage 2/5 complete. cache_hit={meg_prepared.shared_from_cache}")

    _log("Stage 3/5: Computing model RDMs...")
    layer_labels: list[str] = []
    layer_rdms: list[np.ndarray] = []
    for idx, layer in enumerate(layers, start=1):
        _log(f"Computing model RDM {idx}/{len(layers)} for layer '{layer}'")
        layer_file = dirs["features_model"] / f"{model}_{_sanitize_name(layer)}.npy"
        if not layer_file.exists():
            raise SystemExit(f"Missing expected feature file: {layer_file}")
        features = np.load(layer_file)
        if features.ndim != 2:
            raise SystemExit(f"Expected 2D features in {layer_file}, got {features.shape}")
        rdm = compute_rdm(features, method=DEFAULT_RDM_METHOD)
        rdm = rdm[np.ix_(reorder_idx, reorder_idx)]
        layer_label = layer_file.stem
        layer_labels.append(layer_label)
        layer_rdms.append(rdm.astype(np.float64))
        np.save(dirs["rdm_model_data"] / f"{layer_label}_rdm.npy", rdm)
    _log("Model RDM matrices computed.")

    with (dirs["rdm_model_data"] / "rdm_order.txt").open("w", encoding="utf-8") as handle:
        for path in stimulus_order:
            try:
                value = str(path.relative_to(config.images_dir))
            except ValueError:
                value = str(path)
            handle.write(value + "\n")

    stack = np.stack(layer_rdms, axis=0)
    vmin = float(np.nanmin(stack))
    vmax = float(np.nanmax(stack))
    if np.isclose(vmin, vmax):
        eps = 1e-6 if vmax == 0 else abs(vmax) * 0.01
        vmin -= eps
        vmax += eps

    for stale_plot in dirs["rdm_model_plots"].glob("*.png"):
        stale_plot.unlink()
    for stale_plot in dirs["rdm_model_plots_shared"].glob("*.png"):
        stale_plot.unlink()
    for stale_plot in dirs["rdm_model_plots_auto"].glob("*.png"):
        stale_plot.unlink()

    with (dirs["rdm_model_data"] / "layer_order.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer_index", "layer_name"])
        for idx, layer_label in enumerate(layer_labels, start=1):
            writer.writerow([idx, layer_label])

    _log("Rendering model RDM plots (shared_scale + per_layer_scale)...")
    for idx, (layer_label, matrix) in enumerate(zip(layer_labels, layer_rdms), start=1):
        _plot_matrix(
            matrix=matrix,
            title=f"RDM: {idx:02d} {layer_label} (shared scale)",
            out_path=dirs["rdm_model_plots_shared"] / f"{idx:02d}_{layer_label}_rdm.png",
            groups=groups,
            vmin=vmin,
            vmax=vmax,
        )
        _plot_matrix(
            matrix=matrix,
            title=f"RDM: {idx:02d} {layer_label} (per-layer scale)",
            out_path=dirs["rdm_model_plots_auto"] / f"{idx:02d}_{layer_label}_rdm.png",
            groups=groups,
            vmin=None,
            vmax=None,
        )
    _log("Rendering model RDM animation...")
    _save_rdm_animation(
        matrices=layer_rdms,
        labels=[f"{idx:02d}_{name}" for idx, name in enumerate(layer_labels, start=1)],
        out_path=dirs["rdm_model_root"] / "rdm_animation.gif",
        groups=groups,
    )
    _log("Stage 3/5 complete.")

    _log("Stage 4/5: Computing RSA timecourses and plots...")
    corr_by_subject = compute_rsa_timecourses_per_subject(layer_rdms, meg_subjects)
    n_subjects = corr_by_subject.shape[0]
    # Fisher-z aggregate subject correlations for a less biased group estimate.
    with np.errstate(invalid="ignore"):
        corr_clipped = np.clip(corr_by_subject, -0.999999, 0.999999)
        z_by_subject = np.arctanh(corr_clipped)
        z_group_mean = np.nanmean(z_by_subject, axis=0)
        corr_matrix = np.tanh(z_group_mean)

    with np.errstate(invalid="ignore", divide="ignore"):
        valid_count = np.sum(np.isfinite(z_by_subject), axis=0)
        denom = np.sqrt(np.where(valid_count > 1, valid_count, np.nan))
        z_sem = np.nanstd(z_by_subject, axis=0, ddof=1) / denom
        corr_sem = (1.0 - np.square(corr_matrix)) * z_sem

    np.save(dirs["rsa_data"] / "rsa_layer_time_subjects.npy", corr_by_subject)
    np.save(dirs["rsa_data"] / "rsa_layer_time_group_mean.npy", corr_matrix)
    np.save(dirs["rsa_data"] / "rsa_layer_time_group_sem.npy", corr_sem)

    if dirs["rsa_data_per_subject"].exists():
        shutil.rmtree(dirs["rsa_data_per_subject"])
    if dirs["rsa_plots_per_subject"].exists():
        shutil.rmtree(dirs["rsa_plots_per_subject"])
    dirs["rsa_data_per_subject"].mkdir(parents=True, exist_ok=True)
    dirs["rsa_plots_per_subject"].mkdir(parents=True, exist_ok=True)
    with (dirs["rsa_data_per_subject"] / "subject_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_index", "subject_label"])
        for subject_idx in range(n_subjects):
            writer.writerow([subject_idx, _subject_label(subject_idx)])

    _log(f"Writing per-subject RSA outputs for {n_subjects} subjects...")
    for subject_idx in range(n_subjects):
        subject_label = _subject_label(subject_idx)
        _log(f"  RSA per-subject outputs {subject_idx + 1}/{n_subjects}: {subject_label}")
        save_rsa_outputs(
            layer_labels=layer_labels,
            corr_matrix=corr_by_subject[subject_idx],
            time_points=time_points,
            data_dir=dirs["rsa_data_per_subject"] / subject_label,
            plots_dir=dirs["rsa_plots_per_subject"] / subject_label,
            model_name=f"{model} {subject_label}",
        )

    _log("Writing Fisher-aggregated group RSA outputs...")
    save_rsa_outputs(
        layer_labels=layer_labels,
        corr_matrix=corr_matrix,
        time_points=time_points,
        data_dir=dirs["rsa_data"],
        plots_dir=dirs["rsa_plots"],
        model_name=f"{model} (Fisher-z group mean, n={n_subjects})",
    )
    _log("Stage 4/5 complete.")

    _log("Stage 5/5: Writing run manifest...")
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": model,
            "output_root": str(config.output_root.resolve()),
            "images_dir": str(config.images_dir.resolve()),
        },
        "resolved": {
            "model_layers": layers,
            "rdm_method": DEFAULT_RDM_METHOD,
            "pool_mode": pool_mode,
            "source": "torchvision",
            "pretrained": True,
            "device": device,
            "meg_input_mode": "bundle_precomputed",
            "meg_bundle": str(DEFAULT_MEG_BUNDLE_PATH.resolve()),
            "meg_precomputed_root": str(DEFAULT_MEG_PRECOMPUTED_ROOT.resolve()),
            "meg_precomputed_cache_hit": meg_prepared.shared_from_cache,
            "stimulus_order_source": stimulus_source,
            "feature_reused": features_reused,
            "meg_subject_count": int(meg_subjects.shape[3]),
            "rsa_strategy": "subject_wise_then_fisher_z_group_mean",
            "meg_rdm_avg_shape": list(meg_avg.shape),
            "meg_rdm_subject_shape": list(meg_subjects.shape),
        },
        "warnings": warnings,
        "paths": {key: str(path.resolve()) for key, path in dirs.items()},
    }
    _write_manifest(config.output_root / "run_manifest.json", manifest)
    elapsed = perf_counter() - run_start
    _log("Stage 5/5 complete.")
    print(f"Done. Outputs written to: {config.output_root} (elapsed: {elapsed:.1f}s)")
    return manifest


def build_main_parser(description: str | None = None) -> argparse.ArgumentParser:
    """Build the main CLI argument parser with project defaults."""

    parser = argparse.ArgumentParser(
        description=description or "Run MEG+ANN feature/RDM/RSA pipeline.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MODEL_LAYER_PRESETS),
        default=DEFAULT_MODEL,
        help=f"Backbone model with hardcoded layer preset (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=("Root directory for structured outputs (default: outputs/run_<model>)."),
    )
    return parser


def main_cli(
    argv: Sequence[str] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """CLI wrapper that parses args and launches ``run_pipeline``."""

    parser = build_main_parser(description=description)
    args = parser.parse_args(argv)
    resolved_model = str(args.model).lower()
    resolved_output_root = (
        args.output_root if args.output_root is not None else _default_output_root_for_model(resolved_model)
    )
    if args.output_root is None:
        _log(f"No --output-root provided. Using default: {resolved_output_root}")
    config = PipelineConfig(
        model=resolved_model,
        output_root=resolved_output_root,
    )
    return run_pipeline(config)
