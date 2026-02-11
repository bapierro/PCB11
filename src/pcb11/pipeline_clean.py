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
from scipy.io import loadmat
from scipy.stats import rankdata

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from thingsvision.core.rsa import compute_rdm

from .features import FeatureConfig, extract_features

IMAGES_DIR = Path("data/scenes/syns_meg36")
TIME_MAT_DEFAULT = Path("data/meg/time.mat")
STIMULUS_ORDER_DEFAULT = Path("data/meg/stimulus_order.csv")

MODEL_LAYER_PRESETS: dict[str, list[str]] = {
    "resnet50": ["layer1", "layer2", "layer3", "layer4"],
    "resnet18": ["layer1", "layer2", "layer3", "layer4"],
    "resnet101": ["layer1", "layer2", "layer3", "layer4"],
    "alexnet": ["features.2", "features.5", "features.9", "classifier.2", "classifier.5"],
}

DEFAULT_RDM_METHOD = "correlation"


@dataclass
class PipelineConfig:
    model: str
    meg_rdm: Path
    output_root: Path
    images_dir: Path = IMAGES_DIR
    time_mat: Path = TIME_MAT_DEFAULT
    stimulus_order: Path = STIMULUS_ORDER_DEFAULT


def _sanitize_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _subject_label(index: int) -> str:
    return f"subject_{index + 1:02d}"


def _warn(warnings: list[str], message: str) -> None:
    print(f"WARNING: {message}")
    warnings.append(message)


def _log(message: str) -> None:
    print(f"[pipeline] {message}", flush=True)


def _get_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_dirs(output_root: Path, model: str) -> dict[str, Path]:
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


def _discover_image_paths(images_dir: Path) -> list[Path]:
    images_dir = images_dir.resolve()
    supported = {".jpg", ".jpeg", ".png", ".bmp"}
    files = [
        path.resolve()
        for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in supported
    ]
    return sorted(files, key=lambda path: str(path.relative_to(images_dir)))


def _parse_scene_view_from_filename(path: Path) -> tuple[int, int] | None:
    # Expected patterns include S07_Im11.jpg and S1_Im09.jpg.
    stem = path.stem
    if not stem.startswith("S") or "_Im" not in stem:
        return None
    try:
        left, right = stem.split("_Im", 1)
        scene = int(left[1:])
        view = int(right)
        return scene, view
    except ValueError:
        return None


def _load_stimulus_order_from_csv(order_csv: Path, images_dir: Path) -> list[Path]:
    all_images = _discover_image_paths(images_dir)
    by_scene_view: dict[tuple[int, int], Path] = {}
    by_resolved: dict[str, Path] = {}
    for image in all_images:
        by_resolved[str(image.resolve())] = image
        parsed = _parse_scene_view_from_filename(image)
        if parsed is not None:
            by_scene_view[parsed] = image

    with order_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        raise SystemExit(f"Stimulus order file is empty: {order_csv}")

    out: list[Path] = []
    if "file_name" in headers:
        for row in rows:
            token = (row.get("file_name") or "").strip()
            if not token:
                raise SystemExit(f"Missing file_name entry in {order_csv}")
            candidate = Path(token)
            if not candidate.is_absolute():
                candidate = (images_dir / candidate).resolve()
            else:
                candidate = candidate.resolve()
            resolved = by_resolved.get(str(candidate))
            if resolved is None:
                raise SystemExit(f"Stimulus order references missing image: {candidate}")
            out.append(resolved)
    elif {"SYNSscene", "SYNSView"}.issubset(headers):
        for row in rows:
            try:
                scene = int(str(row["SYNSscene"]).strip())
                view = int(str(row["SYNSView"]).strip())
            except ValueError as exc:
                raise SystemExit(f"Invalid SYNSscene/SYNSView value in {order_csv}") from exc
            image = by_scene_view.get((scene, view))
            if image is None:
                raise SystemExit(
                    f"Stimulus order entry SYNSscene={scene}, SYNSView={view} not found in {images_dir}"
                )
            out.append(image)
    else:
        raise SystemExit(
            "Unsupported stimulus order format. Use headers [file_name] or [SYNSscene,SYNSView]."
        )

    if len(set(map(str, out))) != len(out):
        raise SystemExit(f"Stimulus order file contains duplicate images: {order_csv}")
    return out


def _resolve_stimulus_order(config: PipelineConfig, warnings: list[str]) -> tuple[list[Path], str]:
    if config.stimulus_order.exists():
        ordered = _load_stimulus_order_from_csv(config.stimulus_order, config.images_dir)
        return ordered, str(config.stimulus_order.resolve())

    _warn(
        warnings,
        f"No stimulus order file found at {config.stimulus_order}. Falling back to sorted image file order.",
    )
    return _discover_image_paths(config.images_dir), "fallback_sorted_images"


def _save_stimulus_order(order: Sequence[Path], out_csv: Path, source: str) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "file_name", "order_source"])
        for idx, path in enumerate(order):
            writer.writerow([idx, str(path), source])


def _parse_group(path: Path) -> tuple[int, str, str]:
    parent = path.parent.name
    if "_" in parent:
        prefix, rest = parent.split("_", 1)
        if prefix.isdigit():
            return int(prefix), parent, rest
    return 999, parent, parent


def _compute_contiguous_groups(paths: Sequence[Path]) -> list[tuple[int, int, str]]:
    labels = [_parse_group(path)[2] for path in paths]
    groups: list[tuple[int, int, str]] = []
    start = 0
    while start < len(labels):
        label = labels[start]
        end = start
        while end < len(labels) and labels[end] == label:
            end += 1
        groups.append((start, end, label))
        start = end
    return groups


def _plot_matrix(
    matrix: np.ndarray,
    title: str,
    out_path: Path,
    groups: list[tuple[int, int, str]] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
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


def _plot_index_grid(image_paths: Sequence[Path], out_path: Path) -> None:
    n_images = len(image_paths)
    cols = int(np.ceil(np.sqrt(n_images)))
    rows = int(np.ceil(n_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.4 * rows))
    axes_flat = axes.ravel() if isinstance(axes, np.ndarray) else np.array([axes])

    for ax in axes_flat[n_images:]:
        ax.axis("off")
    for idx, (ax, image_path) in enumerate(zip(axes_flat, image_paths)):
        ax.imshow(plt.imread(image_path))
        ax.set_title(f"idx {idx}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(image_path.name, fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _load_meg_rdm(meg_path: Path) -> tuple[np.ndarray, np.ndarray]:
    mat = loadmat(meg_path, squeeze_me=True)
    if "MEGRDMs_2D" not in mat:
        keys = [key for key in mat if not key.startswith("__")]
        raise SystemExit(f"MEG RDM variable 'MEGRDMs_2D' not found in {meg_path}. Found keys: {keys}")
    meg = np.asarray(mat["MEGRDMs_2D"], dtype=np.float64)
    if meg.ndim == 3:
        if meg.shape[0] != meg.shape[1]:
            raise SystemExit(f"MEG RDM must be square on first two dimensions, got {meg.shape}")
        return meg, meg[:, :, :, np.newaxis]
    if meg.ndim == 4:
        if meg.shape[0] != meg.shape[1]:
            raise SystemExit(f"MEG RDM must be square on first two dimensions, got {meg.shape}")
        return meg.mean(axis=3), meg
    raise SystemExit(f"Expected MEG RDM shape [N, N, T] or [N, N, T, S], got {meg.shape}")


def _load_time_points(time_mat_path: Path, n_time: int) -> np.ndarray:
    if time_mat_path.exists():
        mat = loadmat(time_mat_path, squeeze_me=True)
        if "time" in mat:
            time = np.asarray(mat["time"], dtype=np.float64).reshape(-1)
            if time.size == n_time:
                return time
    return np.arange(n_time, dtype=np.float64) - 200.0


def _plot_meg_snapshots(
    meg: np.ndarray,
    time_points: np.ndarray,
    out_path: Path,
    groups: list[tuple[int, int, str]],
) -> None:
    target_times = [-200.0, 0.0, 100.0, 200.0, 400.0, 800.0]
    indices = [int(np.argmin(np.abs(time_points - t))) for t in target_times]

    tri = np.triu_indices(meg.shape[0], k=1)
    values = np.stack([meg[:, :, t][tri] for t in range(meg.shape[2])], axis=0)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    n = len(indices)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 4.2 * rows))
    axes_flat = axes.ravel() if isinstance(axes, np.ndarray) else np.array([axes])
    for ax in axes_flat[n:]:
        ax.axis("off")

    for ax, idx in zip(axes_flat, indices):
        _plot_into_axis(
            ax=ax,
            matrix=meg[:, :, idx],
            title=f"MEG RDM t = {time_points[idx]:.0f} ms",
            groups=groups,
            vmin=vmin,
            vmax=vmax,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_into_axis(
    ax: Any,
    matrix: np.ndarray,
    title: str,
    groups: list[tuple[int, int, str]],
    vmin: float | None,
    vmax: float | None,
) -> None:
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Image")
    ax.set_ylabel("Image")
    ax.set_aspect("equal")
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
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_meg_summary(meg: np.ndarray, time_points: np.ndarray, out_path: Path) -> None:
    tri = np.triu_indices(meg.shape[0], k=1)
    values = np.stack([meg[:, :, t][tri] for t in range(meg.shape[2])], axis=0)
    std_t = values.std(axis=1)
    mean_t = values.mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.plot(time_points, std_t, label="Pairwise dissimilarity std", linewidth=1.4)
    ax.plot(time_points, mean_t, label="Pairwise dissimilarity mean", linewidth=1.2, alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Dissimilarity")
    ax.set_title("MEG RDM Summary Over Time")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_meg_animation(
    meg: np.ndarray,
    time_points: np.ndarray,
    out_path: Path,
    groups: list[tuple[int, int, str]],
    frame_step: int = 10,
    fps: int = 12,
) -> None:
    from matplotlib import animation

    tri = np.triu_indices(meg.shape[0], k=1)
    values = np.stack([meg[:, :, t][tri] for t in range(meg.shape[2])], axis=0)
    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))

    frame_indices = list(range(0, meg.shape[2], max(1, frame_step)))
    fig, ax = plt.subplots(figsize=(6.8, 5.9))
    im = ax.imshow(meg[:, :, frame_indices[0]], cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlabel("Image")
    ax.set_ylabel("Image")
    ax.set_aspect("equal")

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

    title = ax.set_title(f"MEG RDM t = {time_points[frame_indices[0]]:.0f} ms")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    def update(frame_idx: int) -> tuple[Any, ...]:
        t_idx = frame_indices[frame_idx]
        im.set_data(meg[:, :, t_idx])
        title.set_text(f"MEG RDM t = {time_points[t_idx]:.0f} ms")
        return (im, title)

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=int(1000 / max(1, fps)),
        blit=False,
        repeat=True,
    )
    anim.save(out_path, writer=animation.PillowWriter(fps=max(1, fps)), dpi=150)
    plt.close(fig)


def _export_meg_per_subject_assets(
    meg_subjects: np.ndarray,
    time_points: np.ndarray,
    stimulus_order: Sequence[Path],
    groups: list[tuple[int, int, str]],
    data_root: Path,
    plots_root: Path,
    gif_root: Path,
) -> None:
    if data_root.exists():
        shutil.rmtree(data_root)
    if plots_root.exists():
        shutil.rmtree(plots_root)
    if gif_root.exists():
        shutil.rmtree(gif_root)
    data_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    gif_root.mkdir(parents=True, exist_ok=True)

    n_subjects = meg_subjects.shape[3]
    with (data_root / "subject_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_index", "subject_label"])
        for subject_idx in range(n_subjects):
            writer.writerow([subject_idx, _subject_label(subject_idx)])

    for subject_idx in range(n_subjects):
        label = _subject_label(subject_idx)
        _log(f"  MEG per-subject assets {subject_idx + 1}/{n_subjects}: {label}")
        subject_meg = meg_subjects[:, :, :, subject_idx]
        subject_data = data_root / label
        subject_plots = plots_root / label
        subject_gif = gif_root / label
        subject_data.mkdir(parents=True, exist_ok=True)
        subject_plots.mkdir(parents=True, exist_ok=True)
        subject_gif.mkdir(parents=True, exist_ok=True)

        np.save(subject_data / "meg_rdm.npy", subject_meg)
        np.save(subject_data / "time_ms.npy", time_points)
        with (subject_data / "stimulus_order_used.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["index", "file_name"])
            for idx, path in enumerate(stimulus_order):
                writer.writerow([idx, str(path)])

        _plot_meg_snapshots(subject_meg, time_points, subject_plots / "meg_rdm_snapshots.png", groups)
        _plot_meg_summary(subject_meg, time_points, subject_plots / "meg_rdm_summary_over_time.png")
        _save_meg_animation(subject_meg, time_points, subject_gif / "meg_rdm_animation.gif", groups)


def _load_feature_file_names(features_dir: Path, images_dir: Path) -> list[Path]:
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
    warnings: list[str],
) -> tuple[bool, str]:
    reused = _feature_files_complete(features_dir, config.model, layers)
    device = _get_device()
    if reused:
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
        pool_mode="gap",
        batch_size=32,
    )
    extract_features(cfg)

    if not _feature_files_complete(features_dir, config.model, layers):
        _warn(warnings, "Feature extraction finished but expected feature artifacts are incomplete.")
        raise SystemExit("Feature extraction did not produce all expected files.")
    return False, device


def _rank_rows(values: np.ndarray) -> np.ndarray:
    ranked = np.empty_like(values, dtype=np.float64)
    for idx in range(values.shape[0]):
        ranked[idx] = rankdata(values[idx])
    return ranked


def _compute_rsa_timecourses(layer_rdms: list[np.ndarray], meg_rdms: np.ndarray) -> np.ndarray:
    n_images = meg_rdms.shape[0]
    if any(rdm.shape != (n_images, n_images) for rdm in layer_rdms):
        bad = [rdm.shape for rdm in layer_rdms if rdm.shape != (n_images, n_images)][0]
        raise SystemExit(f"Layer RDM shape {bad} does not match MEG shape {(n_images, n_images)}")

    tri = np.triu_indices(n_images, k=1)
    n_time = meg_rdms.shape[2]
    meg_vectors = np.empty((n_time, tri[0].size), dtype=np.float64)
    for t in range(n_time):
        meg_vectors[t] = meg_rdms[:, :, t][tri]
    meg_ranked = _rank_rows(meg_vectors)
    meg_centered = meg_ranked - meg_ranked.mean(axis=1, keepdims=True)
    meg_norms = np.linalg.norm(meg_centered, axis=1)
    meg_norms[meg_norms == 0] = np.nan

    out = np.empty((len(layer_rdms), n_time), dtype=np.float64)
    for idx, model_rdm in enumerate(layer_rdms):
        model_vec = model_rdm[tri]
        model_ranked = rankdata(model_vec)
        model_centered = model_ranked - model_ranked.mean()
        model_norm = np.linalg.norm(model_centered)
        if model_norm == 0:
            out[idx] = np.nan
            continue
        out[idx] = (meg_centered @ model_centered) / (meg_norms * model_norm)
    return out


def _compute_rsa_timecourses_per_subject(
    layer_rdms: list[np.ndarray],
    meg_rdms_subjects: np.ndarray,
) -> np.ndarray:
    if meg_rdms_subjects.ndim != 4:
        raise SystemExit(
            f"Expected subject-resolved MEG RDM shape [N, N, T, S], got {meg_rdms_subjects.shape}"
        )
    n_subjects = meg_rdms_subjects.shape[3]
    n_layers = len(layer_rdms)
    n_time = meg_rdms_subjects.shape[2]
    out = np.empty((n_subjects, n_layers, n_time), dtype=np.float64)
    for subject_idx in range(n_subjects):
        out[subject_idx] = _compute_rsa_timecourses(layer_rdms, meg_rdms_subjects[:, :, :, subject_idx])
    return out


def _save_rsa_outputs(
    layer_labels: Sequence[str],
    corr_matrix: np.ndarray,
    time_points: np.ndarray,
    data_dir: Path,
    plots_dir: Path,
    model_name: str,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    with (data_dir / "layer_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer_index", "layer_name"])
        for idx, layer_name in enumerate(layer_labels):
            writer.writerow([idx, layer_name])

    with (data_dir / "layer_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["layer_index", "layer_name", "peak_corr", "peak_time_ms", "min_corr", "min_time_ms"])
        for idx, layer_name in enumerate(layer_labels):
            corr = corr_matrix[idx]
            safe = _sanitize_name(layer_name)
            np.save(data_dir / f"{safe}_spearman.npy", corr)
            np.savetxt(
                data_dir / f"{safe}_spearman.csv",
                np.column_stack((time_points, corr)),
                delimiter=",",
                header="time,corr",
                comments="",
            )
            if np.all(np.isnan(corr)):
                max_idx = 0
                min_idx = 0
            else:
                max_idx = int(np.nanargmax(corr))
                min_idx = int(np.nanargmin(corr))
            writer.writerow(
                [
                    idx,
                    layer_name,
                    float(corr[max_idx]),
                    float(time_points[max_idx]),
                    float(corr[min_idx]),
                    float(time_points[min_idx]),
                ]
            )

    finite = np.isfinite(corr_matrix)
    if np.any(finite):
        corr_vmin = float(np.nanmin(corr_matrix))
        corr_vmax = float(np.nanmax(corr_matrix))
    else:
        corr_vmin, corr_vmax = -1.0, 1.0
    if np.isclose(corr_vmin, corr_vmax):
        eps = 1e-6 if corr_vmax == 0 else abs(corr_vmax) * 0.01
        corr_vmin -= eps
        corr_vmax += eps

    fig_h, ax_h = plt.subplots(figsize=(12, min(max(6, 2 + 0.14 * len(layer_labels)), 20)))
    im = ax_h.imshow(
        corr_matrix,
        aspect="auto",
        cmap="coolwarm",
        vmin=corr_vmin,
        vmax=corr_vmax,
        interpolation="nearest",
    )
    xticks = np.linspace(0, len(time_points) - 1, 11, dtype=int)
    ax_h.set_xticks(xticks)
    ax_h.set_xticklabels([f"{time_points[idx]:.0f}" for idx in xticks])
    ytick_count = min(len(layer_labels), 20)
    yticks = np.linspace(0, len(layer_labels) - 1, ytick_count, dtype=int)
    ax_h.set_yticks(yticks)
    ax_h.set_yticklabels([str(idx) for idx in yticks])
    zero_idx = int(np.argmin(np.abs(time_points)))
    ax_h.axvline(zero_idx, color="black", linewidth=1.0, linestyle="--")
    ax_h.set_xlabel("Time (ms)")
    ax_h.set_ylabel("Layer index (see layer_index.csv)")
    ax_h.set_title(f"Layerwise MEG RSA ({model_name}, {len(layer_labels)} layers)")
    cbar = fig_h.colorbar(im, ax=ax_h, fraction=0.03, pad=0.02)
    cbar.set_label("Spearman correlation")
    fig_h.tight_layout()
    fig_h.savefig(plots_dir / "rsa_layerwise_heatmap.png", dpi=160)
    plt.close(fig_h)

    fig_l, ax_l = plt.subplots(figsize=(12, 6.5))
    cmap = plt.get_cmap("viridis")
    denom = max(1, len(layer_labels) - 1)
    for idx, corr in enumerate(corr_matrix):
        ax_l.plot(time_points, corr, color=cmap(idx / denom), alpha=0.22, linewidth=0.9)
    ax_l.plot(
        time_points,
        corr_matrix[-1],
        color="black",
        linewidth=2.2,
        label=f"Last layer: {layer_labels[-1]}",
    )
    ax_l.axvline(0.0, color="black", linestyle="--", linewidth=1.0)
    ax_l.set_ylim(corr_vmin, corr_vmax)
    ax_l.set_xlabel("Time (ms)")
    ax_l.set_ylabel("Spearman correlation")
    ax_l.set_title(f"Layerwise RSA Time Overlay ({model_name})")
    ax_l.legend(loc="best")
    fig_l.tight_layout()
    fig_l.savefig(plots_dir / "rsa_layerwise_overlay.png", dpi=160)
    plt.close(fig_l)


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    run_start = perf_counter()
    _log(
        "Starting run: "
        f"model={config.model}, meg_rdm={config.meg_rdm}, output_root={config.output_root}"
    )
    model = config.model.lower()
    if model not in MODEL_LAYER_PRESETS:
        allowed = ", ".join(sorted(MODEL_LAYER_PRESETS))
        raise SystemExit(f"Unsupported model '{config.model}'. Supported models: {allowed}")
    layers = MODEL_LAYER_PRESETS[model]

    if not config.images_dir.exists():
        raise SystemExit(f"Fixed image directory not found: {config.images_dir}")
    if not config.meg_rdm.exists():
        raise SystemExit(f"MEG RDM file not found: {config.meg_rdm}")

    warnings: list[str] = []
    _log("Preparing output directories...")
    dirs = _build_dirs(config.output_root, model)
    _log("Resolving stimulus order...")
    stimulus_order, stimulus_source = _resolve_stimulus_order(config, warnings)
    groups = _compute_contiguous_groups(stimulus_order)
    _log(f"Stimulus order resolved ({len(stimulus_order)} images), source={stimulus_source}")

    if not stimulus_order:
        raise SystemExit("Resolved stimulus order is empty.")

    _log("Stage 1/5: Feature extraction or reuse...")
    reused_features, device = _extract_features_if_needed(config, dirs["features_model"], layers, warnings)
    _log(f"Stage 1/5 complete. feature_reused={reused_features}, device={device}")
    _log("Loading extracted feature image order...")
    model_file_order = _load_feature_file_names(dirs["features_model"], config.images_dir)

    if len(model_file_order) != len(stimulus_order):
        raise SystemExit(
            f"Feature file_names length ({len(model_file_order)}) does not match stimulus order "
            f"({len(stimulus_order)})."
        )
    reorder_idx = _compute_reorder_indices(model_file_order, stimulus_order)

    _log("Stage 2/5: Loading and exporting MEG RDM artifacts...")
    meg_avg, meg_subjects = _load_meg_rdm(config.meg_rdm)
    if meg_avg.shape[0] != len(stimulus_order):
        raise SystemExit(
            f"MEG image dimension ({meg_avg.shape[0]}) does not match resolved stimulus order ({len(stimulus_order)})."
        )
    time_points = _load_time_points(config.time_mat, n_time=meg_avg.shape[2])
    _log(
        f"Loaded MEG tensors: avg_shape={meg_avg.shape}, subjects_shape={meg_subjects.shape}, "
        f"time_points={time_points.shape[0]}"
    )

    # MEG outputs
    np.save(dirs["rdm_meg_data"] / "meg_rdm_avg.npy", meg_avg)
    np.save(dirs["rdm_meg_data"] / "time_ms.npy", time_points)
    _save_stimulus_order(
        stimulus_order,
        dirs["rdm_meg_data"] / "stimulus_order_used.csv",
        source=stimulus_source,
    )
    with (dirs["rdm_meg_root"] / "meg_rdm_index_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "file_name"])
        for idx, path in enumerate(stimulus_order):
            writer.writerow([idx, str(path)])
    _log("Writing MEG plot assets (index grid, snapshots, summary, animation)...")
    _plot_index_grid(stimulus_order, dirs["rdm_meg_plots"] / "meg_rdm_index_grid.png")
    _plot_meg_snapshots(meg_avg, time_points, dirs["rdm_meg_plots"] / "meg_rdm_snapshots.png", groups)
    _plot_meg_summary(meg_avg, time_points, dirs["rdm_meg_plots"] / "meg_rdm_summary_over_time.png")
    _save_meg_animation(meg_avg, time_points, dirs["rdm_meg_root"] / "meg_rdm_animation.gif", groups)
    _log(f"Writing per-subject MEG RDM assets for {meg_subjects.shape[3]} subjects...")
    _export_meg_per_subject_assets(
        meg_subjects=meg_subjects,
        time_points=time_points,
        stimulus_order=stimulus_order,
        groups=groups,
        data_root=dirs["rdm_meg_data_per_subject"],
        plots_root=dirs["rdm_meg_plots_per_subject"],
        gif_root=dirs["rdm_meg_subject_root"],
    )
    _log("Stage 2/5 complete.")

    # Model RDM outputs
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

    # RSA outputs
    _log("Stage 4/5: Computing RSA timecourses and plots...")
    corr_by_subject = _compute_rsa_timecourses_per_subject(layer_rdms, meg_subjects)
    n_subjects = corr_by_subject.shape[0]
    corr_matrix = np.nanmean(corr_by_subject, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid_count = np.sum(np.isfinite(corr_by_subject), axis=0)
        denom = np.sqrt(np.where(valid_count > 1, valid_count, np.nan))
        corr_sem = np.nanstd(corr_by_subject, axis=0, ddof=1) / denom

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
        _save_rsa_outputs(
            layer_labels=layer_labels,
            corr_matrix=corr_by_subject[subject_idx],
            time_points=time_points,
            data_dir=dirs["rsa_data_per_subject"] / subject_label,
            plots_dir=dirs["rsa_plots_per_subject"] / subject_label,
            model_name=f"{model} {subject_label}",
        )

    _log("Writing group-mean RSA outputs...")
    _save_rsa_outputs(
        layer_labels=layer_labels,
        corr_matrix=corr_matrix,
        time_points=time_points,
        data_dir=dirs["rsa_data"],
        plots_dir=dirs["rsa_plots"],
        model_name=f"{model} (group mean, n={n_subjects})",
    )
    _log("Stage 4/5 complete.")

    _log("Stage 5/5: Writing run manifest...")
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": {
            "model": model,
            "meg_rdm": str(config.meg_rdm.resolve()),
            "output_root": str(config.output_root.resolve()),
            "images_dir": str(config.images_dir.resolve()),
            "time_mat": str(config.time_mat.resolve()),
            "stimulus_order_preferred": str(config.stimulus_order.resolve()),
        },
        "resolved": {
            "model_layers": layers,
            "rdm_method": DEFAULT_RDM_METHOD,
            "pool_mode": "gap",
            "source": "torchvision",
            "pretrained": True,
            "device": device,
            "stimulus_order_source": stimulus_source,
            "feature_reused": reused_features,
            "meg_subject_count": int(meg_subjects.shape[3]),
            "rsa_strategy": "subject_wise_then_group_mean",
            "meg_rdm_avg_shape": list(meg_avg.shape),
            "meg_rdm_subject_shape": list(meg_subjects.shape),
        },
        "warnings": warnings,
        "paths": {key: str(path.resolve()) for key, path in dirs.items()},
    }
    _write_manifest(config.output_root / "run_manifest.json", manifest)
    elapsed = perf_counter() - run_start
    _log(f"Stage 5/5 complete.")
    print(f"Done. Outputs written to: {config.output_root} (elapsed: {elapsed:.1f}s)")
    return manifest


def build_canonical_parser(description: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=description or "Run clean MEG+ANN feature/RDM/RSA pipeline.",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=sorted(MODEL_LAYER_PRESETS),
        required=True,
        help="Backbone model with hardcoded layer preset.",
    )
    parser.add_argument(
        "--meg-rdm",
        type=Path,
        required=True,
        help="Path to MEG .mat file containing variable MEGRDMs_2D.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory for structured outputs.",
    )
    return parser


def canonical_cli_main(
    argv: Sequence[str] | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    parser = build_canonical_parser(description=description)
    args = parser.parse_args(argv)
    config = PipelineConfig(
        model=args.model,
        meg_rdm=args.meg_rdm,
        output_root=args.output_root,
    )
    return run_pipeline(config)
