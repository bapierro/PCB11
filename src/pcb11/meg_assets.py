"""Shared MEG asset validation, preparation, and sync helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .meg_bundle import build_meg_bundle, load_meg_bundle


DEFAULT_MEG_BUNDLE_PATH = Path("data/meg/meg_data.npz")
DEFAULT_MEG_PRECOMPUTED_ROOT = Path("data/meg/precomputed_rdm")
DEFAULT_MEG_RDM_MAT_PATH = Path("data/meg/MEGRDMs_2D.mat")
DEFAULT_MEG_TIME_MAT_PATH = Path("data/meg/time.mat")
DEFAULT_MEG_IMAGE_STRUCT_PATH = Path("data/meg/imagestruct_final.mat")


@dataclass(frozen=True)
class MegPrepared:
    """Loaded MEG arrays and metadata for one pipeline run."""

    meg_avg: np.ndarray
    meg_subjects: np.ndarray
    time_points: np.ndarray
    stimulus_order: list[Path]
    stimulus_source: str
    shared_root: Path
    shared_from_cache: bool


def _subject_label(index: int) -> str:
    """Format a stable subject label from a zero-based index."""

    return f"subject_{index + 1:02d}"


def _save_stimulus_order(order: Sequence[Path], out_csv: Path, source: str) -> None:
    """Write resolved stimulus order to CSV for full run reproducibility."""

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "file_name", "order_source"])
        for idx, path in enumerate(order):
            writer.writerow([idx, str(path), source])


def _path_signature(path: Path) -> dict[str, Any]:
    """Capture a lightweight file signature used for cache invalidation."""

    resolved = path.resolve()
    out: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    if resolved.exists():
        stat = resolved.stat()
        out["mtime_ns"] = int(stat.st_mtime_ns)
        out["size"] = int(stat.st_size)
    else:
        out["mtime_ns"] = None
        out["size"] = None
    return out


def _sha256_bytes(payload: bytes) -> str:
    """Return SHA-256 hex digest for arbitrary byte payloads."""

    return hashlib.sha256(payload).hexdigest()


def _compute_meg_subject_cache_key(
    meg_bundle_path: Path,
    stimulus_source: str,
    meg_subjects: np.ndarray,
    time_points: np.ndarray,
    stimulus_order: Sequence[Path],
) -> dict[str, Any]:
    """Build a cache key describing MEG export inputs."""

    stim_payload = "\n".join(str(path.resolve()) for path in stimulus_order).encode("utf-8")
    time_payload = np.asarray(time_points, dtype=np.float64).tobytes()
    return {
        "version": 1,
        "meg_bundle": _path_signature(meg_bundle_path),
        "stimulus_order_source": stimulus_source,
        "stimulus_order_sha256": _sha256_bytes(stim_payload),
        "time_points_sha256": _sha256_bytes(time_payload),
        "meg_subjects_shape": list(map(int, meg_subjects.shape)),
        "subject_count": int(meg_subjects.shape[3]),
    }


def _parse_group(path: Path) -> tuple[int, str, str]:
    """Extract numeric folder prefix and human-readable group label from path."""

    parent = path.parent.name
    if "_" in parent:
        prefix, rest = parent.split("_", 1)
        if prefix.isdigit():
            return int(prefix), parent, rest
    return 999, parent, parent


def compute_contiguous_groups(paths: Sequence[Path]) -> list[tuple[int, int, str]]:
    """Compute contiguous category blocks from ordered stimulus paths."""

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


def _plot_index_grid(image_paths: Sequence[Path], out_path: Path) -> None:
    """Render a thumbnail grid that maps stimulus index to image file."""

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


def _plot_into_axis(
    ax: Any,
    matrix: np.ndarray,
    title: str,
    groups: list[tuple[int, int, str]],
    vmin: float | None,
    vmax: float | None,
) -> None:
    """Draw one RDM matrix into an existing axis with category guides."""

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


def _plot_meg_snapshots(
    meg: np.ndarray,
    time_points: np.ndarray,
    out_path: Path,
    groups: list[tuple[int, int, str]],
) -> None:
    """Plot MEG RDM snapshots at representative time points."""

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


def _plot_meg_summary(meg: np.ndarray, time_points: np.ndarray, out_path: Path) -> None:
    """Plot summary statistics of MEG pairwise dissimilarities over time."""

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
    """Create a time-lapse GIF of MEG RDM evolution."""

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
    logger: Callable[[str], None],
) -> None:
    """Export per-subject MEG RDM artifacts."""

    data_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    gif_root.mkdir(parents=True, exist_ok=True)

    n_subjects = int(meg_subjects.shape[3])
    with (data_root / "subject_index.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_index", "subject_label"])
        for subject_idx in range(n_subjects):
            writer.writerow([subject_idx, _subject_label(subject_idx)])

    for subject_idx in range(n_subjects):
        label = _subject_label(subject_idx)
        logger(f"  MEG per-subject assets {subject_idx + 1}/{n_subjects}: {label}")
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


def _required_meg_artifacts_exist(root: Path, n_subjects: int) -> bool:
    """Return True when shared MEG artifacts are fully present."""

    required = [
        root / "data" / "meg_rdm_avg.npy",
        root / "data" / "meg_rdm_subjects.npy",
        root / "data" / "time_ms.npy",
        root / "data" / "stimulus_order_used.csv",
        root / "plots" / "meg_rdm_snapshots.png",
        root / "plots" / "meg_rdm_summary_over_time.png",
        root / "plots" / "meg_rdm_index_grid.png",
        root / "meg_rdm_animation.gif",
        root / "meg_rdm_index_map.csv",
        root / "data" / "per_subject" / "subject_index.csv",
    ]
    if any(not path.exists() for path in required):
        return False
    for subject_idx in range(n_subjects):
        subject = _subject_label(subject_idx)
        per_subject_required = [
            root / "data" / "per_subject" / subject / "meg_rdm.npy",
            root / "data" / "per_subject" / subject / "time_ms.npy",
            root / "data" / "per_subject" / subject / "stimulus_order_used.csv",
            root / "plots" / "per_subject" / subject / "meg_rdm_snapshots.png",
            root / "plots" / "per_subject" / subject / "meg_rdm_summary_over_time.png",
            root / "per_subject" / subject / "meg_rdm_animation.gif",
        ]
        if any(not path.exists() for path in per_subject_required):
            return False
    return True


def _build_shared_meg_assets(
    shared_root: Path,
    meg_avg: np.ndarray,
    meg_subjects: np.ndarray,
    time_points: np.ndarray,
    stimulus_order: Sequence[Path],
    stimulus_source: str,
    groups: list[tuple[int, int, str]],
    logger: Callable[[str], None],
) -> None:
    """Build MEG data/plot assets in a shared root."""

    if shared_root.exists():
        shutil.rmtree(shared_root)
    (shared_root / "data").mkdir(parents=True, exist_ok=True)
    (shared_root / "plots").mkdir(parents=True, exist_ok=True)
    (shared_root / "per_subject").mkdir(parents=True, exist_ok=True)

    np.save(shared_root / "data" / "meg_rdm_avg.npy", meg_avg)
    np.save(shared_root / "data" / "meg_rdm_subjects.npy", meg_subjects)
    np.save(shared_root / "data" / "time_ms.npy", time_points)
    _save_stimulus_order(
        stimulus_order,
        shared_root / "data" / "stimulus_order_used.csv",
        source=stimulus_source,
    )
    with (shared_root / "meg_rdm_index_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "file_name"])
        for idx, path in enumerate(stimulus_order):
            writer.writerow([idx, str(path)])

    _plot_index_grid(stimulus_order, shared_root / "plots" / "meg_rdm_index_grid.png")
    _plot_meg_snapshots(meg_avg, time_points, shared_root / "plots" / "meg_rdm_snapshots.png", groups)
    _plot_meg_summary(meg_avg, time_points, shared_root / "plots" / "meg_rdm_summary_over_time.png")
    _save_meg_animation(meg_avg, time_points, shared_root / "meg_rdm_animation.gif", groups)
    _export_meg_per_subject_assets(
        meg_subjects=meg_subjects,
        time_points=time_points,
        stimulus_order=stimulus_order,
        groups=groups,
        data_root=shared_root / "data" / "per_subject",
        plots_root=shared_root / "plots" / "per_subject",
        gif_root=shared_root / "per_subject",
        logger=logger,
    )


def prepare_shared_meg_assets(
    images_dir: Path,
    allow_build: bool,
    logger: Callable[[str], None],
    meg_bundle_path: Path = DEFAULT_MEG_BUNDLE_PATH,
    shared_root: Path = DEFAULT_MEG_PRECOMPUTED_ROOT,
) -> MegPrepared:
    """Ensure shared MEG assets are available, optionally building them."""

    if not meg_bundle_path.exists():
        if not allow_build:
            raise SystemExit(
                f"Missing MEG bundle: {meg_bundle_path}. "
                "Create it first with scripts/export_meg_bundle.py."
            )
        required_sources = [
            DEFAULT_MEG_RDM_MAT_PATH,
            DEFAULT_MEG_TIME_MAT_PATH,
            DEFAULT_MEG_IMAGE_STRUCT_PATH,
        ]
        missing_sources = [path for path in required_sources if not path.exists()]
        if missing_sources:
            missing = ", ".join(str(path) for path in missing_sources)
            raise SystemExit(
                "MEG bundle is missing and automatic creation cannot proceed. "
                f"Missing source files: {missing}"
            )
        if not images_dir.exists():
            raise SystemExit(f"Image directory not found for automatic MEG bundle creation: {images_dir}")
        logger(f"MEG bundle missing. Building it at {meg_bundle_path} from MATLAB sources...")
        build_meg_bundle(
            meg_rdm_path=DEFAULT_MEG_RDM_MAT_PATH,
            time_mat_path=DEFAULT_MEG_TIME_MAT_PATH,
            image_struct_path=DEFAULT_MEG_IMAGE_STRUCT_PATH,
            images_dir=images_dir,
            output_path=meg_bundle_path,
        )
        logger("MEG bundle build complete.")

    logger(f"Loading unified MEG bundle from {meg_bundle_path}...")
    meg_avg, meg_subjects, time_points, stimulus_order = load_meg_bundle(meg_bundle_path, images_dir)
    stimulus_source = f"{meg_bundle_path.resolve()}::stimulus_file_name"
    groups = compute_contiguous_groups(stimulus_order)

    cache_key = _compute_meg_subject_cache_key(
        meg_bundle_path=meg_bundle_path,
        stimulus_source=stimulus_source,
        meg_subjects=meg_subjects,
        time_points=time_points,
        stimulus_order=stimulus_order,
    )
    cache_key_path = shared_root / "cache_meta.json"
    n_subjects = int(meg_subjects.shape[3])

    if cache_key_path.exists():
        try:
            cached = json.loads(cache_key_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cached = None
        if isinstance(cached, dict):
            cached_key = cached.get("cache_key")
            if isinstance(cached_key, dict) and cached_key == cache_key:
                if _required_meg_artifacts_exist(shared_root, n_subjects=n_subjects):
                    logger(f"Using precomputed MEG assets from {shared_root}")
                    return MegPrepared(
                        meg_avg=meg_avg,
                        meg_subjects=meg_subjects,
                        time_points=time_points,
                        stimulus_order=stimulus_order,
                        stimulus_source=stimulus_source,
                        shared_root=shared_root,
                        shared_from_cache=True,
                    )

    if not allow_build:
        raise SystemExit(
            f"Precomputed MEG assets are missing or outdated at {shared_root}. "
            "Run scripts/prepare_meg_assets.py once to build them."
        )

    logger(f"Building precomputed MEG assets in {shared_root}...")
    _build_shared_meg_assets(
        shared_root=shared_root,
        meg_avg=meg_avg,
        meg_subjects=meg_subjects,
        time_points=time_points,
        stimulus_order=stimulus_order,
        stimulus_source=stimulus_source,
        groups=groups,
        logger=logger,
    )
    cache_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "cache_key": cache_key,
    }
    cache_key_path.write_text(json.dumps(cache_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return MegPrepared(
        meg_avg=meg_avg,
        meg_subjects=meg_subjects,
        time_points=time_points,
        stimulus_order=stimulus_order,
        stimulus_source=stimulus_source,
        shared_root=shared_root,
        shared_from_cache=False,
    )


def sync_shared_meg_assets(shared_root: Path, run_meg_root: Path) -> None:
    """Copy precomputed shared MEG artifacts into one run output tree."""

    if run_meg_root.exists():
        shutil.rmtree(run_meg_root)
    shutil.copytree(shared_root, run_meg_root)
