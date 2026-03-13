"""Cross-run spatial-vs-semantic RSA comparison on a relative compute axis."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.io import loadmat

from .model_compute import ComputeAxis, build_relative_compute_axis
from .rsa_outputs import compute_model_rsa

import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class RunComparisonData:
    run_root: Path
    model_name: str
    module_names: list[str]
    saved_layer_names: list[str]
    spatial_corr: np.ndarray
    semantic_corr: np.ndarray
    compute_axis: ComputeAxis


def _load_manifest(run_root: Path) -> dict:
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_saved_layer_names(rdm_data_dir: Path) -> list[str]:
    order_path = rdm_data_dir / "layer_order.csv"
    if not order_path.exists():
        raise FileNotFoundError(f"Missing layer_order.csv under {rdm_data_dir}")
    out: list[str] = []
    with order_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            out.append(row["layer_name"])
    return out


def _load_spatial_semantic_from_csv(csv_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not csv_path.exists():
        return None
    spatial: list[float] = []
    semantic: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if "spatial_spearman" not in reader.fieldnames or "semantic_spearman" not in reader.fieldnames:
            return None
        for row in reader:
            spatial.append(float(row["spatial_spearman"]))
            semantic.append(float(row["semantic_spearman"]))
    return np.asarray(spatial, dtype=np.float64), np.asarray(semantic, dtype=np.float64)


def _compute_spatial_semantic_from_rdms(
    rdm_data_dir: Path,
    saved_layer_names: Sequence[str],
    spatial_model_rdm: np.ndarray,
    semantic_model_rdm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    layer_rdms: list[np.ndarray] = []
    for layer_name in saved_layer_names:
        rdm_path = rdm_data_dir / f"{layer_name}_rdm.npy"
        if not rdm_path.exists():
            raise FileNotFoundError(f"Missing layer RDM file: {rdm_path}")
        layer_rdms.append(np.load(rdm_path))
    spatial = compute_model_rsa(layer_rdms, spatial_model_rdm)
    semantic = compute_model_rsa(layer_rdms, semantic_model_rdm)
    return spatial, semantic


def load_run_comparison(run_root: Path, input_size: int = 224) -> RunComparisonData:
    manifest = _load_manifest(run_root)
    model_name = manifest["config"]["model"]
    module_names = list(manifest["resolved"]["model_layers"])
    rdm_data_dir = Path(manifest["paths"]["rdm_model_data"])
    saved_layer_names = _load_saved_layer_names(rdm_data_dir)
    if len(module_names) != len(saved_layer_names):
        raise ValueError(
            f"Layer count mismatch for {run_root}: manifest has {len(module_names)}, RDM order has {len(saved_layer_names)}."
        )

    compute_axis = build_relative_compute_axis(model_name=model_name, layer_labels=module_names, input_size=input_size)
    csv_values = _load_spatial_semantic_from_csv(Path(manifest["paths"]["rsa_data"]) / "rsa_spatial_vs_semantic.csv")
    if csv_values is None:
        spatial_model_rdm = loadmat(Path("data/meg/structureRDM_sq.mat"), squeeze_me=True)["RDM"]["RDM"].item()
        semantic_model_rdm = loadmat(Path("data/meg/semanticRDM_sq.mat"), squeeze_me=True)["RDM"]["RDM"].item()
        spatial_corr, semantic_corr = _compute_spatial_semantic_from_rdms(
            rdm_data_dir=rdm_data_dir,
            saved_layer_names=saved_layer_names,
            spatial_model_rdm=spatial_model_rdm,
            semantic_model_rdm=semantic_model_rdm,
        )
    else:
        spatial_corr, semantic_corr = csv_values

    if spatial_corr.shape != semantic_corr.shape or spatial_corr.shape[0] != len(module_names):
        raise ValueError(
            f"Unexpected curve shapes for {run_root}: spatial={spatial_corr.shape}, semantic={semantic_corr.shape}, "
            f"layers={len(module_names)}."
        )

    return RunComparisonData(
        run_root=run_root,
        model_name=model_name,
        module_names=module_names,
        saved_layer_names=saved_layer_names,
        spatial_corr=spatial_corr,
        semantic_corr=semantic_corr,
        compute_axis=compute_axis,
    )


def _normalize_curve(values: np.ndarray) -> np.ndarray:
    peak = float(np.nanmax(values))
    if not np.isfinite(peak) or peak <= 0.0:
        return np.full_like(values, np.nan, dtype=np.float64)
    return values / peak


def _interpolate_curve(x: np.ndarray, y: np.ndarray, grid: np.ndarray) -> np.ndarray:
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return np.full_like(grid, np.nan, dtype=np.float64)
    x_valid = x[valid]
    y_valid = y[valid]
    order = np.argsort(x_valid)
    x_valid = x_valid[order]
    y_valid = y_valid[order]
    unique_x, unique_indices = np.unique(x_valid, return_index=True)
    y_valid = y_valid[unique_indices]
    if unique_x.size < 2:
        return np.full_like(grid, np.nan, dtype=np.float64)
    return np.interp(grid, unique_x, y_valid, left=np.nan, right=np.nan)


def _mean_and_sem(curves: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    stack = np.stack(curves, axis=0)
    counts = np.sum(np.isfinite(stack), axis=0)
    sums = np.nansum(stack, axis=0)
    mean = np.divide(sums, counts, out=np.full(stack.shape[1], np.nan, dtype=np.float64), where=counts > 0)
    centered = stack - mean
    centered[~np.isfinite(stack)] = np.nan
    sq_sums = np.nansum(np.square(centered), axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = np.divide(
            sq_sums,
            counts - 1,
            out=np.full(stack.shape[1], np.nan, dtype=np.float64),
            where=counts > 1,
        )
        sem = np.sqrt(variance / counts)
    return mean, sem


def _plot_average_curves(
    runs: Sequence[RunComparisonData],
    output_path: Path,
    normalize_y: bool,
    grid_size: int,
) -> None:
    grid = np.linspace(0.0, 1.0, grid_size, dtype=np.float64)
    spatial_curves: list[np.ndarray] = []
    semantic_curves: list[np.ndarray] = []

    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    for run in runs:
        x = run.compute_axis.relative_positions
        spatial = _normalize_curve(run.spatial_corr) if normalize_y else run.spatial_corr
        semantic = _normalize_curve(run.semantic_corr) if normalize_y else run.semantic_corr
        ax.plot(x, spatial, color="#d62728", alpha=0.18, linewidth=1.2)
        ax.plot(x, semantic, color="#1f77b4", alpha=0.18, linewidth=1.2)
        spatial_curves.append(_interpolate_curve(x, spatial, grid))
        semantic_curves.append(_interpolate_curve(x, semantic, grid))

    spatial_mean, spatial_sem = _mean_and_sem(spatial_curves)
    semantic_mean, semantic_sem = _mean_and_sem(semantic_curves)

    ax.plot(grid, spatial_mean, color="#d62728", linewidth=3.0, label="Spatial mean")
    ax.fill_between(grid, spatial_mean - spatial_sem, spatial_mean + spatial_sem, color="#d62728", alpha=0.15)
    ax.plot(grid, semantic_mean, color="#1f77b4", linewidth=3.0, label="Semantic mean")
    ax.fill_between(grid, semantic_mean - semantic_sem, semantic_mean + semantic_sem, color="#1f77b4", alpha=0.15)

    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Relative Cumulative Compute")
    ax.set_ylabel("Normalized correlation" if normalize_y else "Spearman correlation")
    ax.set_title(
        "Spatial vs Semantic RSA across models"
        + (" (peak-normalized)" if normalize_y else " (raw)")
    )
    ax.grid(axis="both", linestyle=":", alpha=0.45)
    ax.legend(loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _peak_position(values: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    if values.size == 0 or not np.any(np.isfinite(values)):
        return np.nan, np.nan
    peak_index = int(np.nanargmax(values))
    return float(x[peak_index]), float(values[peak_index])


def _write_peak_summary(runs: Sequence[RunComparisonData], output_dir: Path) -> None:
    rows: list[dict[str, float | str]] = []
    for run in runs:
        spatial_peak_x, spatial_peak_y = _peak_position(run.spatial_corr, run.compute_axis.relative_positions)
        semantic_peak_x, semantic_peak_y = _peak_position(run.semantic_corr, run.compute_axis.relative_positions)
        rows.append(
            {
                "run_root": str(run.run_root.resolve()),
                "model": run.model_name,
                "spatial_peak_relative_compute": spatial_peak_x,
                "semantic_peak_relative_compute": semantic_peak_x,
                "delta_peak_relative_compute": semantic_peak_x - spatial_peak_x,
                "spatial_peak_corr": spatial_peak_y,
                "semantic_peak_corr": semantic_peak_y,
                "delta_peak_corr": semantic_peak_y - spatial_peak_y,
            }
        )

    csv_path = output_dir / "peak_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    delta_x = np.asarray([row["delta_peak_relative_compute"] for row in rows], dtype=np.float64)
    spatial_x = np.asarray([row["spatial_peak_relative_compute"] for row in rows], dtype=np.float64)
    semantic_x = np.asarray([row["semantic_peak_relative_compute"] for row in rows], dtype=np.float64)
    payload = {
        "n_runs": len(rows),
        "mean_spatial_peak_relative_compute": float(np.nanmean(spatial_x)),
        "mean_semantic_peak_relative_compute": float(np.nanmean(semantic_x)),
        "mean_delta_peak_relative_compute": float(np.nanmean(delta_x)),
        "median_delta_peak_relative_compute": float(np.nanmedian(delta_x)),
        "fraction_semantic_peak_later": float(np.nanmean(delta_x > 0)),
    }
    (output_dir / "peak_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _plot_peak_locations(runs: Sequence[RunComparisonData], output_path: Path) -> None:
    fig_h = max(4.5, 0.45 * len(runs) + 1.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_h))
    y_positions = np.arange(len(runs), dtype=np.float64)
    labels = [run.model_name for run in runs]

    for idx, run in enumerate(runs):
        spatial_peak_x, _ = _peak_position(run.spatial_corr, run.compute_axis.relative_positions)
        semantic_peak_x, _ = _peak_position(run.semantic_corr, run.compute_axis.relative_positions)
        ax.plot([spatial_peak_x, semantic_peak_x], [idx, idx], color="0.75", linewidth=1.4, zorder=1)
        ax.scatter(spatial_peak_x, idx, color="#d62728", s=45, zorder=2)
        ax.scatter(semantic_peak_x, idx, color="#1f77b4", s=45, zorder=2)

    ax.set_xlim(0.0, 1.0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Peak Relative Compute")
    ax.set_title("Peak positions by model")
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.legend(
        handles=[
            plt.Line2D([], [], color="#d62728", marker="o", linestyle="", label="Spatial peak"),
            plt.Line2D([], [], color="#1f77b4", marker="o", linestyle="", label="Semantic peak"),
        ],
        loc="best",
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def discover_run_roots(run_paths: Iterable[Path]) -> list[Path]:
    resolved: list[Path] = []
    for path in run_paths:
        if path.is_file():
            raise ValueError(f"Expected run directory, got file: {path}")
        manifest_path = path / "run_manifest.json"
        if manifest_path.exists():
            resolved.append(path)
    if not resolved:
        raise ValueError("No run directories with run_manifest.json were found.")
    return sorted(resolved)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate spatial vs semantic RSA curves across model runs on a relative-compute axis."
    )
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="*",
        default=None,
        help="Run directories to aggregate. Default: all outputs/run_* directories with run_manifest.json.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/cross_model_rsa"),
        help="Directory where aggregate plots and CSV summaries are written.",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        default=224,
        help="Image size assumption used for compute estimates (default: 224).",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=200,
        help="Number of interpolation points for the average curves.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    run_paths = args.runs if args.runs else sorted(Path("outputs").glob("run_*"))
    run_roots = discover_run_roots(run_paths)
    runs = [load_run_comparison(run_root, input_size=args.input_size) for run_root in run_roots]

    args.output_root.mkdir(parents=True, exist_ok=True)
    raw_plot = args.output_root / "spatial_vs_semantic_relative_compute_raw.png"
    normalized_plot = args.output_root / "spatial_vs_semantic_relative_compute_normalized.png"
    peak_plot = args.output_root / "spatial_vs_semantic_peak_locations.png"
    peak_csv = args.output_root / "peak_summary.csv"
    peak_json = args.output_root / "peak_summary.json"

    _plot_average_curves(
        runs=runs,
        output_path=raw_plot,
        normalize_y=False,
        grid_size=args.grid_size,
    )
    _plot_average_curves(
        runs=runs,
        output_path=normalized_plot,
        normalize_y=True,
        grid_size=args.grid_size,
    )
    _plot_peak_locations(runs, peak_plot)
    _write_peak_summary(runs, args.output_root)
    print(f"Cross-model RSA outputs written to: {args.output_root.resolve()}")
    print(f"  Raw plot: {raw_plot.resolve()}")
    print(f"  Normalized plot: {normalized_plot.resolve()}")
    print(f"  Peak locations: {peak_plot.resolve()}")
    print(f"  Peak summary CSV: {peak_csv.resolve()}")
    print(f"  Peak summary JSON: {peak_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
