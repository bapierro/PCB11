#!/usr/bin/env python3
"""
RSA (Representational Similarity Analysis) computation.

Correlates pre-computed RDMs from ANN layers with MEG RDMs to identify
which layer best explains brain representations across time.

Supports:
- 2D MEG RDMs (36x36): single RDM → bar plot
- 3D MEG RDMs (36x36x1201): time-resolved RDMs → time course plots
- 4D MEG RDMs (36x36x1201x20): automatically averaged across subjects

Example (with 4D MEG data from OSF):
  python scripts/rsa_analysis.py \
    --meg-rdm data/meg/MEGRDMs_2D.mat \
    --ann-rdms outputs/rdms \
    --output outputs/rsa_results \
    --metric spearman \
    --plot

Example (with 2D MEG RDM):
  python scripts/rsa_analysis.py \
    --meg-rdm outputs/meg_rdm_avg.npy \
    --ann-rdms outputs/rdms \
    --output outputs/rsa_results
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.stats import spearmanr, pearsonr, kendalltau
except Exception as exc:  # pragma: no cover
    raise SystemExit("scipy not available. Install it with: pip install scipy") from exc

try:
    import scipy.io as sio
except Exception as exc:  # pragma: no cover
    raise SystemExit("scipy.io not available") from exc


def _load_meg_rdm(meg_path: Path) -> np.ndarray:
    """Load MEG RDM from .npy or .mat file.
    
    Supports both:
    - 2D RDM (36x36): single RDM
    - 4D RDM (36x36x1201x20): averaged across subjects to get (36x36x1201)
    """
    if meg_path.suffix == ".npy":
        rdm = np.load(meg_path)
    elif meg_path.suffix == ".mat":
        mat_data = sio.loadmat(str(meg_path))
        # Try common variable names
        rdm = None
        for key in ["rdm", "RDM", "meg_rdm", "MEGRDMs", "data"]:
            if key in mat_data:
                rdm = mat_data[key]
                break
        # If no match, use first non-metadata key
        if rdm is None:
            for key, val in mat_data.items():
                if not key.startswith("__") and isinstance(val, np.ndarray):
                    rdm = val
                    break
        if rdm is None:
            raise ValueError(f"No RDM data found in {meg_path}")
    else:
        raise ValueError(f"Unsupported file format: {meg_path.suffix}")

    # Handle 4D RDM (36x36x1201x20): average across subjects (last dimension)
    if rdm.ndim == 4:
        print(f"  4D MEG RDM detected (shape: {rdm.shape})")
        print(f"  Averaging across {rdm.shape[3]} subjects...")
        rdm = np.mean(rdm, axis=3)  # Average over subjects dimension
        print(f"  Result shape: {rdm.shape}")
    
    return rdm


def _iterate_rdm_files(rdm_dir: Path) -> List[Tuple[str, Path]]:
    """Find all *_rdm.npy files and extract layer names."""
    files = sorted(
        (p.stem.replace("_rdm", ""), p)
        for p in rdm_dir.glob("*_rdm.npy")
        if p.is_file()
    )
    if not files:
        raise SystemExit(f"No *_rdm.npy files found in {rdm_dir}")
    return files


def _flatten_rdm(rdm: np.ndarray) -> np.ndarray:
    """Convert RDM matrix to 1D upper-triangle vector (excluding diagonal)."""
    n = rdm.shape[0]
    upper_tri_indices = np.triu_indices(n, k=1)
    return rdm[upper_tri_indices]


def _compute_correlation(rdm1: np.ndarray, rdm2: np.ndarray, method: str) -> float:
    """Compute correlation between two RDMs (or their flattened upper triangles)."""
    # Flatten RDMs to 1D vectors (upper triangle)
    flat1 = _flatten_rdm(rdm1)
    flat2 = _flatten_rdm(rdm2)

    if method == "spearman":
        corr, _ = spearmanr(flat1, flat2)
    elif method == "pearson":
        corr, _ = pearsonr(flat1, flat2)
    elif method == "kendall":
        corr, _ = kendalltau(flat1, flat2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return float(corr)


def _plot_rsa_results(
    layer_names: List[str],
    correlations: List[float],
    output_path: Path,
) -> None:
    """Plot RSA correlation results as a bar plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(layer_names)))
    bars = ax.barh(layer_names, correlations, color=colors)

    ax.set_xlabel("Spearman Correlation with MEG RDM")
    ax.set_ylabel("ANN Layer")
    ax.set_title("RSA: ANN Layers vs MEG Brain Responses")
    ax.set_xlim([-1, 1])

    # Add value labels on bars
    for bar, corr in zip(bars, correlations):
        width = bar.get_width()
        ax.text(
            width + 0.02 if width > 0 else width - 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{corr:.3f}",
            ha="left" if width > 0 else "right",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_rsa_timecourse(
    layer_names: List[str],
    correlations_by_time: np.ndarray,
    time_vector: np.ndarray | None,
    output_path: Path,
) -> None:
    """Plot RSA correlation time course for each layer."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, layer_name in enumerate(layer_names):
        ax.plot(time_vector if time_vector is not None else range(correlations_by_time.shape[0]),
                correlations_by_time[:, i],
                label=layer_name,
                linewidth=2)

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Stimulus onset")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Spearman Correlation with MEG RDM")
    ax.set_title("RSA Time Course: ANN Layers vs MEG")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_results_table(
    layer_names: List[str],
    correlations: List[float],
    output_path: Path,
) -> None:
    """Save RSA results as a tab-separated table."""
    lines = ["Layer\tSpearman_Correlation\n"]
    for name, corr in zip(layer_names, correlations):
        lines.append(f"{name}\t{corr:.6f}\n")
    output_path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RSA between ANN layers and MEG data."
    )
    parser.add_argument(
        "--meg-rdm",
        type=Path,
        required=True,
        help="Path to MEG RDM (.npy or .mat file).",
    )
    parser.add_argument(
        "--time-vector",
        type=Path,
        default=None,
        help="Path to time vector (time.mat). If None, uses data/meg/time.mat.",
    )
    parser.add_argument(
        "--ann-rdms",
        type=Path,
        required=True,
        help="Directory containing ANN layer RDMs (*_rdm.npy files).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rsa_results"),
        help="Output directory for results.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="spearman",
        choices=["spearman", "pearson", "kendall"],
        help="Correlation metric for RDM comparison.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save visualization plots.",
    )
    args = parser.parse_args()

    # Create output directory
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load MEG RDM
    print(f"Loading MEG RDM from {args.meg_rdm}...")
    try:
        meg_rdm = _load_meg_rdm(args.meg_rdm)
    except Exception as e:
        raise SystemExit(f"Failed to load MEG RDM: {e}") from e

    print(f"  MEG RDM shape: {meg_rdm.shape}")

    # Validate MEG RDM
    if meg_rdm.ndim not in (2, 3):
        raise SystemExit(
            f"MEG RDM must be 2D or 3D, got shape {meg_rdm.shape}"
        )
    
    if meg_rdm.shape[0] != meg_rdm.shape[1]:
        raise SystemExit(
            f"First two dimensions of MEG RDM must be square, got {meg_rdm.shape[:2]}"
        )

    # Load time vector if 3D
    time_vector = None
    if meg_rdm.ndim == 3:
        time_mat_path = args.time_vector or Path("data/meg/time.mat")
        if time_mat_path.exists():
            try:
                time_data = sio.loadmat(str(time_mat_path))
                time_vector = time_data["time"].flatten()
                print(f"  Loaded time vector: {time_vector.shape[0]} timepoints")
            except Exception as e:
                print(f"  ⚠️  Could not load time vector: {e}")

    # Load ANN RDMs
    print(f"\nLoading ANN RDMs from {args.ann_rdms}...")
    rdm_files = _iterate_rdm_files(args.ann_rdms)
    print(f"  Found {len(rdm_files)} layer RDMs")

    # Compute correlations
    print(f"\nComputing {args.metric} correlations...")
    layer_names: List[str] = []
    correlations: List[float] = []
    
    # For 3D MEG, store time course
    correlations_by_time: np.ndarray | None = None
    if meg_rdm.ndim == 3:
        correlations_by_time = np.zeros((meg_rdm.shape[2], len(rdm_files)))

    for layer_idx, (layer_name, rdm_path) in enumerate(rdm_files):
        ann_rdm = np.load(rdm_path)

        if ann_rdm.ndim != 2 or ann_rdm.shape[0] != ann_rdm.shape[1]:
            print(
                f"  ⚠️  Skipping {layer_name}: invalid shape {ann_rdm.shape}"
            )
            continue

        # Ensure RDMs have compatible sizes
        if ann_rdm.shape[0] != meg_rdm.shape[0]:
            print(
                f"  ⚠️  Size mismatch for {layer_name}: "
                f"ANN={ann_rdm.shape}, MEG={meg_rdm.shape[:2]}"
            )
            continue

        layer_names.append(layer_name)

        # Compute correlation(s)
        if meg_rdm.ndim == 2:
            # Single RDM
            corr = _compute_correlation(meg_rdm, ann_rdm, args.metric)
            correlations.append(corr)
            print(f"  {layer_name}: {corr:+.4f}")
        else:
            # 3D MEG: compute time course
            print(f"  {layer_name}...", end=" ", flush=True)
            for t in range(meg_rdm.shape[2]):
                corr = _compute_correlation(meg_rdm[:, :, t], ann_rdm, args.metric)
                correlations_by_time[t, layer_idx] = corr
            # Use mean correlation as summary
            mean_corr = np.mean(correlations_by_time[:, layer_idx])
            correlations.append(mean_corr)
            print(f"mean={mean_corr:+.4f}")

    if not correlations:
        raise SystemExit("No valid correlations computed!")

    # Save results
    print(f"\nSaving results to {output_dir}...")
    
    # Summary table
    if meg_rdm.ndim == 2:
        results_table = output_dir / f"rsa_results_{args.metric}.tsv"
        _save_results_table(layer_names, correlations, results_table)
        print(f"  ✅ Results table: {results_table.name}")
    else:
        # Time course data
        timecourse_file = output_dir / f"rsa_timecourse_{args.metric}.npy"
        np.save(timecourse_file, correlations_by_time)
        print(f"  ✅ Time course data: {timecourse_file.name}")
        
        # Summary statistics at each timepoint
        summary_file = output_dir / f"rsa_summary_{args.metric}.npy"
        np.save(summary_file, correlations)
        print(f"  ✅ Summary (mean correlations): {summary_file.name}")

    if args.plot:
        if meg_rdm.ndim == 2:
            results_plot = output_dir / f"rsa_results_{args.metric}.png"
            _plot_rsa_results(layer_names, correlations, results_plot)
        else:
            results_plot = output_dir / f"rsa_timecourse_{args.metric}.png"
            _plot_rsa_timecourse(layer_names, correlations_by_time, time_vector, results_plot)
        print(f"  ✅ Plot: {results_plot.name}")

    # Print summary
    best_idx = np.argmax(np.abs(correlations))
    print(f"\n{'='*50}")
    print(f"RSA Analysis Complete!")
    print(f"Best matching layer: {layer_names[best_idx]}")
    print(f"  Mean Correlation: {correlations[best_idx]:+.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
