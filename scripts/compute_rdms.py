#!/usr/bin/env python3
"""
Compute model RDMs from saved feature .npy files (one per layer).

Example:
  python scripts/compute_rdms.py \
    --features-dir features/test_gap \
    --output outputs/rdms \
    --method correlation
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from thingsvision.core.rsa import compute_rdm
except Exception as exc:  # pragma: no cover - dependency error path
    raise SystemExit(
        "thingsvision RSA helpers not available. Ensure thingsvision is installed (poetry install)."
    ) from exc


def _iter_feature_files(features_dir: Path) -> list[Path]:
    files = sorted(p for p in features_dir.glob("*.npy") if p.is_file())
    if not files:
        raise SystemExit(f"No .npy feature files found in {features_dir}")
    return files


def _load_file_names(features_dir: Path, file_names_path: Path | None) -> list[str] | None:
    if file_names_path is None:
        candidate = features_dir / "file_names.txt"
        if candidate.exists():
            file_names_path = candidate
    if file_names_path is None or not file_names_path.exists():
        return None
    lines = [l.strip() for l in file_names_path.read_text().splitlines() if l.strip()]
    return lines if lines else None


def _parse_group(name: str) -> tuple[int, str, str]:
    parent = Path(name).parent.name
    if "_" in parent:
        prefix, rest = parent.split("_", 1)
        if prefix.isdigit():
            return int(prefix), parent, rest
    return 999, parent, parent


def _compute_sort_order(file_names: list[str]) -> tuple[list[int], list[str], list[tuple[int, int, str]]]:
    items = []
    for idx, name in enumerate(file_names):
        order_num, group_full, group_label = _parse_group(name)
        base = Path(name).name
        items.append((idx, order_num, group_full, group_label, base))

    items_sorted = sorted(items, key=lambda x: (x[1], x[2], x[4]))
    order_idx = [i[0] for i in items_sorted]

    # group boundaries and labels in sorted order
    groups: list[tuple[int, int, str]] = []
    start = 0
    while start < len(items_sorted):
        gname = items_sorted[start][3]
        end = start
        while end < len(items_sorted) and items_sorted[end][3] == gname:
            end += 1
        groups.append((start, end, gname))
        start = end

    ordered_names = [file_names[i] for i in order_idx]
    return order_idx, ordered_names, groups


def _plot_matrix(
    matrix: np.ndarray,
    title: str,
    out_path: Path,
    groups: list[tuple[int, int, str]] | None = None,
) -> None:
    try:
        import seaborn as sns  # type: ignore
    except Exception:
        sns = None

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    if sns is not None:
        sns.heatmap(matrix, ax=ax, cmap="viridis", square=True)
    else:
        im = ax.imshow(matrix, cmap="viridis")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("Image")
    ax.set_ylabel("Image")

    if groups:
        boundaries = [g[0] for g in groups] + [groups[-1][1]]
        for b in boundaries[1:-1]:
            pos = b - 0.5
            ax.axhline(pos, color="white", linewidth=1.0)
            ax.axvline(pos, color="white", linewidth=1.0)

        centers = [(g[0] + g[1] - 1) / 2 for g in groups]
        labels = [g[2] for g in groups]
        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute RDMs from model features.")
    parser.add_argument(
        "--features-dir",
        type=Path,
        required=True,
        help="Directory containing .npy feature files (one per layer).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/rdms"),
        help="Output directory for RDM .npy files.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="correlation",
        help="RDM computation method for features (thingsvision).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save a heatmap plot for each RDM.",
    )
    parser.add_argument(
        "--file-names",
        type=Path,
        default=None,
        help="Optional file_names.txt path to sort and annotate by parent folder.",
    )
    parser.add_argument(
        "--sort-by-group",
        action="store_true",
        help="Sort images by parent folder (e.g., 1_Residence, 2_Nature).",
    )
    args = parser.parse_args()

    features_dir: Path = args.features_dir
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    file_names = _load_file_names(features_dir, args.file_names)
    order_idx = None
    ordered_names = None
    groups = None
    if args.sort_by_group and file_names:
        order_idx, ordered_names, groups = _compute_sort_order(file_names)

        order_path = output_dir / "rdm_order.txt"
        order_path.write_text("\n".join(ordered_names) + "\n", encoding="utf-8")

    feature_files = _iter_feature_files(features_dir)
    for feat_path in feature_files:
        features = np.load(feat_path)
        if features.ndim != 2:
            raise SystemExit(f"Expected 2D features in {feat_path}, got shape {features.shape}")

        rdm = compute_rdm(features, method=args.method)
        if order_idx is not None:
            rdm = rdm[np.ix_(order_idx, order_idx)]
        out_path = output_dir / f"{feat_path.stem}_rdm.npy"
        np.save(out_path, rdm)
        print(f"{feat_path.name} -> {out_path.name} (shape={rdm.shape})")
        if args.plot:
            out_plot = output_dir / f"{feat_path.stem}_rdm.png"
            _plot_matrix(rdm, f"RDM: {feat_path.stem}", out_plot, groups=groups)

    print(f"Saved RDMs to {output_dir}")


if __name__ == "__main__":
    main()
