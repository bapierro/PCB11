from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import matplotlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    confusion_matrix,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
import umap

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class ClusterConfig:
    features_dir: Path
    output_dir: Path | None = None
    layers: list[str] | None = None
    n_neighbors: int = 10
    min_dist: float = 0.1
    metric: str = "euclidean"
    random_state: int = 42
    k: int | None = None
    model_name: str | None = None
    cluster_space: Literal["umap", "feature"] = "umap"


def _log(message: str) -> None:
    print(f"[cluster] {message}", flush=True)


def _sanitize_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise SystemExit(f"Missing required file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_class_token(parent_name: str) -> tuple[int | None, str]:
    match = re.match(r"^(\d+)[_ -](.+)$", parent_name)
    if match:
        return int(match.group(1)), match.group(2).strip()
    return None, parent_name


def _resolve_labels(file_names: Sequence[str]) -> tuple[np.ndarray, list[str], list[dict[str, Any]]]:
    parsed: list[tuple[int | None, str, str]] = []
    for file_name in file_names:
        parent = Path(file_name).parent.name
        class_number, class_name = _parse_class_token(parent)
        parsed.append((class_number, class_name, parent))

    unique_tokens = {(num, cls) for num, cls, _ in parsed}
    if all(num is not None for num, _ in unique_tokens):
        ordered = sorted(unique_tokens, key=lambda item: (item[0] or 0, item[1].lower()))
    else:
        ordered = sorted(unique_tokens, key=lambda item: item[1].lower())

    class_names = [cls for _, cls in ordered]
    class_map = {token: idx for idx, token in enumerate(ordered)}
    true_labels = np.array([class_map[(num, cls)] for num, cls, _ in parsed], dtype=np.int64)

    index_rows: list[dict[str, Any]] = []
    for class_idx, (num, cls) in enumerate(ordered):
        folder_name = next(parent for n, c, parent in parsed if n == num and c == cls)
        index_rows.append(
            {
                "class_id": class_idx,
                "class_name": cls,
                "folder_name": folder_name,
                "class_number": "" if num is None else num,
            }
        )
    return true_labels, class_names, index_rows


def _infer_output_dir(features_dir: Path, model_name: str) -> Path:
    # Typical shape: <RUN_ROOT>/features/<model>
    if features_dir.parent.name == "features":
        run_root = features_dir.parent.parent
        return run_root / "clustering" / model_name
    return features_dir / "clustering"


def _find_layer_file(features_dir: Path, model_name: str, layer_name: str) -> Path:
    preferred = features_dir / f"{model_name}_{_sanitize_name(layer_name)}.npy"
    if preferred.exists():
        return preferred

    candidates = list(features_dir.glob(f"*_{_sanitize_name(layer_name)}.npy"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates[:5])
        raise SystemExit(f"Ambiguous feature file for layer '{layer_name}': {names}")
    raise SystemExit(f"Missing feature file for layer '{layer_name}' under {features_dir}")


def _save_class_index(rows: Sequence[dict[str, Any]], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_id", "class_name", "folder_name", "class_number"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _save_sample_index(file_names: Sequence[str], true_labels: np.ndarray, class_names: Sequence[str], out_csv: Path) -> None:
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "file_name", "true_class_id", "true_class_name"])
        for idx, file_name in enumerate(file_names):
            class_id = int(true_labels[idx])
            writer.writerow([idx, file_name, class_id, class_names[class_id]])


def _aligned_labels_and_accuracy(true_labels: np.ndarray, pred_labels: np.ndarray, n_classes: int) -> tuple[np.ndarray, float]:
    true_unique = np.unique(true_labels)
    pred_unique = np.unique(pred_labels)
    cm = confusion_matrix(true_labels, pred_labels, labels=np.arange(max(n_classes, pred_unique.max() + 1)))
    cm = cm[: n_classes, :]
    cost = cm.max() - cm
    row_ind, col_ind = linear_sum_assignment(cost)

    mapping: dict[int, int] = {}
    for true_idx, pred_idx in zip(row_ind, col_ind):
        mapping[pred_idx] = true_idx

    aligned = np.array([mapping.get(int(pred), -1) for pred in pred_labels], dtype=np.int64)
    valid = aligned >= 0
    if np.any(valid):
        accuracy = float(np.mean(aligned[valid] == true_labels[valid]))
    else:
        accuracy = float("nan")
    return aligned, accuracy


def _plot_umap_comparison(
    embedding: np.ndarray,
    true_labels: np.ndarray,
    aligned_labels: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
    title_prefix: str,
    cluster_space: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6), sharex=True, sharey=True)
    cmap = plt.get_cmap("tab10")
    colors = [cmap(idx % 10) for idx in range(len(class_names))]
    for class_idx, class_name in enumerate(class_names):
        mask_true = true_labels == class_idx
        axes[0].scatter(
            embedding[mask_true, 0],
            embedding[mask_true, 1],
            s=42,
            alpha=0.88,
            color=colors[class_idx],
            label=class_name,
            edgecolors="none",
        )
    axes[0].set_title(f"{title_prefix} - Ground Truth")
    axes[0].set_xlabel("UMAP-1")
    axes[0].set_ylabel("UMAP-2")
    axes[0].legend(loc="best", fontsize=8)

    for class_idx, class_name in enumerate(class_names):
        mask_pred = aligned_labels == class_idx
        axes[1].scatter(
            embedding[mask_pred, 0],
            embedding[mask_pred, 1],
            s=42,
            alpha=0.88,
            color=colors[class_idx],
            label=class_name,
            edgecolors="none",
        )
    if np.any(aligned_labels < 0):
        unknown = aligned_labels < 0
        axes[1].scatter(
            embedding[unknown, 0],
            embedding[unknown, 1],
            s=46,
            alpha=0.9,
            color="black",
            marker="x",
            label="Unmapped",
        )
    axes[1].set_title(f"{title_prefix} - KMeans on {cluster_space} (Aligned)")
    axes[1].set_xlabel("UMAP-1")
    axes[1].legend(loc="best", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def _plot_confusion(
    confusion: np.ndarray,
    class_names: Sequence[str],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    im = ax.imshow(confusion, cmap="Blues", interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            val = int(confusion[i, j])
            ax.text(j, i, str(val), ha="center", va="center", color="black", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def run_clustering(config: ClusterConfig) -> None:
    features_dir = config.features_dir.resolve()
    if not features_dir.exists():
        raise SystemExit(f"Features directory not found: {features_dir}")

    model_name = config.model_name or features_dir.name
    output_dir = (config.output_dir.resolve() if config.output_dir else _infer_output_dir(features_dir, model_name))
    output_dir.mkdir(parents=True, exist_ok=True)

    module_names = _read_lines(features_dir / "module_names.txt")
    file_names = _read_lines(features_dir / "file_names.txt")
    selected_layers = config.layers or module_names
    if not selected_layers:
        raise SystemExit("No layers resolved for clustering.")

    true_labels, class_names, class_index_rows = _resolve_labels(file_names)
    n_classes = len(class_names)
    k = config.k if config.k is not None else n_classes
    if k < 2:
        raise SystemExit(f"k must be >= 2, got {k}")
    if len(file_names) < 3:
        raise SystemExit("Need at least 3 samples for UMAP + clustering.")

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    _save_class_index(class_index_rows, labels_dir / "class_index.csv")
    _save_sample_index(file_names, true_labels, class_names, labels_dir / "sample_index.csv")

    n_neighbors = min(config.n_neighbors, max(2, len(file_names) - 1))
    if n_neighbors != config.n_neighbors:
        _log(f"Adjusted n_neighbors from {config.n_neighbors} to {n_neighbors} for sample count {len(file_names)}")

    layer_rows: list[dict[str, Any]] = []
    _log(
        f"Starting UMAP+KMeans clustering for {len(selected_layers)} layers, "
        f"k={k}, classes={n_classes}, cluster_space={config.cluster_space}"
    )

    for layer_idx, layer_name in enumerate(selected_layers, start=1):
        _log(f"Layer {layer_idx}/{len(selected_layers)}: {layer_name}")
        layer_file = _find_layer_file(features_dir, model_name, layer_name)
        features = np.load(layer_file)
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)
        if features.ndim != 2:
            raise SystemExit(f"Expected 2D features after reshape for {layer_file}, got {features.shape}")
        if features.shape[0] != len(file_names):
            raise SystemExit(
                f"Feature sample count mismatch for {layer_file.name}: {features.shape[0]} != {len(file_names)}"
            )

        features_scaled = StandardScaler().fit_transform(features)
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=config.min_dist,
            metric=config.metric,
            random_state=config.random_state,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="n_jobs value .* overridden to 1 by setting random_state.*",
                category=UserWarning,
            )
            embedding = reducer.fit_transform(features_scaled)
        if config.cluster_space == "umap":
            cluster_input = embedding
        else:
            cluster_input = features_scaled

        kmeans = KMeans(n_clusters=k, n_init=20, random_state=config.random_state)
        pred_labels = kmeans.fit_predict(cluster_input).astype(np.int64)

        aligned_labels, cluster_acc = _aligned_labels_and_accuracy(true_labels, pred_labels, n_classes=n_classes)
        ari = float(adjusted_rand_score(true_labels, pred_labels))
        nmi = float(normalized_mutual_info_score(true_labels, pred_labels))
        if len(np.unique(pred_labels)) > 1:
            sil_cluster = float(silhouette_score(cluster_input, pred_labels))
            sil_feature = float(silhouette_score(features_scaled, pred_labels))
            sil_umap = float(silhouette_score(embedding, pred_labels))
        else:
            sil_cluster = float("nan")
            sil_feature = float("nan")
            sil_umap = float("nan")

        layer_safe = _sanitize_name(layer_name)
        layer_root = output_dir / layer_safe
        layer_data = layer_root / "data"
        layer_plots = layer_root / "plots"
        layer_data.mkdir(parents=True, exist_ok=True)
        layer_plots.mkdir(parents=True, exist_ok=True)

        np.save(layer_data / "umap_embedding.npy", embedding)
        np.save(layer_data / "kmeans_labels.npy", pred_labels)
        np.save(layer_data / "kmeans_labels_aligned.npy", aligned_labels)
        np.save(layer_data / "true_labels.npy", true_labels)

        valid_mask = aligned_labels >= 0
        if np.any(valid_mask):
            cm_aligned = confusion_matrix(
                true_labels[valid_mask],
                aligned_labels[valid_mask],
                labels=np.arange(n_classes),
            )
        else:
            cm_aligned = np.zeros((n_classes, n_classes), dtype=np.int64)
        np.savetxt(layer_data / "confusion_matrix_aligned.csv", cm_aligned, delimiter=",", fmt="%d")
        _plot_confusion(
            confusion=cm_aligned,
            class_names=class_names,
            out_path=layer_plots / "confusion_matrix_aligned.png",
            title=f"{layer_name}: aligned confusion",
        )
        _plot_umap_comparison(
            embedding=embedding,
            true_labels=true_labels,
            aligned_labels=aligned_labels,
            class_names=class_names,
            out_path=layer_plots / "umap_ground_truth_vs_cluster.png",
            title_prefix=layer_name,
            cluster_space=config.cluster_space,
        )

        metrics = {
            "layer_name": layer_name,
            "layer_file": str(layer_file.resolve()),
            "n_samples": int(features.shape[0]),
            "n_features": int(features.shape[1]),
            "n_classes": int(n_classes),
            "kmeans_k": int(k),
            "cluster_space": config.cluster_space,
            "ari": ari,
            "nmi": nmi,
            "cluster_accuracy_aligned": cluster_acc,
            "silhouette_cluster_space": sil_cluster,
            "silhouette_feature_space": sil_feature,
            "silhouette_umap_space": sil_umap,
        }
        (layer_data / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        layer_rows.append(metrics)

    summary_csv = output_dir / "layer_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "layer_name",
                "ari",
                "nmi",
                "cluster_accuracy_aligned",
                "cluster_space",
                "silhouette_cluster_space",
                "silhouette_feature_space",
                "silhouette_umap_space",
                "n_samples",
                "n_features",
                "n_classes",
                "kmeans_k",
            ]
        )
        for row in layer_rows:
            writer.writerow(
                [
                    row["layer_name"],
                    row["ari"],
                    row["nmi"],
                    row["cluster_accuracy_aligned"],
                    row["cluster_space"],
                    row["silhouette_cluster_space"],
                    row["silhouette_feature_space"],
                    row["silhouette_umap_space"],
                    row["n_samples"],
                    row["n_features"],
                    row["n_classes"],
                    row["kmeans_k"],
                ]
            )

    run_manifest = {
        "features_dir": str(features_dir),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "layers": selected_layers,
        "n_neighbors_requested": config.n_neighbors,
        "n_neighbors_used": n_neighbors,
        "min_dist": config.min_dist,
        "metric": config.metric,
        "random_state": config.random_state,
        "kmeans_k": k,
        "cluster_space": config.cluster_space,
        "n_classes_ground_truth": n_classes,
        "class_names": class_names,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log(f"Done. Clustering outputs written to: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cluster extracted ANN features with UMAP + KMeans and compare against ground truth classes.",
    )
    parser.add_argument("--features-dir", type=Path, required=True, help="Feature directory containing file_names.txt, module_names.txt and layer .npy files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory. Default: <RUN_ROOT>/clustering/<model> when features_dir is <RUN_ROOT>/features/<model>.")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer names. Default: all layers from module_names.txt.")
    parser.add_argument("--n-neighbors", type=int, default=10, help="UMAP n_neighbors.")
    parser.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist.")
    parser.add_argument("--metric", type=str, default="euclidean", help="UMAP distance metric.")
    parser.add_argument("--k", type=int, default=None, help="KMeans cluster count. Default: number of ground-truth classes.")
    parser.add_argument(
        "--cluster-space",
        type=str,
        default="umap",
        choices=["umap", "feature"],
        help="Where KMeans runs: 'umap' (default) or 'feature' (cluster before UMAP).",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for UMAP and KMeans.")
    parser.add_argument("--model-name", type=str, default=None, help="Optional model name override (default: features-dir folder name).")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    layers = None
    if args.layers:
        layers = [item.strip() for item in args.layers.split(",") if item.strip()]

    config = ClusterConfig(
        features_dir=args.features_dir,
        output_dir=args.output_dir,
        layers=layers,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.random_state,
        k=args.k,
        model_name=args.model_name,
        cluster_space=args.cluster_space,
    )
    run_clustering(config)
    if config.cluster_space not in {"umap", "feature"}:
        raise SystemExit(f"Unsupported cluster_space '{config.cluster_space}'. Use 'umap' or 'feature'.")
