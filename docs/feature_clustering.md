# Feature Clustering (UMAP + KMeans)

This document explains how to run clustering on extracted ANN features, how to choose between clustering modes, and how to interpret the outputs.

## Goal

Given already extracted layer features (from `features/<model>/`), this workflow:

1. Builds 2D UMAP embeddings for visualization.
2. Runs KMeans clustering with `k` equal to number of ground-truth classes by default.
3. Compares predicted clusters against ground-truth labels inferred from image folders.

## Input Requirements

Required folder (`--features-dir`) must contain:

- `file_names.txt`
- `module_names.txt`
- layer feature files like `<model>_<layer>.npy`

Ground-truth labels are inferred from `file_names.txt` by parsing parent folders.
For SYNS-36 this maps to:

- `1_Residence`
- `2_Nature`
- `3_Farm`
- `4_CarPark`

## CLI Usage

Default mode (cluster on UMAP embedding):

```bash
poetry run python scripts/cluster_features.py \
  --features-dir outputs/clean_refactor_resnet101/features/resnet101
```

Cluster-first mode (cluster on original standardized feature vectors):

```bash
poetry run python scripts/cluster_features.py \
  --features-dir outputs/clean_refactor_resnet101/features/resnet101 \
  --cluster-space feature
```

Run only selected layers:

```bash
poetry run python scripts/cluster_features.py \
  --features-dir outputs/clean_refactor_resnet101/features/resnet101 \
  --layers layer1,layer2
```

Custom output folder and custom `k`:

```bash
poetry run python scripts/cluster_features.py \
  --features-dir outputs/clean_refactor_resnet101/features/resnet101 \
  --output-dir outputs/cluster_runs/resnet101_k4 \
  --k 4
```

## Key Arguments

- `--features-dir`: required feature directory.
- `--cluster-space {umap,feature}`:
  - `umap` (default): KMeans runs on 2D UMAP.
  - `feature`: KMeans runs on standardized original features; UMAP is used only for visualization.
- `--layers`: comma-separated layer names; default is all layers from `module_names.txt`.
- `--k`: KMeans cluster count; default is inferred number of ground-truth classes.
- `--n-neighbors`, `--min-dist`, `--metric`: UMAP parameters.
- `--random-state`: seed for reproducibility.
- `--output-dir`: optional explicit output location.

## Output Structure

If `--features-dir` is `<RUN_ROOT>/features/<model>` and `--output-dir` is omitted, outputs go to:

`<RUN_ROOT>/clustering/<model>/`

Main files:

- `run_manifest.json`: resolved parameters (including `cluster_space`).
- `layer_summary.csv`: one row per layer with comparison metrics.
- `labels/class_index.csv`: class id/name mapping.
- `labels/sample_index.csv`: sample index to file and true class.

Per layer (`<layer>/`):

- `data/umap_embedding.npy`
- `data/kmeans_labels.npy`
- `data/kmeans_labels_aligned.npy`
- `data/true_labels.npy`
- `data/confusion_matrix_aligned.csv`
- `data/metrics.json`
- `plots/umap_ground_truth_vs_cluster.png`
- `plots/confusion_matrix_aligned.png`

## Metrics in `layer_summary.csv`

- `ari`: Adjusted Rand Index (cluster-label permutation invariant).
- `nmi`: Normalized Mutual Information.
- `cluster_accuracy_aligned`: cluster accuracy after optimal label alignment (Hungarian matching).
- `cluster_space`: which space KMeans used (`umap` or `feature`).
- `silhouette_cluster_space`: silhouette in the actual KMeans input space.
- `silhouette_feature_space`: silhouette computed in standardized feature space.
- `silhouette_umap_space`: silhouette computed in UMAP 2D space.

## Practical Interpretation

- Compare `umap` vs `feature` mode per layer using `ari`, `nmi`, and `cluster_accuracy_aligned`.
- Do not over-interpret silhouette across different spaces; it is most meaningful within the same `cluster_space`.
- Use the UMAP plot as visual intuition, but rely on numeric metrics for mode/layer comparison.

## Caveats

- Validity depends on correct image ordering and correct folder-based ground-truth labels.
- Small sample size (36 images) can make clustering unstable; keep `random_state` fixed for comparability.
- UMAP embeddings are dimensionality-reduction projections, not full-representation geometry.
