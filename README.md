# PCB11

Topic #11: Can behaviour-trained ANNs reveal the brain's temporal hierarchy in scene processing?

## Overview

The repository now uses one canonical pipeline command that runs end-to-end:
1. ANN feature extraction (fixed image set, fixed layer preset by model)
2. ANN RDM computation and visualization
3. MEG RDM export and visualization
4. RSA between ANN layer RDMs and MEG RDMs

The canonical script is:

```bash
poetry run python scripts/run_meg_ann_rsa.py \
  --model resnet50 \
  --meg-rdm data/meg/MEGRDMs_2D.mat \
  --output-root outputs/clean_run_resnet50
```

## Requirements

- Python 3.10 or 3.11
- [Poetry](https://python-poetry.org/docs/#installation)

Setup:

```bash
poetry env use python3.10
poetry install
```

## Canonical CLI

`scripts/run_meg_ann_rsa.py` accepts exactly:
- `--model` (`resnet50`, `resnet18`, `resnet101`, `alexnet`)
- `--meg-rdm` (path to `.mat` with variable `MEGRDMs_2D`)
- `--output-root` (run output folder)

No layer argument is exposed in the canonical CLI. Layer presets are hardcoded:
- `resnet50`: `layer1,layer2,layer3,layer4`
- `resnet18`: `layer1,layer2,layer3,layer4`
- `resnet101`: `layer1,layer2,layer3,layer4`
- `alexnet`: `features.2,features.5,features.9,classifier.2,classifier.5`

## Feature Clustering CLI (UMAP + KMeans)

You can cluster already extracted features with UMAP + KMeans and compare clusters to dataset ground truth labels:

```bash
poetry run python scripts/cluster_features.py \
  --features-dir outputs/clean_refactor_resnet101/features/resnet101
```

Cluster first in original feature space (then use UMAP only for visualization):

```bash
poetry run python scripts/cluster_features.py \
  --features-dir outputs/clean_refactor_resnet101/features/resnet101 \
  --cluster-space feature
```

Ground truth labels are inferred from image folder names in `file_names.txt` (for SYNS-36: `1_Residence`, `2_Nature`, `3_Farm`, `4_CarPark`).

Defaults:
- UMAP `n_neighbors=10`, `min_dist=0.1`, `metric=euclidean`
- KMeans `k = number of ground-truth classes`
- KMeans clustering space: `umap` (use `--cluster-space feature` for cluster-first)
- all layers from `module_names.txt`

Main outputs (under `<RUN_ROOT>/clustering/<model>/` by default):
- `run_manifest.json`
- `layer_summary.csv` (ARI, NMI, aligned cluster accuracy, silhouette)
- `labels/class_index.csv`, `labels/sample_index.csv`
- per-layer folders with:
  - `data/umap_embedding.npy`, `data/kmeans_labels.npy`, `data/kmeans_labels_aligned.npy`
  - `data/metrics.json`, `data/confusion_matrix_aligned.csv`
  - `plots/umap_ground_truth_vs_cluster.png`, `plots/confusion_matrix_aligned.png`

Full clustering documentation (arguments, outputs, and interpretation):
- `docs/feature_clustering.md`

Fixed defaults in the pipeline:
- image directory: `data/scenes/syns_meg36`
- feature extraction source: `torchvision`
- pooling: `gap`
- weights: pretrained
- RDM method: correlation (`thingsvision.core.rsa.compute_rdm`)
- RSA strategy: subject-wise first, then group-mean summary

## Output Structure

For `--output-root <RUN_ROOT>`:

```text
<RUN_ROOT>/
  features/
    <model>/
      file_names.txt
      module_names.txt
      <model>_<layer>.npy
  RDM/
    MEG/
      data/
        meg_rdm_avg.npy
        time_ms.npy
        stimulus_order_used.csv
        per_subject/
          subject_index.csv
          subject_01/
            meg_rdm.npy
            time_ms.npy
            stimulus_order_used.csv
      plots/
        meg_rdm_snapshots.png
        meg_rdm_summary_over_time.png
        meg_rdm_index_grid.png
        per_subject/
          subject_01/
            meg_rdm_snapshots.png
            meg_rdm_summary_over_time.png
      meg_rdm_animation.gif
      per_subject/
        subject_01/
          meg_rdm_animation.gif
      meg_rdm_index_map.csv
    <model>/
      data/
        <model>_<layer>_rdm.npy
        rdm_order.txt
      plots/
        shared_scale/
          <model>_<layer>_rdm.png
        per_layer_scale/
          <model>_<layer>_rdm.png
      rdm_animation.gif
  RSA/
    data/
      layer_index.csv
      layer_summary.csv
      rsa_layer_time_subjects.npy
      rsa_layer_time_group_mean.npy
      rsa_layer_time_group_sem.npy
      <layer>_spearman.csv
      <layer>_spearman.npy
      per_subject/
        subject_index.csv
        subject_01/
          layer_index.csv
          layer_summary.csv
          <layer>_spearman.csv
          <layer>_spearman.npy
    plots/
      rsa_layerwise_heatmap.png
      rsa_layerwise_overlay.png
      per_subject/
        subject_01/
          rsa_layerwise_heatmap.png
          rsa_layerwise_overlay.png
  run_manifest.json
```

## Stimulus Order Policy

The pipeline resolves stimulus ordering with this priority:
1. `data/meg/stimulus_order.csv` (if present)
2. fallback to sorted image file order from `data/scenes/syns_meg36`

If fallback is used, the run prints a warning and stores it in `run_manifest.json`.
The exact order used is always written to:
- `<RUN_ROOT>/RDM/MEG/data/stimulus_order_used.csv`

Critical validity note:
- ANN<->MEG RSA is only interpretable if ANN and MEG RDM indices refer to the same stimuli in the same order.
- If `stimulus_order_source` in `run_manifest.json` is `fallback_sorted_images`, treat RSA results as exploratory and not as confirmatory evidence.

## Legacy Script Wrappers

These scripts are kept as deprecated wrappers and now forward to the canonical pipeline with canonical args:
- `scripts/rsa.py`
- `scripts/compute_rdms.py`
- `scripts/plot_meg_rdms.py`

Use `scripts/run_meg_ann_rsa.py` for all new runs.

## Notes

For a non-technical intuition of feature extraction, see:
- `docs/feature_extraction_explained.md`

For a baseline snapshot of the current pipeline behavior, graph interpretation, and known limitations/open questions, see:
- `docs/current_pipeline_baseline.md`

For detailed clustering usage and interpretation, see:
- `docs/feature_clustering.md`
