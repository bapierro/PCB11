# Current Pipeline Baseline

This document captures what the current clean pipeline does **today**, what each generated graph means, and which drawbacks/open questions remain before further changes.

## Critical validity condition for ANN<->MEG RSA

ANN<->MEG RSA is only scientifically interpretable when ANN and MEG RDM rows/columns refer to the **same stimulus order**.

- If the mapping from MEG index to image is unknown or wrong, RSA values can be arbitrary/misleading.
- In that case, ANN-only and MEG-only RDM visualizations are still valid descriptively, but cross-modal RSA is not.
- Therefore, `data/meg/stimulus_order.csv` (or equivalent verified order) is a required artifact for trustworthy RSA conclusions.
- If fallback sorted-image order is used, treat RSA outputs as exploratory only.

## Canonical entrypoint

Run:

```bash
poetry run python scripts/run_meg_ann_rsa.py \
  --model <resnet18|resnet50|resnet101|alexnet> \
  --meg-rdm data/meg/MEGRDMs_2D.mat \
  --output-root outputs/<run_name>
```

Code path:
- `scripts/run_meg_ann_rsa.py`
- `src/pcb11/pipeline_clean.py`

## What the pipeline currently does

### 1) Inputs and fixed defaults

- Fixed image set: `data/scenes/syns_meg36`
- MEG input variable expected in `.mat`: `MEGRDMs_2D`
- Time vector source: `data/meg/time.mat` variable `time` (fallback: `np.arange(T)-200`)
- Feature source: `thingsvision` extractor with `torchvision` backbones
- Pooling: `gap`
- RDM method for ANN: `correlation` via `thingsvision.core.rsa.compute_rdm`
- Layer presets (hardcoded):
  - `resnet18|resnet50|resnet101`: `layer1, layer2, layer3, layer4`
  - `alexnet`: `features.2, features.5, features.9, classifier.2, classifier.5`

### 2) Stimulus order resolution and alignment

- Preferred order file: `data/meg/stimulus_order.csv`
- Supported order formats:
  - `file_name`
  - `SYNSscene,SYNSView`
- If missing, pipeline falls back to sorted image-file discovery under `data/scenes/syns_meg36` and records a warning.
- ANN features are reordered to the resolved stimulus order using `file_names.txt`.
- Pipeline hard-fails if MEG image dimension and resolved order length differ.

### 3) Stage-by-stage execution

- Stage 1: Extract features or reuse existing complete features for selected model preset.
- Stage 2: Load MEG RDM tensor, keep subject-resolved RDMs when available, export group-average and per-subject MEG RDM files/plots.
- Stage 3: Compute ANN layer RDMs, save matrices, render ANN RDM plots in two scale modes, and render ANN RDM animation.
- Stage 4: Compute layer-by-time RSA per subject (Spearman on vectorized upper triangles), then write group-mean summaries and per-subject outputs.
- Stage 5: Write `run_manifest.json` with resolved configuration, warnings, and paths.

## What the graphs mean

## MEG plots

- `RDM/MEG/plots/meg_rdm_index_grid.png`
  - Visual lookup table of stimulus index -> actual image thumbnail.
  - Used to interpret rows/columns in all MEG/ANN RDMs.
- `RDM/MEG/plots/meg_rdm_snapshots.png`
  - Multiple time-slice RDMs (currently around -200, 0, 100, 200, 400, 800 ms).
  - Uses one shared color range across selected snapshots.
- `RDM/MEG/plots/meg_rdm_summary_over_time.png`
  - Time course of mean and std of pairwise dissimilarities (upper-triangle entries).
  - Descriptive signal summary, not an RSA significance test.
- `RDM/MEG/meg_rdm_animation.gif`
  - Temporal evolution of MEG RDMs over time points.
- `RDM/MEG/plots/per_subject/<subject>/...` and `RDM/MEG/per_subject/<subject>/meg_rdm_animation.gif`
  - Same MEG RDM visualizations for each subject separately.

## ANN RDM plots

- `RDM/<model>/plots/shared_scale/*.png`
  - Layer RDMs with one common color range across layers.
  - Good for absolute across-layer comparison.
- `RDM/<model>/plots/per_layer_scale/*.png`
  - Layer RDMs with each layer auto-scaled independently.
  - Good for seeing weak early-layer structure.
- `RDM/<model>/rdm_animation.gif`
  - Layer progression animation using shared scale across frames.

## RSA plots

- `RSA/plots/rsa_layerwise_heatmap.png`
  - Matrix of layer (rows) vs time (columns), values are Spearman correlations between ANN RDM and MEG RDM.
  - Vertical dashed line marks image onset (`t=0`).
- `RSA/plots/rsa_layerwise_overlay.png`
  - Time courses for all layers overlaid.
  - Last layer highlighted in black.
- `RSA/plots/per_subject/<subject>/...`
  - Subject-specific RSA heatmap/overlay for each participant.

## What works already

- Single canonical CLI for end-to-end run.
- Structured output tree (`features/`, `RDM/`, `RSA/`, `run_manifest.json`).
- Deterministic layer presets per model.
- Feature reuse logic to skip unnecessary re-extraction.
- Explicit stage logging and warning propagation to manifest.
- Two ANN RDM scale views (`shared_scale`, `per_layer_scale`).
- Subject-specific MEG RDM outputs and subject-specific RSA outputs with structured folders.

## Current drawbacks and open questions

1. Stimulus order is a hard validity condition for RSA, and confidence is currently limited without a verified `data/meg/stimulus_order.csv`.
2. Current runs in this repo used fallback sorted-image order (as recorded in run manifests), so alignment should be treated as provisional until validated.
3. MEG input is already precomputed RDMs; this pipeline does not recompute MEG RDMs from raw sensor trials, so dissimilarity provenance depends on upstream preprocessing.
4. Subject-level RSA is now exported, but inferential statistics are still missing (e.g., permutation tests, cluster correction, confidence intervals).
5. Group-level outputs currently summarize with means/SEMs; they are still descriptive unless explicit hypothesis testing is added.
6. Layer coverage is intentionally sparse (stage-level presets), which is efficient but may miss informative intermediate computations.
7. Group boundary lines in RDM plots assume contiguous category blocks from resolved order; if order is interleaved, boundaries can be visually misleading.
8. Current pipeline does not include baseline model controls (untrained/random/task controls) or noise-ceiling estimates.
9. `data/meg/MEGRDMs_2D_avg.mat` currently does not expose `MEGRDMs_2D` under that exact variable name, so canonical loading fails for that file without conversion.

## Suggested baseline checks before next changes

1. Add and verify authoritative `data/meg/stimulus_order.csv` against the MEG generation pipeline.
2. Decide whether next iteration should keep subject-level MEG RDMs through RSA and statistics.
3. Decide layer sampling policy per architecture (stage checkpoints vs denser taps).
4. Define required inferential outputs (permutation/cluster tests, latency tests, uncertainty bands).
5. Record MEG RDM provenance (exact dissimilarity metric and preprocessing) in docs.
