# Pipeline Status and Limitations

This document captures what the current main pipeline does today, what each generated graph means, and which limitations/open questions remain.

## Critical validity condition for ANN<->MEG RSA

ANN<->MEG RSA is only scientifically interpretable when ANN and MEG RDM rows/columns refer to the **same stimulus order**.

- If the mapping from MEG index to image is unknown or wrong, RSA values can be arbitrary/misleading.
- In that case, ANN-only and MEG-only RDM visualizations are still valid descriptively, but cross-modal RSA is not.
- Therefore, `stimulus_file_name` in `data/meg/meg_data.npz` must be verified for trustworthy RSA conclusions.

## Main entrypoint

One-time MEG preparation:

```bash
poetry run python scripts/export_meg_bundle.py
poetry run python scripts/prepare_meg_assets.py
```

Run:

```bash
poetry run python scripts/run_pipeline.py
```

Optional overrides:

```bash
poetry run python scripts/run_pipeline.py \
  --model <resnet18|resnet50|resnet101|alexnet> \
  --output-root outputs/<run_name>
```

Defaults:
- `--model resnet101`
- `--output-root outputs/run_<model>`

Code path:
- `scripts/run_pipeline.py`
- `src/pcb11/pipeline.py`

## What the pipeline currently does

### 1) Inputs and fixed defaults

- Fixed image set: `data/scenes/syns_meg36`
- MEG input: `data/meg/meg_data.npz` unified bundle (auto-built if missing and MATLAB sources exist)
- Shared precomputed MEG artifacts: `data/meg/precomputed_rdm` (auto-built/reused)
- Time vector source: `time_ms` in the bundle
- Feature source: `thingsvision` extractor with `torchvision` backbones
- Pooling: `gap`
- RDM method for ANN: `correlation` via `thingsvision.core.rsa.compute_rdm`
- RSA correlation method: `spearman` via `thingsvision.core.rsa.correlate_rdms`
- Layer presets (hardcoded):
  - `resnet18|resnet50|resnet101`: `layer1, layer2, layer3, layer4`
  - `alexnet`: `features.2, features.5, features.7, features.9, features.12, classifier.2, classifier.5, classifier.6`

### 2) Stimulus order resolution and alignment

- Order source: `stimulus_file_name` from `data/meg/meg_data.npz`
- ANN features are reordered to the resolved stimulus order using `file_names.txt`.
- Pipeline hard-fails if MEG image dimension and resolved order length differ.

### 3) Stage-by-stage execution

- Stage 1: Extract features or reuse existing complete features for selected model preset.
- Stage 2: Ensure/refresh MEG assets, then copy them into the current run output.
- Stage 3: Compute ANN layer RDMs, save matrices, render ANN RDM plots in two scale modes, and render ANN RDM animation.
- Stage 4: Compute layer-by-time RSA per subject using `thingsvision.core.rsa.correlate_rdms` (`spearman`), aggregate group curves with Fisher-z mean (`atanh` -> average -> `tanh`), and write group and per-subject outputs.
- Stage 5: Write `run_manifest.json` with resolved configuration, warnings, and paths.

### Why Fisher-z for group RSA

Raw correlation values are bounded (`-1..1`) and not additive, so directly averaging subject correlations on the `r` scale can be biased.

The current group aggregation is:
- subject-level RSA first (one curve per subject),
- Fisher transform each subject value (`z = arctanh(r)`),
- average in `z` space,
- back-transform to correlation scale (`r_group = tanh(mean(z))`).

This preserves subject-level inference and yields a more stable group summary than direct arithmetic mean on raw correlations.

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
  - Time courses for all layers overlaid with equal visual treatment.
  - Legend includes all layers.
- `RSA/plots/per_subject/<subject>/...`
  - Subject-specific RSA heatmap/overlay for each participant.

## Current RSA timing observation

From the latest `resnet101` run (`outputs/run_resnet101`), there is no clear early-to-late latency ordering across layers.

- Group peak times from `RSA/data/layer_summary.csv`:
  - `layer1`: `105 ms`
  - `layer2`: `102 ms`
  - `layer3`: `102 ms`
  - `layer4`: `-97 ms` (global-maximum artifact outside post-stimulus window)
- In a restricted `0..300 ms` window, `layer4` peaks later (around `254 ms`) but with weaker magnitude than layers 1-3.
- Interpretation: this run shows a shared response window across layers rather than a clean temporal hierarchy.
- Practical takeaway: use windowed latency metrics and uncertainty estimates, not only global maxima over the full time range.

## What works already

- Single main CLI for end-to-end run.
- Structured output tree (`features/`, `RDM/`, `RSA/`, `run_manifest.json`).
- Deterministic layer presets per model.
- Feature reuse logic to skip unnecessary re-extraction.
- Explicit stage logging and warning propagation to manifest.
- Two ANN RDM scale views (`shared_scale`, `per_layer_scale`).
- Subject-specific MEG RDM outputs and subject-specific RSA outputs with structured folders.

## Current drawbacks and open questions

1. Stimulus order is a hard validity condition for RSA; verify the bundle order against the experimental protocol.
2. MEG input is already precomputed RDMs; this pipeline does not recompute MEG RDMs from raw sensor trials, so dissimilarity provenance depends on upstream preprocessing.
3. Subject-level RSA is now exported, but inferential statistics are still missing (e.g., permutation tests, cluster correction, confidence intervals).
4. Group-level outputs currently summarize with means/SEMs; they are still descriptive unless explicit hypothesis testing is added.
5. Layer coverage is intentionally sparse (stage-level presets), which is efficient but may miss informative intermediate computations.
6. Group boundary lines in RDM plots assume contiguous category blocks from resolved order; if order is interleaved, boundaries can be visually misleading.
7. Current pipeline does not include baseline model controls (untrained/random/task controls) or noise-ceiling estimates.

## Suggested baseline checks before next changes

1. Keep bundle metadata and stimulus order verified against the MEG generation pipeline.
2. Decide whether next iteration should keep subject-level MEG RDMs through RSA and statistics.
3. Decide layer sampling policy per architecture (stage checkpoints vs denser taps).
4. Define required inferential outputs (permutation/cluster tests, latency tests, uncertainty bands).
5. Record MEG RDM provenance (exact dissimilarity metric and preprocessing) in docs.
