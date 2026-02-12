# PCB11

Topic #11: Can behaviour-trained ANNs reveal the brain's temporal hierarchy in scene processing?

## Overview

The repository now uses one main pipeline command that runs end-to-end:
1. ANN feature extraction (fixed image set, fixed layer preset by model)
2. ANN RDM computation and visualization
3. MEG RDM export and visualization
4. RSA between ANN layer RDMs and MEG RDMs

## MEG Data Download

Published MEG data are available on OSF:
- `https://osf.io/jp26k/overview`

Download these files and place them under `data/meg/`:
- from `MEG RDMs`: `MEGRDMs_2D.mat` -> `data/meg/MEGRDMs_2D.mat`
- from `MEG RDMs`: `time.mat` -> `data/meg/time.mat`
- from `Stimulus images (small icons for visualization)`: `imagestruct_final.mat` -> `data/meg/imagestruct_final.mat`

Notes:
- `MEGRDMs_2D.mat` has shape `36 x 36 x 1201 x 20` (images x images x time x subjects)
- `time.mat` contains the time vector (length `1201`), with baseline before `0 ms`
- `imagestruct_final.mat` provides low-resolution visualization icons plus `imF`/`imI` indices to original SYNS images
- optional reference/model RDMs (for example `semanticRDM_sq.mat`) are in OSF folder `Model RDMs`

## Requirements

- Python 3.10 or 3.11
- [Poetry](https://python-poetry.org/docs/#installation)

Setup:

```bash
poetry env use python3.10
poetry install
```

## Pipeline Steps

Run the main ANN+MEG pipeline:

```bash
poetry run python scripts/run_pipeline.py
```

The pipeline auto-handles MEG assets:
- if `data/meg/meg_data.npz` is missing, it builds it from MATLAB source files
- if `data/meg/precomputed_rdm` is missing/outdated, it rebuilds it

Example override:

```bash
poetry run python scripts/run_pipeline.py \
  --model resnet50 \
  --output-root outputs/run_resnet50
```

Optional manual precompute commands (usually not needed):

```bash
poetry run python scripts/export_meg_bundle.py
poetry run python scripts/prepare_meg_assets.py
```



## Pipeline CLI

`scripts/run_pipeline.py` accepts:
- `--model` (`resnet50`, `resnet18`, `resnet101`, `alexnet`, `vit_b_16`, `vit_b_32`; default: `resnet101`)
- `--output-root` (run output folder; default: `outputs/run_<model>`)

No layer argument is exposed in the pipeline CLI. Layer presets are hardcoded:
- `resnet50`: `layer1,layer2,layer3,layer4`
- `resnet18`: `layer1,layer2,layer3,layer4`
- `resnet101`: `layer1,layer2,layer3,layer4`
- `alexnet`: `features.2,features.5,features.7,features.9,features.12,classifier.2,classifier.5,classifier.6`
- `vit_b_16`: `encoder.layers.encoder_layer_0` ... `encoder.layers.encoder_layer_11`
- `vit_b_32`: `encoder.layers.encoder_layer_0` ... `encoder.layers.encoder_layer_11`

Fixed defaults in the pipeline:
- image directory: `data/scenes/syns_meg36`
- feature extraction source: `torchvision`
- pooling: `gap` for CNN presets, `cls` for ViT presets
- weights: pretrained
- RDM method: correlation (`thingsvision.core.rsa.compute_rdm`)
- RSA correlation: Spearman (`thingsvision.core.rsa.correlate_rdms`)

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

The pipeline auto-ensures MEG assets:
- if `data/meg/meg_data.npz` is missing, it tries to build it from MATLAB source files
- if precomputed MEG artifacts are missing/outdated, it rebuilds them under `data/meg/precomputed_rdm`

If required MATLAB source files are missing, it fails with a clear error.
The exact order used is always written to:
- `<RUN_ROOT>/RDM/MEG/data/stimulus_order_used.csv`

Critical validity note:
- ANN<->MEG RSA is only interpretable if ANN and MEG RDM indices refer to the same stimuli in the same order.
- `stimulus_order_source` in `run_manifest.json` should point to `meg_data.npz::stimulus_file_name`.

Use `scripts/run_pipeline.py` for all analysis runs.

## Notes

For a non-technical intuition of feature extraction, see:
- `docs/feature_extraction_explained.md`

For current pipeline behavior, graph interpretation, and known limitations/open questions, see:
- `docs/pipeline_status_and_limitations.md`
