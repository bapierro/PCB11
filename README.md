# PCB11

Topic #11: Can behaviour-trained ANNs reveal the brain's temporal hierarchy in scene processing?

## Overview

This project extracts CNN features at multiple layers and compares them with brain data (MEG/fMRI) using Representational Similarity Analysis (RSA).

See [docs/feature_extraction_explained.md](docs/feature_extraction_explained.md) for a non-technical overview.

## Installation

### Requirements
- Python 3.10 or 3.11
- [Poetry](https://python-poetry.org/docs/#installation)

### Setup
```bash
# Clone and enter the repository
git clone <repo-url>
cd PCB11

# Create environment and install dependencies
poetry env use python3.10
poetry install
```

### Alternative (without Poetry)
```bash
# Create a virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run commands with PYTHONPATH set to src/
PYTHONPATH=src python -m pcb11.features_cli data/scenes/syns_meg36
```

## Feature Extraction

Extract CNN features with Global Average Pooling (GAP) for RSA analysis.

### Basic Usage
```bash
# Extract ResNet50 features (default)
poetry run python -m pcb11.features_cli data/scenes/syns_meg36

# Use AlexNet instead
poetry run python -m pcb11.features_cli data/scenes/syns_meg36 --model alexnet

# Specify output directory
poetry run python -m pcb11.features_cli data/scenes/ -o features/my_experiment
```

### Options
| Flag | Description | Default |
|------|-------------|---------|
| `-m, --model` | Model name (resnet50, alexnet, vgg16, etc.) | resnet50 |
| `-l, --layers` | Comma-separated layers, or `resnet`/`alexnet` presets | auto |
| `-p, --pool` | Pooling mode: `gap`, `flatten`, `none` | gap |
| `-o, --output` | Output directory for .npy files | features/ |
| `-d, --device` | Device: `cpu`, `cuda`, `mps` | auto-detect |

### Output
Each layer produces a `.npy` file with shape `(N, C)` where:
- `N` = number of images
- `C` = number of channels (256, 512, 1024, 2048 for ResNet50)

## Project Structure
```
PCB11/
├── src/pcb11/
│   ├── features.py       # Feature extraction with GAP pooling
│   ├── features_cli.py   # CLI for feature extraction
│   └── data_utils.py     # Image directory utilities
├── scripts/
│   ├── rsa_analysis.py   # RSA analysis
│   └── train_behaviour_model.py
├── data/
│   └── scenes/           # Input images
├── features/             # Extracted .npy files (gitignored)
└── docs/
    └── feature_extraction_explained.md
```

## Regenerating requirements.txt
```bash
poetry export -f requirements.txt --output requirements.txt --without-hashes
```

## Platform Notes
- **Apple Silicon**: Uses `tensorflow-macos` automatically
- **Linux/Windows**: Standard TensorFlow via Poetry
