#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'USAGE'
Usage:
  scripts/extract_resnet50_blockrelu.sh [IMAGES_DIR] [OUTPUT_DIR] [DEVICE] [BATCH_SIZE]

Defaults:
  IMAGES_DIR  = data/scenes/syns_meg36
  OUTPUT_DIR  = features/resnet50_blockrelu20
  DEVICE      = mps
  BATCH_SIZE  = 16

Example:
  scripts/extract_resnet50_blockrelu.sh data/scenes/syns_meg36 features/resnet50_blockrelu20 mps 16
USAGE
  exit 0
fi

IMAGES_DIR="${1:-data/scenes/syns_meg36}"
OUTPUT_DIR="${2:-features/resnet50_blockrelu20}"
DEVICE="${3:-mps}"
BATCH_SIZE="${4:-16}"

# Curated ResNet50 progression: stem -> block outputs -> head.
LAYERS="conv1,maxpool,\
layer1.0.relu,layer1.1.relu,layer1.2.relu,\
layer2.0.relu,layer2.1.relu,layer2.2.relu,layer2.3.relu,\
layer3.0.relu,layer3.1.relu,layer3.2.relu,layer3.3.relu,layer3.4.relu,layer3.5.relu,\
layer4.0.relu,layer4.1.relu,layer4.2.relu,\
avgpool,fc"

poetry run python -m pcb11.features_cli "${IMAGES_DIR}" \
  --model resnet50 \
  --layers "${LAYERS}" \
  --pool gap \
  --device "${DEVICE}" \
  --batch-size "${BATCH_SIZE}" \
  --output "${OUTPUT_DIR}"
