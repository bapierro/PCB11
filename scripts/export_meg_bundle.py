#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcb11.meg_bundle import build_meg_bundle


def build_parser() -> argparse.ArgumentParser:
    """Create CLI parser for one-time MEG bundle export."""

    parser = argparse.ArgumentParser(
        description=(
            "Export a unified MEG bundle (.npz) containing signals, time vector, "
            "stimulus order, image IDs, and category metadata."
        )
    )
    parser.add_argument("--meg-rdm", type=Path, default=Path("data/meg/MEGRDMs_2D.mat"))
    parser.add_argument("--time-mat", type=Path, default=Path("data/meg/time.mat"))
    parser.add_argument("--image-struct", type=Path, default=Path("data/meg/imagestruct_final.mat"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/scenes/syns_meg36"))
    parser.add_argument("--output", type=Path, default=Path("data/meg/meg_data.npz"))
    return parser


def main() -> None:
    """Run MEG bundle export and print summary."""

    args = build_parser().parse_args()
    summary = build_meg_bundle(
        meg_rdm_path=args.meg_rdm,
        time_mat_path=args.time_mat,
        image_struct_path=args.image_struct,
        images_dir=args.images_dir,
        output_path=args.output,
    )
    print("Exported MEG bundle:")
    print(f"  path: {summary['output_path']}")
    print(f"  images: {summary['n_images']}")
    print(f"  time points: {summary['n_time']}")
    print(f"  subjects: {summary['n_subjects']}")


if __name__ == "__main__":
    main()
