#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcb11.pipeline import canonical_cli_main


def main() -> None:
    canonical_cli_main(
        description=(
            "Canonical pipeline: ANN feature extraction -> ANN/MEG RDM visualizations -> RSA. "
            "Layer presets are hardcoded per model."
        )
    )


if __name__ == "__main__":
    main()
