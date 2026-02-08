#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcb11.pipeline_clean import canonical_cli_main


def main() -> None:
    print(
        "DEPRECATED: scripts/compute_rdms.py now forwards to scripts/run_meg_ann_rsa.py "
        "and accepts canonical arguments only: --model, --meg-rdm, --output-root."
    )
    canonical_cli_main(
        description=(
            "DEPRECATED wrapper for scripts/compute_rdms.py. "
            "Use scripts/run_meg_ann_rsa.py with canonical arguments."
        )
    )


if __name__ == "__main__":
    main()
