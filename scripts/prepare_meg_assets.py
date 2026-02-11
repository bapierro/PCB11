#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pcb11.meg_assets import DEFAULT_MEG_PRECOMPUTED_ROOT, prepare_shared_meg_assets


def _log(message: str) -> None:
    print(f"[prepare_meg_assets] {message}", flush=True)


def main() -> None:
    prepared = prepare_shared_meg_assets(
        images_dir=Path("data/scenes/syns_meg36"),
        allow_build=True,
        logger=_log,
    )
    print(
        "Done. Shared MEG assets are ready at "
        f"{DEFAULT_MEG_PRECOMPUTED_ROOT} (from_cache={prepared.shared_from_cache})."
    )


if __name__ == "__main__":
    main()
