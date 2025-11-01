#!/usr/bin/env python3
"""Legacy shim to preserve historical import paths."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
scripts_dir = ROOT / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))


def main() -> None:
    print(
        "⚠️  inference.Olmo_2.run_from_snapshot is deprecated. "
        "Use scripts/run_from_snapshot.py instead.",
        file=sys.stderr,
    )
    module = importlib.import_module("run_from_snapshot")
    module.main()


if __name__ == "__main__":
    main()
