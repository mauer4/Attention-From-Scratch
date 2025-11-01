#!/usr/bin/env python3
"""Deprecated entry point kept for backward compatibility."""

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
        "⚠️  inference.Olmo_2.run_snapshot_sanity is deprecated. "
        "Use scripts/run_snapshot_sanity.py instead.",
        file=sys.stderr,
    )
    module = importlib.import_module("run_snapshot_sanity")
    module.main()


if __name__ == "__main__":
    main()
