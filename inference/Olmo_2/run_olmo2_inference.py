#!/usr/bin/env python3
"""Shim to preserve legacy import paths for OLMo inference."""

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
        "⚠️  inference.Olmo_2.run_olmo2_inference is deprecated. "
        "Use scripts/run_olmo2_inference.py instead.",
        file=sys.stderr,
    )
    module = importlib.import_module("run_olmo2_inference")
    module.main()


if __name__ == "__main__":
    main()
