#!/usr/bin/env python3
"""Legacy shim: delegates to setup_env/check_gpu.py."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
setup_dir = ROOT / "setup_env"
if str(setup_dir) not in sys.path:
    sys.path.insert(0, str(setup_dir))


def main() -> None:
    print(
        "⚠️  inference.Olmo_2.check_gpu is deprecated. "
        "Use setup_env/check_gpu.py instead.",
        file=sys.stderr,
    )
    module = importlib.import_module("check_gpu")
    module.main()


if __name__ == "__main__":
    main()
