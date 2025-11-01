#!/usr/bin/env python3
"""Legacy shim for GPU profiler; delegates to analysis/gpu_profile_report.py."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
analysis_dir = ROOT / "analysis"
if str(analysis_dir) not in sys.path:
    sys.path.insert(0, str(analysis_dir))


def main() -> None:
    print(
        "⚠️  inference.Olmo_2.check_gpu_profiler is deprecated. "
        "Use analysis/gpu_profile_report.py instead.",
        file=sys.stderr,
    )
    module = importlib.import_module("gpu_profile_report")
    module.main()


if __name__ == "__main__":
    main()
