#!/usr/bin/env python3
"""Thin wrapper around inference.Olmo_2.run_olmo2_inference."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.Olmo_2.run_olmo2_inference import main  # noqa: E402

if __name__ == "__main__":
    main()
