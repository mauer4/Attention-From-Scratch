#!/usr/bin/env python3
"""Run the GPU health check from llm_raw.olmo_2.test."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_raw.olmo_2.test.check_olmo_gpu import main  # noqa: E402

if __name__ == "__main__":
    main()\n\n