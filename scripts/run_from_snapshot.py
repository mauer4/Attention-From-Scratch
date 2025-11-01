#!/usr/bin/env python3
"""Thin wrapper around inference.Olmo_2.run_olmo2_inference."""
from __future__ import annotations

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_olmo2_inference import main

if __name__ == "__main__":
    main()
