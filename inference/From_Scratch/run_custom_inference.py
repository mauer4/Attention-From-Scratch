#!/usr/bin/env python3
"""Utility to exercise the stub custom engine."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from custom_engine import load_engine
from inference import InferenceEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the custom inference stub.")
    parser.add_argument("prompt", help="Prompt string to convert into token IDs (ascii codes).")
    parser.add_argument("--max-new-tokens", type=int, default=8, dest="max_new_tokens")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    prompt_ids = [ord(ch) for ch in args.prompt]
    engine = InferenceEngine(load_engine())
    output = engine.run(prompt_ids, max_tokens=args.max_new_tokens)
    print("Generated token ids:", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
