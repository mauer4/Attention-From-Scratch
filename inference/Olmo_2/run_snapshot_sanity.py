#!/usr/bin/env python3
"""Run a basic snapshot inference and report whether CUDA is available."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from inference.Olmo_2.run_olmo2_inference import main as run_inference  # noqa: E402


def notify_platform(message: str, level: str = "notice") -> None:
    """Emit a best-effort notification that CI/CD platforms can pick up."""
    level = level.lower()
    gha_levels = {"notice": "::notice::", "warning": "::warning::", "error": "::error::"}
    prefix = gha_levels.get(level)
    if prefix:
        print(f"{prefix}{message}")
    else:
        print(message)


def main() -> None:
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(torch.cuda.current_device())
        notify_platform(f"OLMo snapshot sanity running on GPU: {device_name}")
    else:
        warning_msg = (
            "CUDA device not available; running OLMo snapshot sanity on CPU. "
            "Generation may be slow."
        )
        warnings.warn(warning_msg, RuntimeWarning, stacklevel=2)
        notify_platform(warning_msg, level="warning")

    run_inference()


if __name__ == "__main__":
    main()
