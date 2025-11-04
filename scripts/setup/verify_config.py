#!/usr/bin/env python3
"""Validate configuration files and staged assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_env import (
    get_model_identifiers,
    get_model_root,
    get_runtime_preferences,
)

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORTS_DIR / "config_report.json"
REQUIRED_METADATA = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
}
REQUIRED_TOKENIZER = {
    "tokenizer.json",
}


def check_paths(root: Path) -> Dict[str, Any]:
    exists = root.exists()
    result: Dict[str, Any] = {
        "weights": {"path": str(root), "exists": exists},
        "tokenizer": {"path": str(root), "exists": exists},
        "metadata": {"path": str(root), "exists": exists},
        "missing_files": [],
    }

    missing: List[str] = []

    if not exists:
        missing.extend(
            [
                "weights directory",
                "tokenizer directory",
                "metadata directory",
            ]
        )
    else:
        shards = sorted(root.glob("model-*.safetensors"))
        if not shards:
            missing.append("safetensors shards in weights directory")

        for name in REQUIRED_TOKENIZER:
            if not (root / name).exists():
                missing.append(f"tokenizer/{name}")

        for name in REQUIRED_METADATA:
            if not (root / name).exists():
                missing.append(f"metadata/{name}")

    result["missing_files"] = missing
    result["status"] = "ok" if not missing else "warn"
    return result


def main() -> int:
    model_name, _, _ = get_model_identifiers()
    runtime = get_runtime_preferences()

    model_root = get_model_root()
    validation = check_paths(model_root)

    report = {
        "model_name": model_name,
        "runtime": runtime,
        "paths": validation,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    if validation["status"] == "ok":
        print("✅ Configuration OK")
    else:
        print("⚠️  Configuration issues detected")
        for item in validation["missing_files"]:
            print(f"   → Missing {item}")

    print(f"Model: {model_name}")
    print(f"Device preference: {runtime.get('device', 'auto')} | dtype: {runtime.get('dtype', 'fp32')}")
    print(f"Weights: {model_root}")
    print(f"Tokenizer: {model_root}")
    print(f"Metadata: {model_root}")
    print(f"Report saved to {REPORT_PATH.relative_to(ROOT)}")

    return 0 if validation["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
