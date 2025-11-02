#!/usr/bin/env python3
"""Validate configuration files and staged assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_env import (
    get_model_identifiers,
    get_model_paths,
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


def check_paths(paths: Dict[str, Path]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "weights": {"path": str(paths["weights"]), "exists": paths["weights"].exists()},
        "tokenizer": {"path": str(paths["tokenizer"]), "exists": paths["tokenizer"].exists()},
        "metadata": {"path": str(paths["metadata"]), "exists": paths["metadata"].exists()},
        "missing_files": [],
    }

    missing: List[str] = []

    if result["weights"]["exists"]:
        shards = sorted(paths["weights"].glob("model-*.safetensors"))
        if not shards:
            missing.append("safetensors shards in weights directory")
    else:
        missing.append("weights directory")

    if result["tokenizer"]["exists"]:
        for name in REQUIRED_TOKENIZER:
            if not (paths["tokenizer"] / name).exists():
                missing.append(f"tokenizer/{name}")
    else:
        missing.append("tokenizer directory")

    if result["metadata"]["exists"]:
        for name in REQUIRED_METADATA:
            if not (paths["metadata"] / name).exists():
                missing.append(f"metadata/{name}")
    else:
        missing.append("metadata directory")

    result["missing_files"] = missing
    result["status"] = "ok" if not missing else "warn"
    return result


def main() -> int:
    model_name, _, _ = get_model_identifiers()
    runtime = get_runtime_preferences()

    paths = get_model_paths()
    validation = check_paths(paths)

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
    print(f"Weights: {paths['weights']}")
    print(f"Tokenizer: {paths['tokenizer']}")
    print(f"Metadata: {paths['metadata']}")
    print(f"Report saved to {REPORT_PATH.relative_to(ROOT)}")

    return 0 if validation["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
