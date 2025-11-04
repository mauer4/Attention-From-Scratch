#!/usr/bin/env python3
"""Derive and print environment exports for model assets."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config" / "config.yaml"


def parse_simple_yaml(path: Path) -> dict[str, str]:
    """Parse a minimal key:value YAML mapping without external dependencies."""
    data: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def normalize_path(*parts: str) -> str:
    path = Path(parts[0])
    for segment in parts[1:]:
        path = path.joinpath(segment)
    return str(path.expanduser().resolve())


def main() -> int:
    if not CONFIG_PATH.exists():
        print(f"Config file missing: {CONFIG_PATH}", file=sys.stderr)
        return 1

    config = parse_simple_yaml(CONFIG_PATH)

    model_name = config.get("model")
    model_variant = config.get("model_variant")

    missing = [name for name, value in (("model", model_name), ("model_variant", model_variant)) if not value]
    if missing:
        print(f"Config {CONFIG_PATH} missing keys: {', '.join(missing)}", file=sys.stderr)
        return 1

    weights_root = Path(ROOT_DIR / "weights" / model_name)
    snapshot_dir = weights_root.joinpath(*Path(model_variant).parts)

    exports = {
        "PROJECT_ROOT": normalize_path(str(ROOT_DIR)),
        "MODEL_NAME": model_name,
        "MODEL_VARIANT": model_variant,
        "MODEL_REPO_ID": model_variant,
        "MODEL_SNAPSHOT_DIR": normalize_path(snapshot_dir),
        "MODEL_TOKENIZER_DIR": normalize_path(snapshot_dir),
        "MODEL_CONFIG_DIR": normalize_path(snapshot_dir),
        "MODEL_WEIGHTS_ROOT": normalize_path(snapshot_dir),
    }

    for key, value in exports.items():
        print(f"export {key}={shlex.quote(value)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
