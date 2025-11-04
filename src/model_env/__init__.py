"""Utilities for loading model configuration and resolving asset paths."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"

MODEL_NAME_ENV = "MODEL_NAME"
MODEL_VARIANT_ENV = "MODEL_VARIANT"
MODEL_REPO_ENV = "MODEL_REPO_ID"
MODEL_SNAPSHOT_ENV = "MODEL_SNAPSHOT_DIR"
MODEL_TOKENIZER_ENV = "MODEL_TOKENIZER_DIR"
MODEL_CONFIG_ENV = "MODEL_CONFIG_DIR"


def _parse_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _resolve_optional_path(value: str | os.PathLike[str] | None) -> Path | None:
    if value is None:
        return None
    path = Path(os.path.expanduser(str(value)))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def load_model_config() -> dict[str, str]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file missing: {CONFIG_PATH}")
    config = _parse_simple_yaml(CONFIG_PATH)
    missing = [name for name in ("model", "model_variant") if not config.get(name)]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Config {CONFIG_PATH} missing keys: {joined}")
    return config


def get_model_identifiers() -> Tuple[str, str, str]:
    config = load_model_config()
    model_name = os.environ.get(MODEL_NAME_ENV, config["model"])
    model_variant = os.environ.get(MODEL_VARIANT_ENV, config["model_variant"])
    repo_id = os.environ.get(MODEL_REPO_ENV, model_variant)
    return model_name, model_variant, repo_id


def compute_snapshot_dir(model_variant: str) -> Path:
    variant_path = Path(model_variant)
    return (ROOT / "weights").joinpath(*variant_path.parts).expanduser().resolve()


def get_model_root(
    override: str | os.PathLike[str] | None = None,
    *,
    model_variant: str | None = None,
) -> Path:
    direct_override = _resolve_optional_path(override)
    if direct_override is not None and direct_override.exists():
        return direct_override

    env_override_value = os.environ.get(MODEL_SNAPSHOT_ENV)
    env_override = _resolve_optional_path(env_override_value)
    if env_override is not None and env_override.exists():
        return env_override

    if model_variant is None:
        model_name, model_variant, _ = get_model_identifiers()
    else:
        model_name, _, _ = get_model_identifiers()

    primary_snapshot = compute_snapshot_dir(model_variant)
    if primary_snapshot.exists():
        return primary_snapshot

    legacy_snapshot = (ROOT / "weights" / model_name).joinpath(*Path(model_variant).parts)
    if legacy_snapshot.exists():
        return legacy_snapshot

    # Fall back to whichever override was provided, even if missing, so callers can surface helpful errors.
    if env_override is not None:
        return env_override
    if direct_override is not None:
        return direct_override
    return primary_snapshot


def get_runtime_preferences() -> Dict[str, str]:
    return {
        "device": os.environ.get("TORCH_DEVICE", "cuda"),
        "dtype": os.environ.get("TORCH_DTYPE", "float16"),
    }


__all__ = [
    "CONFIG_PATH",
    "MODEL_NAME_ENV",
    "MODEL_VARIANT_ENV",
    "MODEL_REPO_ENV",
    "MODEL_SNAPSHOT_ENV",
    "MODEL_TOKENIZER_ENV",
    "MODEL_CONFIG_ENV",
    "compute_snapshot_dir",
    "get_model_identifiers",
    "get_model_root",
    "get_runtime_preferences",
    "load_model_config",
]
