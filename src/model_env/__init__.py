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


def compute_snapshot_dir(model_name: str, model_variant: str) -> Path:
    base = ROOT / "weights" / model_name
    variant_path = Path(model_variant)
    return base.joinpath(*variant_path.parts).expanduser().resolve()


def get_snapshot_dir(
    override: str | os.PathLike[str] | None = None,
    *,
    model_name: str | None = None,
    model_variant: str | None = None,
) -> Path:
    direct_override = _resolve_optional_path(override)
    if direct_override is not None:
        return direct_override

    env_override = _resolve_optional_path(os.environ.get(MODEL_SNAPSHOT_ENV))
    if env_override is not None:
        return env_override

    if model_name is None or model_variant is None:
        model_name, model_variant, _ = get_model_identifiers()

    return compute_snapshot_dir(model_name, model_variant)


def get_tokenizer_dir(snapshot_dir: Path | None = None) -> Path:
    env_dir = _resolve_optional_path(os.environ.get(MODEL_TOKENIZER_ENV))
    if env_dir is not None:
        return env_dir
    if snapshot_dir is None:
        snapshot_dir = get_snapshot_dir()
    return snapshot_dir


def get_metadata_dir(snapshot_dir: Path | None = None) -> Path:
    env_dir = _resolve_optional_path(os.environ.get(MODEL_CONFIG_ENV))
    if env_dir is not None:
        return env_dir
    if snapshot_dir is None:
        snapshot_dir = get_snapshot_dir()
    metadata_dir = snapshot_dir / "metadata"
    if metadata_dir.exists():
        return metadata_dir
    return snapshot_dir


def get_model_paths() -> Dict[str, Path]:
    snapshot_dir = get_snapshot_dir()
    tokenizer_dir = get_tokenizer_dir(snapshot_dir)
    metadata_dir = get_metadata_dir(snapshot_dir)
    return {
        "weights": snapshot_dir,
        "tokenizer": tokenizer_dir,
        "metadata": metadata_dir,
    }


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
    "get_metadata_dir",
    "get_model_identifiers",
    "get_model_paths",
    "get_runtime_preferences",
    "get_snapshot_dir",
    "get_tokenizer_dir",
    "load_model_config",
]
