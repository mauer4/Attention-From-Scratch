"""Configuration helpers for the Attention-From-Scratch project."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml
from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"
ENV_FILE = ROOT / ".env"


def _ensure_yaml_suffix(name: str) -> str:
    if name.endswith((".yaml", ".yml")):
        return name
    return f"{name}.yaml"


def _resolve_config_path(value: str | Path) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    for base in (CONFIG_DIR, ROOT):
        path = base / _ensure_yaml_suffix(str(candidate))
        if path.exists():
            return path
    return (CONFIG_DIR / _ensure_yaml_suffix(str(candidate))).resolve()


def _merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_config(path: Path, visited: Iterable[Path] | None = None) -> Dict[str, Any]:
    visited = set(visited or [])
    if path in visited:
        raise ValueError(f"Cyclic config include detected via {path}")
    visited.add(path)

    data = yaml.safe_load(path.read_text()) or {}
    include = data.pop("include", None)
    if include:
        base_path = _resolve_config_path(include)
        parent_cfg = _load_config(base_path, visited)
        return _merge_dict(parent_cfg, data)
    return data


def load_config(config: str | Path | None = None) -> Dict[str, Any]:
    """Load configuration merged with .env overrides."""
    cfg_name = config or os.environ.get("AFS_CONFIG", "default")
    cfg_path = _resolve_config_path(cfg_name)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {cfg_path}")

    cfg = _load_config(cfg_path)

    env_values = dotenv_values(ENV_FILE) if ENV_FILE.exists() else {}
    if env_values:
        cfg.setdefault("env", {})
        cfg["env"] = _merge_dict(
            cfg["env"],
            {k: v for k, v in env_values.items() if v is not None},
        )
    return cfg


def resolve_path(value: str | os.PathLike[str]) -> Path:
    path = Path(os.path.expandvars(str(value)))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def get_model_paths(config: Dict[str, Any] | None = None) -> Dict[str, Path]:
    cfg = config or load_config()
    model_cfg = cfg.get("model", {})
    weights_dir = resolve_path(model_cfg.get("weights_dir", "weights/olmo2"))
    tokenizer_dir = resolve_path(model_cfg.get("tokenizer_dir", weights_dir / "tokenizer"))
    metadata_dir = resolve_path(model_cfg.get("metadata_dir", weights_dir / "metadata"))
    return {
        "weights": weights_dir,
        "tokenizer": tokenizer_dir,
        "metadata": metadata_dir,
    }


def get_runtime_preferences(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    cfg = config or load_config()
    return cfg.get("runtime", {})


__all__ = ["load_config", "get_model_paths", "get_runtime_preferences", "resolve_path"]
