"""Helpers for locating staged model assets on the local filesystem."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ENV_VAR = "MODEL_WEIGHTS_ROOT"
DEFAULT_ROOT = "weights/olmo2"


def _resolve_root(value: str | os.PathLike[str]) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def get_storage_root() -> Path:
    """Return the base directory where model snapshots are staged."""
    return _resolve_root(os.environ.get(MODEL_ENV_VAR, DEFAULT_ROOT))


def get_model_dir(model_name: str) -> Path:
    """
    Compute the directory that should contain snapshot files for ``model_name``.

    The directory is derived from ``MODEL_WEIGHTS_ROOT`` when set. If that value
    contains a ``{model}`` placeholder it is formatted directly; otherwise the
    model name is appended to the storage root (defaulting to
    ``weights/olmo2/<model_name>`` under the repository root). When the root
    already points at a populated snapshot directory, fall back to it instead of
    nesting the model name a second time.
    """
    root_value = os.environ.get(MODEL_ENV_VAR)
    if root_value and "{model}" in root_value:
        return _resolve_root(root_value.format(model=model_name))

    root = get_storage_root()
    candidate = root / model_name

    if candidate.exists():
        return candidate

    sentinel_files = (
        "config.json",
        "model.safetensors.index.json",
        "generation_config.json",
        "tokenizer.json",
    )
    if root.exists() and any((root / name).exists() for name in sentinel_files):
        return root

    if root.name == model_name:
        return root

    return candidate


__all__ = ["get_model_dir", "get_storage_root", "MODEL_ENV_VAR", "DEFAULT_ROOT"]
