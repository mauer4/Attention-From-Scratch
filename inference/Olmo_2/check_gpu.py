#!/usr/bin/env python3
"""Run the GPU health check from llm_raw.olmo_2.test.

This wrapper prepares a HF-style snapshot view from the staged
`llm_raw/olmo_2` assets (weights, metadata, tokenizer) so the
underlying test can load the model from disk without hitting the
Hugging Face Hub.
"""
from __future__ import annotations

import sys
import tempfile
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_raw.olmo_2.test.check_olmo_gpu import main  # noqa: E402


def _prepare_local_snapshot_view(root: Path) -> tuple[tempfile.TemporaryDirectory | None, Path]:
    """Create a temporary directory that contains HF-style model files by
    linking staged assets from llm_raw/olmo_2. Returns (tempdir_obj, view_path).

    If the staged assets are missing, returns (None, Path()) so the caller
    can fall back to default behavior.
    """
    weights_dir = root / "llm_raw" / "olmo_2" / "raw_weights"
    metadata_dir = root / "llm_raw" / "olmo_2" / "metadata"
    tokenizer_dir = root / "llm_raw" / "olmo_2" / "raw_tokenizer"

    if not weights_dir.exists() or not metadata_dir.exists():
        return None, Path("")

    temp_dir = tempfile.TemporaryDirectory()
    view_path = Path(temp_dir.name)

    def link_if_exists(src: Path) -> None:
        if not src.exists() or not src.is_file():
            return
        dest = view_path / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        try:
            os.symlink(src, dest)
        except OSError:
            # If symlink fails (e.g., on some CI filesystems), copy as a fallback.
            from shutil import copy2

            copy2(src, dest)

    # Link weight shard files
    for file_path in weights_dir.glob("*"):
        link_if_exists(file_path)

    # Link metadata files such as config.json and safetensors index
    for metadata_name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        link_if_exists(metadata_dir / metadata_name)

    # Link tokenizer files into the same view so AutoTokenizer finds them locally
    if tokenizer_dir.exists():
        for token_file in tokenizer_dir.glob("*"):
            link_if_exists(token_file)

    return temp_dir, view_path

if __name__ == "__main__":
    # Prepare a local HF-style snapshot view when possible and pass it as
    # --model-id so the test loads the model from staged assets instead of
    # attempting to download from the Hub.
    tmpobj, view = _prepare_local_snapshot_view(ROOT)
    original_argv = sys.argv[:]
    try:
        if view and view.exists():
            # Prepend script name and ensure any user-provided args are preserved.
            # If the caller already provided --model-id, prefer the caller's value.
            has_model_id = any(a.startswith("--model-id") for a in sys.argv[1:])
            if not has_model_id:
                sys.argv = [sys.argv[0], "--model-id", str(view)] + sys.argv[1:]
        main()
    finally:
        # Restore argv and clean up temporary view directory
        sys.argv = original_argv
        if tmpobj is not None:
            try:
                tmpobj.cleanup()
            except Exception:
                pass