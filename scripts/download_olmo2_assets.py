#!/usr/bin/env python3
"""Download OLMo-2 weights/metadata, stage them under llm_raw/, and run basic checks."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = ROOT / "llm_raw" / "olmo_2"
HF_SNAPSHOT = RAW_ROOT / "hf_snapshot"
RAW_WEIGHTS = RAW_ROOT / "raw_weights"
RAW_TOKENIZER = RAW_ROOT / "raw_tokenizer"
METADATA = RAW_ROOT / "metadata"

WEIGHT_SUFFIXES = {".safetensors"}
TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
}

METADATA_FILES = {
    "config.json",
    "generation_config.json",
    "README.md",
    "model.safetensors.index.json",
}


def assets_present() -> bool:
    required_weights = sorted(f"model-0000{i}-of-00006.safetensors" for i in range(1, 7))
    if not RAW_WEIGHTS.exists():
        return False
    existing_weights = sorted(child.name for child in RAW_WEIGHTS.glob("model-*.safetensors"))
    if existing_weights != required_weights:
        return False

    if not RAW_TOKENIZER.exists():
        return False
    missing_tokenizers = [name for name in TOKENIZER_FILES if not (RAW_TOKENIZER / name).exists()]
    if missing_tokenizers:
        return False

    if not METADATA.exists():
        return False
    missing_metadata = [name for name in METADATA_FILES if not (METADATA / name).exists()]
    if missing_metadata:
        return False

    return True


def stage_files(snapshot_dir: Path) -> None:
    RAW_WEIGHTS.mkdir(parents=True, exist_ok=True)
    RAW_TOKENIZER.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)

    def move_file(src: Path, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        shutil.move(str(src), str(dest))

    for path in snapshot_dir.rglob("*"):
        if path.is_dir():
            continue
        rel = path.relative_to(snapshot_dir)
        if rel.name in TOKENIZER_FILES:
            move_file(path, RAW_TOKENIZER / rel.name)
        elif rel.suffix in WEIGHT_SUFFIXES:
            move_file(path, RAW_WEIGHTS / rel.name)
        elif rel.name in METADATA_FILES:
            move_file(path, METADATA / rel.name)

    shutil.rmtree(snapshot_dir, ignore_errors=True)


def run(cmd: list[str], env: dict[str, str]) -> None:
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=env)


def maybe_run_gpu_check(env: dict[str, str]) -> None:
    try:
        import torch  # type: ignore
    except Exception:
        print("[warn] torch not available; skipping GPU verification")
        return
    if not torch.cuda.is_available():  # type: ignore[attr-defined]
        print("[info] CUDA not available; skipping GPU verification")
        return
    run(["python", "inference/Olmo_2/check_gpu.py", "--prompt", "GPU health check"], env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OLMo-2 assets and verify layout.")
    parser.add_argument("--model-id", default="allenai/OLMo-2-1124-13B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--revision", default=None, help="Optional model revision/commit hash")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if assets already exist under llm_raw/olmo_2/",
    )
    args = parser.parse_args()

    if assets_present() and not args.force:
        print("[cache] Existing weights/metadata detected under llm_raw/olmo_2; skipping download.")
    else:
        HF_SNAPSHOT.mkdir(parents=True, exist_ok=True)
        print(f"[download] Fetching {args.model_id} ...")
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_dir=HF_SNAPSHOT,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        stage_files(HF_SNAPSHOT)
        print("[stage] Weights and metadata staged under llm_raw/olmo_2")

    env = dict(os.environ)
    env.setdefault("PYTHONPATH", str(ROOT))

    run(["python", "llm_setup/analysis/test_analysis.py"], env=env)
    run(["python", "llm_setup/analysis/get_tensor_shapes_form_safetensors.py"], env=env)
    run(["python", "llm_setup/analysis/verify_tensor_extraction.py", "--tensor-name", "lm_head.weight"], env=env)

    maybe_run_gpu_check(env)

    print("[done] Asset download and validation completed.")


if __name__ == "__main__":
    main()
