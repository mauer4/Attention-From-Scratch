#!/usr/bin/env python3
"""Legacy downloader retained for reference. Prefer scripts/download_weights.py."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from huggingface_hub import snapshot_download

from model_env import get_model_paths
MODEL_PATHS = get_model_paths()
WEIGHTS_DIR = MODEL_PATHS["weights"]
TOKENIZER_DIR = MODEL_PATHS["tokenizer"]
METADATA_DIR = MODEL_PATHS["metadata"]
HF_SNAPSHOT = WEIGHTS_DIR / "_hf_snapshot"

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
    if not WEIGHTS_DIR.exists():
        return False
    existing_weights = sorted(child.name for child in WEIGHTS_DIR.glob("model-*.safetensors"))
    if existing_weights != required_weights:
        return False

    if not TOKENIZER_DIR.exists():
        return False
    missing_tokenizers = [name for name in TOKENIZER_FILES if not (TOKENIZER_DIR / name).exists()]
    if missing_tokenizers:
        return False

    if not METADATA_DIR.exists():
        return False
    missing_metadata = [name for name in METADATA_FILES if not (METADATA_DIR / name).exists()]
    if missing_metadata:
        return False

    return True


def stage_files(snapshot_dir: Path) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

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
            move_file(path, TOKENIZER_DIR / rel.name)
        elif rel.suffix in WEIGHT_SUFFIXES:
            move_file(path, WEIGHTS_DIR / rel.name)
        elif rel.name in METADATA_FILES:
            move_file(path, METADATA_DIR / rel.name)

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
        help="Force download even if assets already exist under weights/",
    )
    args = parser.parse_args()

    if assets_present() and not args.force:
        print("[cache] Existing weights/metadata detected under the configured weights directory; skipping download.")
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
        print(f"[stage] Weights and metadata staged under {WEIGHTS_DIR.parent}")

    env = dict(os.environ)
    pythonpath_entries = [str(ROOT), str(SRC_DIR)]
    existing = env.get("PYTHONPATH")
    combined = pythonpath_entries + ([existing] if existing else [])
    env["PYTHONPATH"] = os.pathsep.join(
        [entry for entry in combined if entry]
    )

    run(["python", "llm_setup/analysis/test_analysis.py"], env=env)
    run(["python", "llm_setup/analysis/get_tensor_shapes_form_safetensors.py"], env=env)
    run(["python", "llm_setup/analysis/verify_tensor_extraction.py", "--tensor-name", "lm_head.weight"], env=env)

    maybe_run_gpu_check(env)

    print("[done] Asset download and validation completed.")


if __name__ == "__main__":
    main()
