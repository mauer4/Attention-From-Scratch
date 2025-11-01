#!/usr/bin/env python3
"""Download model weights and stage them under weights/<model>."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from huggingface_hub import snapshot_download

from project_config import get_model_paths, load_config, resolve_path

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY = {
    "olmo2": "allenai/OLMo-2-1124-13B-Instruct",
}

METADATA_FILES = {
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "README.md",
}

TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "merges.txt",
    "vocab.json",
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_model(model_name: str, repo_id: str, dest: Path, revision: str | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=tmp_path,
            local_dir_use_symlinks=False,
            resume_download=True,
        )

        dest_tmp = dest.with_suffix(".staging")
        if dest_tmp.exists():
            shutil.rmtree(dest_tmp)
        shutil.move(str(tmp_path), str(dest_tmp))

        backup = None
        if dest.exists():
            backup = dest.with_suffix(".legacy")
            if backup.exists():
                shutil.rmtree(backup)
            shutil.move(str(dest), str(backup))

        if dest.exists():
            shutil.rmtree(dest)
        shutil.move(str(dest_tmp), str(dest))

        if backup and backup.exists():
            shutil.rmtree(backup)

    cache_dir = dest / ".cache"
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)


def restructure_snapshot(root: Path) -> None:
    metadata_dir = root / "metadata"
    tokenizer_dir = root / "tokenizer"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    for name in METADATA_FILES:
        src = root / name
        if src.exists():
            src.replace(metadata_dir / name)

    for name in TOKENIZER_FILES:
        src = root / name
        if src.exists():
            src.replace(tokenizer_dir / name)


def compute_file_hashes(root: Path) -> Dict[str, str]:
    file_hashes: Dict[str, str] = {}
    for file_path in root.rglob("*"):
        if file_path.is_file():
            relative = file_path.relative_to(root).as_posix()
            file_hashes[relative] = sha256sum(file_path)
    return file_hashes


def assets_present(root: Path) -> bool:
    if not root.exists():
        return False
    shards = list(root.glob("model-*.safetensors"))
    if not shards:
        return False

    metadata_dir = root / "metadata"
    tokenizer_dir = root / "tokenizer"
    if not metadata_dir.exists() or not tokenizer_dir.exists():
        return False

    if any((metadata_dir / name).exists() is False for name in METADATA_FILES):
        return False
    if any((tokenizer_dir / name).exists() is False for name in TOKENIZER_FILES):
        return False
    return True


def cleanup_temp_dirs(root: Path) -> None:
    for temp_dir in root.glob("_tmp_snapshot*"):
        if temp_dir.is_dir():
            shutil.rmtree(temp_dir, ignore_errors=True)


def write_manifest(model_name: str, dest: Path, file_hashes: Dict[str, str]) -> Path:
    manifest = {
        "model": model_name,
        "destination": str(dest),
        "files": [
            {"path": path, "sha256": digest, "size": (dest / path).stat().st_size}
            for path, digest in sorted(file_hashes.items())
        ],
    }
    manifest_path = REPORTS_DIR / f"weights_manifest_{model_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and stage model weights.")
    parser.add_argument("--model-name", default=None, help="Logical model name (defaults to config model).")
    parser.add_argument("--repo-id", default=None, help="Override Hugging Face repo id.")
    parser.add_argument("--dest", default=None, help="Override destination directory.")
    parser.add_argument("--revision", default=None, help="Optional repo revision or commit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    default_model = config.get("model", {}).get("name", "olmo2")
    model_name = args.model_name or default_model
    paths = get_model_paths(config)

    dest = resolve_path(args.dest) if args.dest else paths["weights"]
    repo_id = args.repo_id or MODEL_REGISTRY.get(model_name)
    if repo_id is None:
        print(f"❌ Unknown model '{model_name}'. Provide --repo-id.", file=sys.stderr)
        return 1

    if assets_present(dest) and not args.revision:
        print("ℹ️  Assets already present; skipping download.")
        file_hashes = compute_file_hashes(dest)
        manifest_path = write_manifest(model_name, dest, file_hashes)
        print(f"✅ Manifest refreshed at {manifest_path.relative_to(ROOT)}")
        return 0

    print(f"⚙️  Downloading {model_name} from {repo_id}")
    stage_model(model_name, repo_id, dest, args.revision)
    restructure_snapshot(dest)
    cleanup_temp_dirs(dest)

    file_hashes = compute_file_hashes(dest)

    manifest_path = write_manifest(model_name, dest, file_hashes)

    print(f"✅ Staged weights at {dest}")
    print(f"✅ Manifest written to {manifest_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
