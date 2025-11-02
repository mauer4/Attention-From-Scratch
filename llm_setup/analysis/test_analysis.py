"""
Sanity checks for the OLMo analysis tooling.

Run this script to ensure required assets still exist and basic invariants hold
before relying on the other helpers.
"""

import csv
import json
import sys
from pathlib import Path
from typing import Dict, Iterable

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_env import get_model_root


def require_paths(label: str, paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{label} missing: {missing}")


def load_weight_index(metadata_dir: Path) -> Dict[str, str]:
    index_path = metadata_dir / "model.safetensors.index.json"
    with index_path.open("r", encoding="utf-8") as handle:
        index = json.load(handle)
    if "weight_map" not in index:
        raise KeyError(f"'weight_map' not present in {index_path}")
    weight_map = index["weight_map"]
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("weight_map is empty or malformed")
    return weight_map


def check_inventory(metadata_dir: Path, expected_rows: int) -> None:
    inventory_path = metadata_dir / "tensor_inventory.csv"
    with inventory_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    required_columns = {
        "tensor_number",
        "tensor_name",
        "num_parameters",
        "shape",
        "shard_file",
        "data_offset_bytes",
    }
    if set(reader.fieldnames or []) != required_columns:
        raise ValueError(f"Unexpected columns in {inventory_path}: {reader.fieldnames}")
    if len(rows) != expected_rows:
        raise ValueError(
            f"tensor_inventory.csv row count {len(rows)} does not match index size {expected_rows}"
        )


def main() -> None:
    model_root = get_model_root()
    weights_dir = model_root
    metadata_dir = model_root
    tokenizer_dir = model_root

    require_paths(
        "core directories",
        [weights_dir, metadata_dir, tokenizer_dir],
    )

    weight_map = load_weight_index(metadata_dir)
    shard_paths = [weights_dir / shard for shard in set(weight_map.values())]
    require_paths("weight shards", shard_paths)

    tokenizer_files = [
        tokenizer_dir / "tokenizer.json",
        tokenizer_dir / "tokenizer_config.json",
        tokenizer_dir / "vocab.json",
        tokenizer_dir / "merges.txt",
        tokenizer_dir / "special_tokens_map.json",
    ]
    require_paths("tokenizer files", tokenizer_files)

    check_inventory(metadata_dir, expected_rows=len(weight_map))

    print("Analysis checks passed: assets and metadata look consistent.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
