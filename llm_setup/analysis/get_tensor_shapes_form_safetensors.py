"""
Generate a tensor inventory CSV from the OLMo safetensor shards.

The script inspects each shard header, recording tensor names, shapes,
parameter counts, and byte offsets so downstream tooling can reference them
without re-reading the entire checkpoint.
"""

import csv
import json
import math
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_env import get_model_paths


def read_header(path: Path) -> Dict[str, Dict[str, List[int]]]:
    """Return the safetensors header mapping tensor name -> metadata."""
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    header.pop("__metadata__", None)
    return header


def main() -> None:
    model_paths = get_model_paths()
    weights_dir = model_paths["weights"]
    metadata_dir = model_paths["metadata"]

    index_path = metadata_dir / "model.safetensors.index.json"
    weight_map: Dict[str, str] = json.loads(index_path.read_text())["weight_map"]

    shard_headers: Dict[str, Dict[str, Dict[str, List[int]]]] = {}
    for shard in set(weight_map.values()):
        shard_headers[shard] = read_header(weights_dir / shard)

    rows: List[Tuple[int, str, int, str, str, int]] = []
    for i, (tensor_name, shard) in enumerate(weight_map.items(), start=1):
        entry = shard_headers[shard][tensor_name]
        shape = entry["shape"]
        num_params = math.prod(shape)
        offset = entry["data_offsets"][0]
        rows.append(
            (
                i,
                tensor_name,
                num_params,
                "x".join(str(dim) for dim in shape),
                shard,
                offset,
            )
        )

    output_path = metadata_dir / "tensor_inventory.csv"
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["tensor_number", "tensor_name", "num_parameters", "shape", "shard_file", "data_offset_bytes"]
        )
        writer.writerows(rows)

    print(f"Wrote tensor metadata for {len(rows)} tensors to {output_path}")


if __name__ == "__main__":
    main()
