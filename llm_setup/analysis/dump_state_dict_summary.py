#!/usr/bin/env python3
"""
Inspect a Hugging Face snapshot (OLMo 2 or similar) and emit a summary of every
tensor contained in the safetensors shards. The script reconstructs a logical
state dict mapping without instantiating the full model, so no GPU is required.

Output mirrors the structure of `layer_summary.csv` while capturing the real
tensor shapes/dtypes reported by the checkpoint. Both a console table and a CSV
file are produced for easy diffing.
"""

import argparse
import csv
import json
import os
import struct
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple


def render_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    materialized_rows: List[List[str]] = []
    for row in rows:
        str_row = [str(cell) for cell in row]
        materialized_rows.append(str_row)
        for idx, cell in enumerate(str_row):
            widths[idx] = max(widths[idx], len(cell))

    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_line = "| " + " | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))) + " |"
    body_lines = [
        "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |" for row in materialized_rows
    ]
    return "\n".join([border, header_line, border] + body_lines + [border])


def read_weight_index(index_path: str) -> Dict[str, str]:
    with open(index_path, "r") as f:
        index = json.load(f)
    return index["weight_map"]


def read_safetensors_header(path: str) -> Dict[str, dict]:
    with open(path, "rb") as f:
        header_len_bytes = f.read(8)
        if len(header_len_bytes) != 8:
            raise ValueError(f"{path}: expected 8-byte header length, found {len(header_len_bytes)}")
        (header_len,) = struct.unpack("<Q", header_len_bytes)
        header_json = f.read(header_len)
        if len(header_json) != header_len:
            raise ValueError(f"{path}: truncated header (expected {header_len} bytes)")
        header = json.loads(header_json.decode("utf-8"))
    return {k: v for k, v in header.items() if k != "__metadata__"}


def derive_scope_and_name(tensor_name: str) -> Tuple[str, str]:
    parts = tensor_name.split(".")
    if len(parts) >= 2:
        scope = ".".join(parts[:-1])
        leaf = parts[-1]
        if leaf == "weight" and len(parts) >= 2:
            scope = ".".join(parts[:-2]) or parts[0]
            leaf = ".".join(parts[-2:])
        return scope, leaf
    return tensor_name, ""


def build_rows(snapshot_dir: str) -> Tuple[Sequence[str], List[Sequence[str]]]:
    index_path = os.path.join(snapshot_dir, "model.safetensors.index.json")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"{index_path} not found. Did you point at a Hugging Face snapshot directory?")

    weight_map = read_weight_index(index_path)
    grouped: Dict[str, List[str]] = defaultdict(list)
    for tensor_name, shard in weight_map.items():
        grouped[shard].append(tensor_name)

    headers = ["Scope", "Tensor", "Shape", "Dtype", "Shard"]
    rows: List[Sequence[str]] = []

    for shard, tensor_names in sorted(grouped.items()):
        shard_path = os.path.join(snapshot_dir, shard)
        if not os.path.exists(shard_path):
            raise FileNotFoundError(f"Expected shard {shard_path} (from index) but file is missing")
        header = read_safetensors_header(shard_path)
        for name in sorted(tensor_names):
            if name not in header:
                raise KeyError(f"{name} not present in shard {shard_path}")
            info = header[name]
            shape = " x ".join(str(dim) for dim in info["shape"])
            dtype = info["dtype"]
            scope, short_name = derive_scope_and_name(name)
            rows.append((scope, short_name or name, shape, dtype, shard))

    return headers, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump a state-dict style summary from a safetensors checkpoint.")
    parser.add_argument("input", type=str, help="Path to the Hugging Face snapshot directory")
    parser.add_argument(
        "--output",
        type=str,
        default="state_dict_summary.csv",
        help="Destination CSV file (default: %(default)s)",
    )
    args = parser.parse_args()

    headers, rows = build_rows(args.input)

    print("=== State Dict Summary ===")
    print(render_table(headers, rows))
    print(f"\nTotal tensors: {len(rows)}")

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)

    print(f"Wrote state dict summary to {args.output}")


if __name__ == "__main__":
    main()
