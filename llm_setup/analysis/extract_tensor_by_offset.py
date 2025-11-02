"""
Load a single tensor from an OLMo safetensor shard using its recorded offset.

Useful for spot-checking weights or exporting individual tensors as `.pt`
files without loading the entire checkpoint.
"""

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, Tuple

import sys

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch

from model_env import get_model_root


DTYPE_MAP: Dict[str, torch.dtype] = {
    "BF16": torch.bfloat16,
    "F16": torch.float16,
    "F32": torch.float32,
    "F64": torch.float64,
    "I8": torch.int8,
    "I16": torch.int16,
    "I32": torch.int32,
    "I64": torch.int64,
    "U8": torch.uint8,
    "BOOL": torch.bool,
}


def read_header(path: Path) -> Tuple[Dict[str, Dict[str, object]], int]:
    """Return (header, header_byte_length) for a safetensors file."""
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    return header, header_len


def find_tensor_by_offset(header: Dict[str, Dict[str, object]], offset: int) -> Tuple[str, Dict[str, object]]:
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        if start == offset:
            return name, meta
    raise KeyError(f"No tensor with data offset {offset}")


def load_tensor(path: Path, offset: int) -> Tuple[str, torch.Tensor]:
    header, header_len = read_header(path)
    tensor_name, meta = find_tensor_by_offset(header, offset)

    dtype_key = meta["dtype"]
    try:
        dtype = DTYPE_MAP[dtype_key]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype '{dtype_key}'") from exc

    shape = meta["shape"]
    start, end = meta["data_offsets"]
    num_bytes = end - start

    with path.open("rb") as handle:
        handle.seek(8 + header_len + start)
        raw = handle.read(num_bytes)

    tensor = torch.frombuffer(memoryview(raw), dtype=dtype).clone().reshape(shape)
    return tensor_name, tensor


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a tensor from a safetensors shard by byte offset.")
    parser.add_argument("--file", required=True, type=Path, help="Path to the safetensors shard.")
    parser.add_argument("--offset", required=True, type=int, help="Byte offset inside the shard.")
    parser.add_argument("--save-pt", type=Path, help="Optional path to save the tensor with torch.save.")
    parser.add_argument("--to-float32", action="store_true", help="Convert tensor to float32 before saving.")
    args = parser.parse_args()

    default_weights_dir = get_model_root()
    shard_path = args.file
    if not shard_path.is_absolute():
        candidate = default_weights_dir / shard_path
        if candidate.exists():
            shard_path = candidate
    shard_path = shard_path.resolve()

    tensor_name, tensor = load_tensor(shard_path, args.offset)

    if args.to_float32 and tensor.is_floating_point():
        tensor = tensor.to(torch.float32)

    print(f"Tensor: {tensor_name}")
    print(f"Shape : {tuple(tensor.shape)}")
    print(f"Dtype : {tensor.dtype}")
    print(f"Params: {tensor.numel():,}")

    if args.save_pt:
        args.save_pt.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, args.save_pt)
        print(f"Saved tensor to {args.save_pt}")


if __name__ == "__main__":
    main()
