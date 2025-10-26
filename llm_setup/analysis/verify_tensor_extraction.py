"""
Cross-check manual safetensor extraction against safetensors.safe_open.

For each requested tensor the script loads data via byte offsets and via
safe_open, then compares flattened tensors to catch mismatches caused by
path or format regressions.
"""

import argparse
import json
import struct
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from safetensors.torch import safe_open


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


def read_header(path: Path) -> Tuple[Dict[str, Dict[str, List[int]]], int]:
    """Return (header, header_byte_length) for a safetensors file."""
    with path.open("rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    header.pop("__metadata__", None)
    return header, header_len


def load_flat_manual(shard_path: Path, meta: Dict[str, List[int]], header_len: int) -> torch.Tensor:
    dtype_key = meta["dtype"]
    if dtype_key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype_key}' in {shard_path.name}")
    dtype = DTYPE_MAP[dtype_key]
    shape = meta["shape"]
    start, end = meta["data_offsets"]
    num_bytes = end - start

    with shard_path.open("rb") as handle:
        handle.seek(8 + header_len + start)
        raw = handle.read(num_bytes)

    tensor = torch.frombuffer(memoryview(raw), dtype=dtype).clone().reshape(shape).reshape(-1)
    return tensor


def load_flat_safe(shard_path: Path, tensor_name: str) -> torch.Tensor:
    with safe_open(shard_path, framework="pt") as f:
        tensor = f.get_tensor(tensor_name).clone().reshape(-1)
    return tensor


def verify_tensor(
    tensor_name: str,
    shard_path: Path,
    meta: Dict[str, List[int]],
    header_len: int,
    *,
    verbose: bool = False,
) -> bool:
    manual_vec = load_flat_manual(shard_path, meta, header_len)
    safe_vec = load_flat_safe(shard_path, tensor_name)

    if manual_vec.shape != safe_vec.shape:
        if verbose:
            print(f"[FAIL] {tensor_name}: shape mismatch {manual_vec.shape} vs {safe_vec.shape}")
        return False

    if manual_vec.dtype != safe_vec.dtype:
        safe_vec = safe_vec.to(manual_vec.dtype)

    if manual_vec.is_floating_point():
        equal = torch.allclose(manual_vec.float(), safe_vec.float(), rtol=0.0, atol=0.0)
    else:
        equal = torch.equal(manual_vec, safe_vec)

    if verbose:
        status = "OK" if equal else "FAIL"
        numel = manual_vec.numel()
        print(f"[{status}] {tensor_name} | elements={numel:,} | dtype={manual_vec.dtype}")
    return equal


def iter_tensor_names(weight_map: Dict[str, str], selected: Iterable[str]) -> List[str]:
    if selected:
        names = []
        for name in selected:
            if name not in weight_map:
                raise KeyError(f"Tensor '{name}' not found in weight map")
            names.append(name)
        return names
    return sorted(weight_map.keys())


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify manual safetensor extraction against safe_open.")
    parser.add_argument(
        "--tensor-name",
        action="append",
        help="Specific tensor name to verify. May be provided multiple times. "
        "If omitted, all tensors are checked.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first mismatch.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a line for each tensor.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    data_root = root / "llm_raw" / "olmo_2"
    weights_dir = data_root / "raw_weights"
    metadata_dir = data_root / "metadata"

    index = json.loads((metadata_dir / "model.safetensors.index.json").read_text())
    weight_map: Dict[str, str] = index["weight_map"]

    shards = {shard: read_header(weights_dir / shard) for shard in set(weight_map.values())}

    names = iter_tensor_names(weight_map, args.tensor_name or [])

    failures = 0
    for idx, tensor_name in enumerate(names, start=1):
        shard_name = weight_map[tensor_name]
        header, header_len = shards[shard_name]
        meta = header[tensor_name]
        if not verify_tensor(
            tensor_name,
            weights_dir / shard_name,
            meta,
            header_len,
            verbose=args.verbose,
        ):
            failures += 1
            if args.fail_fast:
                break

    total = len(names)
    if failures == 0:
        print(f"Verified {total} tensor(s); all matched.")
    else:
        print(f"Verified {total} tensor(s); {failures} mismatch(es) found.")


if __name__ == "__main__":
    main()
