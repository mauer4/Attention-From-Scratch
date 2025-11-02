#!/usr/bin/env python3
"""
Offline inference script for OLMo-2 checkpoints downloaded from Hugging Face.
Loads local config/tokenizer/weight shards (HF format) and runs a single prompt.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_storage import get_model_dir

DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "olmo2")
DEFAULT_MODEL_DIR = get_model_dir(DEFAULT_MODEL_NAME)
DEFAULT_PROMPT = "Explain why attention mechanisms improved transformer models."
DEFAULT_MAX_NEW_TOKENS = 128
DEFAULT_TEMPERATURE = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local OLMo-2 inference (HF-format checkpoint).")
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help="Model nickname used when staging weights (defaults to MODEL_NAME env or 'olmo2').",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Directory containing the HF-style config/tokenizer/weight shards. "
        "Overrides --model-name/MODEL_WEIGHTS_ROOT when provided.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help="Directory containing tokenizer files (defaults to model-dir/tokenizer or model-dir).",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt text to feed into the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help="Number of tokens to generate beyond the prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device preference (auto selects CUDA when available).",
    )
    return parser.parse_args()


def resolve_device(preference: str) -> str:
    if preference == "cpu":
        return "cpu"
    if preference == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        print("Requested CUDA but none detected; falling back to CPU.")
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def select_dtype(device: str) -> torch.dtype:
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def _symlink(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    os.symlink(src, dest)


def prepare_snapshot_view(model_dir: Path) -> tuple[tempfile.TemporaryDirectory | None, Path]:
    """
    Ensure Hugging Face expects config/index files next to shard weights.
    If `config.json` lives in a `metadata/` subfolder, mirror it into a temp directory.
    """
    config_path = model_dir / "config.json"
    if config_path.exists():
        return None, model_dir

    metadata_dir = model_dir / "metadata"
    if not metadata_dir.exists():
        raise FileNotFoundError(
            f"Expected HF config next to weights or under metadata/, but neither found in {model_dir}."
        )

    tmp = tempfile.TemporaryDirectory()
    view_path = Path(tmp.name)

    for shard in model_dir.glob("*.safetensors*"):
        _symlink(shard, view_path / shard.name)

    for name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        src = metadata_dir / name
        if src.exists():
            _symlink(src, view_path / name)

    if not (view_path / "config.json").exists():
        raise FileNotFoundError("config.json missing in metadata; cannot construct HF snapshot view.")

    return tmp, view_path


def ensure_hf_model_type(snapshot_dir: Path) -> None:
    """Ensure the snapshot config declares the HF model type."""
    config_path = snapshot_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json missing at {config_path}")

    try:
        data = json.loads(config_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse {config_path}: {exc}") from exc

    updated = False

    if data.get("model_type") != "olmo2":
        data["model_type"] = "olmo2"
        updated = True

    arch = data.get("architectures")
    if not arch or "Olmo2ForCausalLM" not in arch:
        data["architectures"] = ["Olmo2ForCausalLM"]
        updated = True

    if updated:
        config_path.write_text(json.dumps(data, indent=2))


def determine_tokenizer_dir(model_dir: Path, override: Path | None) -> Path:
    if override is not None:
        if not override.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {override}")
        return override
    candidate = model_dir / "tokenizer"
    if candidate.exists():
        return candidate
    return model_dir


def safe_decode(tokenizer: AutoTokenizer, token_ids: torch.Tensor) -> str:
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    except TypeError:
        tokens = tokenizer.convert_ids_to_tokens(token_ids.tolist(), skip_special_tokens=True)
        tokens = [tok for tok in tokens if isinstance(tok, str)]
        return tokenizer.convert_tokens_to_string(tokens)


def main() -> None:
    args = parse_args()
    hf_logging.set_verbosity_error()

    if args.model_dir is not None:
        model_dir = args.model_dir.expanduser().resolve()
    else:
        model_dir = get_model_dir(args.model_name).expanduser().resolve()

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer_dir = determine_tokenizer_dir(
        model_dir,
        args.tokenizer_dir.expanduser().resolve() if args.tokenizer_dir else None,
    )

    snapshot_tmp, snapshot_dir = prepare_snapshot_view(model_dir)
    ensure_hf_model_type(snapshot_dir)

    device = resolve_device(args.device)
    dtype = select_dtype(device)
    device_map = "auto" if device == "cuda" else None

    print(f"[INFO] Using device={device}, dtype={dtype}")
    print("[INFO] Loading Hugging Face OLMo model weights...")

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        local_files_only=True,
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        snapshot_dir,
        torch_dtype=dtype,
        device_map=device_map,
        local_files_only=True,
    )
    model.eval()

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
        )

    completion = safe_decode(tokenizer, outputs[0])

    print("\n" + "=" * 80)
    print(completion)
    print("=" * 80)

    if snapshot_tmp is not None:
        snapshot_tmp.cleanup()


if __name__ == "__main__":
    main()
