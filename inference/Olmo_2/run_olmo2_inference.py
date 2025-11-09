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
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
GPU_INFO_PATH = REPO_ROOT / "reports" / "system_gpu.json"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_env import (
    MODEL_SNAPSHOT_ENV,
    MODEL_TOKENIZER_ENV,
    get_model_identifiers,
    get_model_root,
)

try:
    DEFAULT_MODEL_NAME, _, DEFAULT_REPO_ID = get_model_identifiers()
except (FileNotFoundError, ValueError):
    DEFAULT_MODEL_NAME = os.environ.get("MODEL_NAME", "olmo2")
    default_variant = os.environ.get("MODEL_VARIANT", "allenai/OLMo-2-1124-13B")
    DEFAULT_REPO_ID = os.environ.get("MODEL_REPO_ID", default_variant)

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
        f"Overrides --model-name/${MODEL_SNAPSHOT_ENV} when provided.",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help=f"Directory containing tokenizer files (defaults to ${MODEL_TOKENIZER_ENV} or model-dir).",
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
    parser.add_argument(
        "--no-print",
        action="store_true",
        help="Suppress all standard output (overrides --analyze JSON output).",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Emit a JSON report capturing model metadata and performance metrics.",
    )
    parser.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        help="Enable sampling during generation (default).",
    )
    parser.add_argument(
        "--no-do-sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling during generation.",
    )
    parser.set_defaults(do_sample=True)
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
    env_value = os.environ.get(MODEL_TOKENIZER_ENV)
    if env_value:
        env_path = Path(env_value).expanduser().resolve()
        if not env_path.exists():
            raise FileNotFoundError(f"Tokenizer directory not found: {env_path}")
        return env_path
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


def round_float(value: float | None, decimals: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), decimals)


def load_gpu_report(report_path: Path = GPU_INFO_PATH) -> Dict[str, Any] | None:
    if not report_path.exists():
        return None
    try:
        return json.loads(report_path.read_text())
    except json.JSONDecodeError:
        return {"error": f"Failed to parse {report_path}"}


def summarize_gpu_details(report: Dict[str, Any] | None) -> Dict[str, Any] | None:
    if report is None:
        return None
    if "error" in report:
        return {"error": report["error"]}

    gpu_section = report.get("gpu", {})
    details = gpu_section.get("details") or []
    first = details[0] if details else {}

    summary = {
        "name": first.get("name"),
        "driver_version": first.get("driver_version") or report.get("detected_driver_version"),
        "cuda_version": first.get("cuda_version") or report.get("detected_cuda_version"),
        "details": details,
    }
    warnings = gpu_section.get("warnings") or report.get("warnings")
    if warnings:
        summary["warnings"] = warnings
    return summary


def main() -> None:
    args = parse_args()
    hf_logging.set_verbosity_error()

    log_enabled = not args.no_print

    def log(message: str) -> None:
        if log_enabled:
            print(message)

    if args.model_dir is not None:
        model_dir = args.model_dir.expanduser().resolve()
        model_variant = str(model_dir)
    else:
        model_dir = get_model_root(model_variant=DEFAULT_REPO_ID)
        model_variant = DEFAULT_REPO_ID

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

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    log(f"[INFO] Using device={device}, dtype={dtype}")
    log("[INFO] Loading Hugging Face OLMo model weights...")

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

    if device == "cuda":
        torch.cuda.synchronize()
    generation_start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
        )
    if device == "cuda":
        torch.cuda.synchronize()
    generation_duration = time.perf_counter() - generation_start

    input_length = inputs["input_ids"].shape[-1]
    output_length = outputs.shape[-1]
    new_tokens = max(output_length - input_length, 0)

    completion = safe_decode(tokenizer, outputs[0])
    continuation = safe_decode(tokenizer, outputs[0, input_length:])

    if device == "cuda":
        max_memory_bytes = torch.cuda.max_memory_allocated()
    else:
        max_memory_bytes = None
    if generation_duration > 0 and new_tokens:
        tokens_per_second = new_tokens / generation_duration
    else:
        tokens_per_second = None

    rounded_time_seconds = round_float(generation_duration)
    rounded_tokens_per_second = round_float(tokens_per_second)
    rounded_temperature = round_float(args.temperature)
    max_memory_megabytes = (
        round_float(max_memory_bytes / (1024**2)) if max_memory_bytes is not None else None
    )

    if not args.no_print:
        print("\n" + "=" * 80)
        print(completion)
        print("=" * 80)

    analysis_payload: Dict[str, Any] | None = None
    if args.analyze:
        device_map_runtime = getattr(model, "hf_device_map", device_map)
        if isinstance(device_map_runtime, dict):
            device_map_serializable = {k: str(v) for k, v in device_map_runtime.items()}
        else:
            device_map_serializable = device_map_runtime
        gpu_summary = summarize_gpu_details(load_gpu_report())
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        analysis_payload = {
            "timestamp": timestamp,
            "model": {
                "name": args.model_name,
                "variant": model_variant,
                "config_name": getattr(model.config, "_name_or_path", None),
                "path": str(snapshot_dir),
                "tokenizer_path": str(tokenizer_dir),
            },
            "generation": {
                "prompt": args.prompt,
                "completion": completion,
                "continuation": continuation,
                "input_length_tokens": int(input_length),
                "output_length_tokens": int(output_length),
                "new_tokens": int(new_tokens),
                "max_new_tokens_requested": args.max_new_tokens,
                "temperature": rounded_temperature,
                "time_seconds": rounded_time_seconds,
                "tokens_per_second": rounded_tokens_per_second,
                "do_sample": args.do_sample,
            },
            "runtime": {
                "device_preference": args.device,
                "resolved_device": device,
                "dtype": str(dtype),
                "device_map": device_map_serializable,
                "max_memory_bytes": max_memory_bytes,
                "max_memory_megabytes": max_memory_megabytes,
                "torch_version": torch.__version__,
                "python_version": sys.version,
            },
        }
        if gpu_summary is not None:
            analysis_payload["runtime"]["gpu"] = gpu_summary
        print(json.dumps(analysis_payload, indent=2))

    if snapshot_tmp is not None:
        snapshot_tmp.cleanup()


if __name__ == "__main__":
    main()
