#!/usr/bin/env python3
"""Sweep OLMo-2 inference latency across prompt/output lengths."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import os

from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_env import (  # noqa: E402  pylint: disable=wrong-import-position
    MODEL_TOKENIZER_ENV,
    get_model_identifiers,
    get_model_root,
)

RUN_SCRIPT = REPO_ROOT / "inference" / "Olmo_2" / "run_olmo2_inference.py"
REPORTS_ROOT = REPO_ROOT / "reports"
GPU_INFO_PATH = REPORTS_ROOT / "system_gpu.json"
DEFAULT_PROMPT_TOKEN = " stress"
DEFAULT_MIN_LENGTH = 8
DEFAULT_MULTIPLIER = 2


DEFAULT_MODEL_NAME, _, DEFAULT_REPO_ID = get_model_identifiers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stress-test OLMo inference across prompt/output lengths.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Name used when staging model weights.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Optional path to a specific checkpoint directory (overrides --model-name lookup).",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=None,
        help=f"Optional tokenizer directory (defaults to ${MODEL_TOKENIZER_ENV} or model dir).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--min-input-length", type=int, default=DEFAULT_MIN_LENGTH)
    parser.add_argument("--min-output-length", type=int, default=DEFAULT_MIN_LENGTH)
    parser.add_argument(
        "--multiplier",
        "--step",
        dest="multiplier",
        type=int,
        default=DEFAULT_MULTIPLIER,
        help="Multiplicative growth factor applied to each successive length (default 2).",
    )
    parser.add_argument(
        "--max-input-length",
        type=int,
        default=None,
        help="Optional cap for input sweep (defaults to model context window).",
    )
    parser.add_argument(
        "--max-output-length",
        type=int,
        default=None,
        help="Optional cap for output sweep (defaults to model context window).",
    )
    parser.add_argument(
        "--context-buffer",
        type=int,
        default=32,
        help="Reserve this many tokens from the reported context window as safety slack.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many input/output combinations (for quick smoke tests).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPORTS_ROOT,
        help="Base directory for writing stress test artifacts (defaults to repo reports/).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without invoking the model.")
    parser.add_argument(
        "--do-sample",
        dest="do_sample",
        action="store_true",
        help="Enable sampling for downstream runs (default).",
    )
    parser.add_argument(
        "--no-do-sample",
        dest="do_sample",
        action="store_false",
        help="Disable sampling for downstream runs.",
    )
    parser.set_defaults(do_sample=True)
    return parser.parse_args()


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


def load_model_config(model_dir: Path) -> Dict[str, Any]:
    candidates = [model_dir / "config.json", model_dir / "metadata" / "config.json"]
    for cand in candidates:
        if cand.exists():
            try:
                return json.loads(cand.read_text())
            except json.JSONDecodeError as exc:  # pragma: no cover - configuration issue
                raise ValueError(f"Failed to parse {cand}: {exc}") from exc
    raise FileNotFoundError("config.json not found in model directory or metadata/")


def extract_context_limit(config: Dict[str, Any]) -> int | None:
    keys = ("max_position_embeddings", "context_window", "n_ctx")
    for key in keys:
        value = config.get(key)
        if isinstance(value, int):
            return value
    for subkey in ("model", "params", "config"):
        nested = config.get(subkey)
        if isinstance(nested, dict):
            for key in keys:
                value = nested.get(key)
                if isinstance(value, int):
                    return value
    return None


def sanitize_segment(name: str | None, fallback: str) -> str:
    text = (name or fallback).strip() or fallback
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    sanitized = sanitized.strip("._-") or fallback
    return sanitized


def load_gpu_name(report_path: Path = GPU_INFO_PATH) -> str:
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text())
            details = data.get("gpu", {}).get("details") or []
            if details:
                name = details[0].get("name")
                if name:
                    return name
        except json.JSONDecodeError:
            return "unknown_gpu"
    return "unknown_gpu"


def ensure_positive(value: int, label: str) -> None:
    if value <= 0:
        raise ValueError(f"{label} must be positive (got {value}).")


def generate_lengths(start: int, stop: int, multiplier: int) -> List[int]:
    ensure_positive(multiplier, "multiplier")
    ensure_positive(start, "min length")
    if stop < start:
        return []
    if multiplier == 1:
        return list(range(start, stop + 1))
    values: List[int] = []
    current = start
    while current <= stop:
        values.append(current)
        next_value = current * multiplier
        if next_value == current:
            break
        current = next_value
    return values


def build_prompt(tokenizer: AutoTokenizer, base_ids: List[int], length: int) -> str:
    if length <= 0:
        return ""
    repeated: List[int] = []
    while len(repeated) < length:
        repeated.extend(base_ids)
    repeated = repeated[:length]
    tokens = tokenizer.convert_ids_to_tokens(repeated)
    return tokenizer.convert_tokens_to_string(tokens)


def parse_json_output(payload: str) -> Dict[str, Any]:
    text = payload.strip()
    if not text:
        raise ValueError("Empty analyzer output from run_olmo2_inference.py")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        last_brace = text.rfind("{")
        if last_brace == -1:
            raise
        return json.loads(text[last_brace:])


def invoke_inference(
    prompt: str,
    model_args: Dict[str, Any],
    max_new_tokens: int,
    do_sample: bool,
) -> Dict[str, Any]:
    cmd: List[str] = [sys.executable, str(RUN_SCRIPT)]
    cmd.extend(["--model-name", model_args["model_name"]])
    if model_args.get("model_dir") is not None:
        cmd.extend(["--model-dir", str(model_args["model_dir"])])
    if model_args.get("tokenizer_dir") is not None:
        cmd.extend(["--tokenizer-dir", str(model_args["tokenizer_dir"])])
    cmd.extend(["--device", model_args["device"], "--temperature", str(model_args["temperature"])])
    cmd.extend(["--max-new-tokens", str(max_new_tokens), "--prompt", prompt, "--analyze", "--no-print"])
    if do_sample:
        cmd.append("--do-sample")
    else:
        cmd.append("--no-do-sample")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Inference command failed (code {result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
        )
    return parse_json_output(result.stdout)


def main() -> None:
    args = parse_args()
    hf_logging.set_verbosity_error()

    if args.model_dir is not None:
        model_dir = args.model_dir.expanduser().resolve()
        variant_label = model_dir.name
    else:
        variant_label = DEFAULT_REPO_ID if args.model_name == DEFAULT_MODEL_NAME else args.model_name
        model_dir = get_model_root(model_variant=variant_label)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    tokenizer_dir = determine_tokenizer_dir(
        model_dir,
        args.tokenizer_dir.expanduser().resolve() if args.tokenizer_dir else None,
    )

    gpu_name = load_gpu_name()

    config = load_model_config(model_dir)
    context_limit = extract_context_limit(config)
    if context_limit is None:
        if args.max_input_length is None:
            raise ValueError("Context window could not be inferred; please pass --max-input-length explicitly.")
        context_limit = args.max_input_length

    usable_context = context_limit - max(args.context_buffer, 0)
    if usable_context <= 0:
        raise ValueError("Context buffer exceeds the reported context window.")

    max_input = min(args.max_input_length or usable_context, usable_context)
    max_output = min(args.max_output_length or usable_context, usable_context)

    input_lengths = generate_lengths(args.min_input_length, max_input, args.multiplier)
    output_lengths = generate_lengths(args.min_output_length, max_output, args.multiplier)
    combinations: List[Tuple[int, int]] = []
    for inp in input_lengths:
        for outp in output_lengths:
            if inp + outp > usable_context:
                continue
            combinations.append((inp, outp))
    if args.limit is not None:
        combinations = combinations[: args.limit]
    if not combinations:
        raise ValueError("No valid input/output length pairs computed; adjust limits or step size.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True, use_fast=False)
    base_ids = tokenizer.encode(DEFAULT_PROMPT_TOKEN, add_special_tokens=False)
    if not base_ids:
        base_ids = [tokenizer.eos_token_id or 0]

    model_args = {
        "model_name": args.model_name,
        "model_dir": model_dir,
        "tokenizer_dir": tokenizer_dir,
        "device": args.device,
        "temperature": args.temperature,
    }

    output_root = args.output_root.expanduser().resolve()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target_dir = (
        output_root
        / "stress"
        / sanitize_segment(args.model_name, "model")
        / sanitize_segment(variant_label, "variant")
        / sanitize_segment(gpu_name, "gpu")
        / stamp
    ).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    results_path = target_dir / "results.jsonl"
    metadata = {
        "model_name": args.model_name,
        "model_variant": variant_label,
        "model_dir": str(model_dir),
        "tokenizer_dir": str(tokenizer_dir),
        "context_limit": context_limit,
        "usable_context": usable_context,
        "min_input_length": args.min_input_length,
        "min_output_length": args.min_output_length,
        "max_input_length": max_input,
        "max_output_length": max_output,
        "multiplier": args.multiplier,
        "temperature": args.temperature,
        "do_sample": args.do_sample,
        "device": args.device,
        "gpu_name": gpu_name,
        "combinations": len(combinations),
        "limit": args.limit,
        "context_buffer": args.context_buffer,
        "output_dir": str(target_dir),
    }
    (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    print(f"[STRESS] Writing results to {results_path}")
    for idx, (input_tokens, output_tokens) in enumerate(combinations, start=1):
        print(f"[STRESS] {idx}/{len(combinations)} -> input={input_tokens}, output={output_tokens}")
        if args.dry_run:
            continue
        prompt = build_prompt(tokenizer, base_ids, input_tokens)
        try:
            result = invoke_inference(prompt, model_args, output_tokens, args.do_sample)
        except Exception as exc:  # pragma: no cover - runtime errors handled interactively
            print(f"[ERROR] Failed combo input={input_tokens} output={output_tokens}: {exc}")
            continue
        record = {
            "target_input_tokens": input_tokens,
            "target_output_tokens": output_tokens,
            "result": result,
        }
        with results_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    if args.dry_run:
        print("[STRESS] Dry run complete. No inference commands executed.")
    else:
        print(f"[STRESS] Completed {len(combinations)} combinations.")


if __name__ == "__main__":
    main()
