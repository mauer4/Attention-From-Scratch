#!/usr/bin/env python3
"""Run a minimal inference sanity check and record telemetry."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

try:
    import torch
except ModuleNotFoundError as exc:  # pragma: no cover
    print("❌ PyTorch is not installed. Run setup_env/install_deps.sh first.")
    raise SystemExit(1) from exc

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ModuleNotFoundError as exc:  # pragma: no cover
    print("❌ transformers is not installed. Run setup_env/install_deps.sh first.")
    raise SystemExit(1) from exc

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_env import (
    get_model_identifiers,
    get_snapshot_dir,
    get_tokenizer_dir,
)

REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
JSON_PATH = REPORTS_DIR / "test_summary.json"
MARKDOWN_PATH = REPORTS_DIR / "test_summary.md"


def pick_device(preference: str) -> torch.device:
    normalized = preference.lower()
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if normalized == "cuda":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(weights: Path, tokenizer_dir: Path, device: torch.device) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(weights, local_files_only=True)
    model.to(device)
    model.eval()

    prompt = "Sanity test."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=2)
    duration = (time.perf_counter() - start) * 1000

    generated = tokenizer.decode(output[0], skip_special_tokens=True)
    metrics: Dict[str, Any] = {
        "prompt": prompt,
        "generated": generated,
        "elapsed_ms": round(duration, 2),
        "device": str(device),
        "tokens": output.shape[-1],
    }

    if device.type == "cuda":
        metrics["max_memory_mb"] = round(torch.cuda.max_memory_allocated(device) / (1024 * 1024), 2)
    return metrics


def main() -> int:
    try:
        default_model, _, default_repo = get_model_identifiers()
    except (FileNotFoundError, ValueError):
        default_model = os.environ.get("MODEL_NAME", "olmo2")
        default_variant = os.environ.get("MODEL_VARIANT", "allenai/OLMo-2-1124-13B")
        default_repo = os.environ.get("MODEL_REPO_ID", default_variant)

    model_name = os.environ.get("MODEL_NAME", default_model)
    preferred_device = os.environ.get("TORCH_DEVICE", "cuda")
    weights_path = get_snapshot_dir(
        None,
        model_name=model_name,
        model_variant=default_repo,
    )
    tokenizer_path = get_tokenizer_dir(weights_path)

    status = "passed"
    warnings = []
    results: Dict[str, Any] = {}

    missing_dirs = [
        label
        for label, path in (
            ("weights", weights_path),
            ("tokenizer", tokenizer_path),
        )
        if not path.exists()
    ]
    if missing_dirs:
        status = "failed"
        warnings.append(f"Missing required asset directories: {', '.join(missing_dirs)}")
    else:
        missing_assets = []
        required_snapshot_files = [
            "config.json",
            "generation_config.json",
            "model.safetensors.index.json",
        ]
        for name in required_snapshot_files:
            if not (weights_path / name).exists():
                missing_assets.append(name)

        if not list(weights_path.glob("model-*.safetensors")):
            missing_assets.append("model-*.safetensors")

        required_tokenizer_files = [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ]
        for name in required_tokenizer_files:
            if not (tokenizer_path / name).exists():
                missing_assets.append(f"tokenizer:{name}")

        if missing_assets:
            status = "failed"
            warnings.append(
                "Missing expected snapshot assets: "
                + ", ".join(sorted(missing_assets))
            )
        else:
            try:
                device = pick_device(preferred_device)
                if device.type == "cuda" and not torch.cuda.is_available():
                    warnings.append("CUDA requested but not available; falling back to CPU")
                    device = torch.device("cpu")
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                results = run_inference(weights_path, tokenizer_path, device)
                if device.type == "cpu":
                    warnings.append("Inference executed on CPU; enable CUDA for full validation")
            except Exception as exc:  # noqa: BLE001
                status = "failed"
                message = str(exc)
                if "tokenizer" in message and "config.json" in message:
                    warnings.append(
                        "Inference failed: tokenizer assets appear incomplete. "
                        "Run 'python scripts/download_weights.py --model-name olmo2' to restage files."
                    )
                else:
                    warnings.append(f"Inference failed: {exc}")

    summary = {
        "status": status,
        "model": model_name,
        "results": results,
        "warnings": warnings,
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2))

    markdown_lines = ["# Test Summary", f"**Status:** {status.upper()}" ]
    if results:
        markdown_lines.append(f"**Device:** {results.get('device')}")
        markdown_lines.append(f"**Elapsed:** {results.get('elapsed_ms')} ms")
        if "max_memory_mb" in results:
            markdown_lines.append(f"**Max Memory:** {results['max_memory_mb']} MB")
    if warnings:
        markdown_lines.append("## Warnings")
        markdown_lines.extend(f"- {w}" for w in warnings)
    MARKDOWN_PATH.write_text("\n".join(markdown_lines) + "\n")

    if status == "passed":
        print("✅ Sanity inference passed")
        return 0
    print("❌ Sanity inference failed")
    for warning in warnings:
        print(f"   → {warning}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
