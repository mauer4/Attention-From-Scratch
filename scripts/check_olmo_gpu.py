#!/usr/bin/env python3
"""
Simple utility to load allenai/OLMo-2-1124-13B, move it to CUDA,
generate a short completion, and report GPU utilization.
"""

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_dtype() -> torch.dtype:
    """Prefer bfloat16 when the GPU supports it to reduce memory pressure."""
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with an OLMo checkpoint and report GPU utilization."
    )
    parser.add_argument(
        "--model-id",
        default="allenai/OLMo-2-1124-13B",
        help="Hugging Face model identifier (default: %(default)s).",
    )
    parser.add_argument(
        "--prompt",
        default=(
            "Seattle's weather inspires OLMo to write a poem about "
            "machine learning breakthroughs:\n"
        ),
        help="Prompt text for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Number of tokens to sample beyond the prompt (default: %(default)s).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature (default: %(default)s).",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Explicit Hugging Face token. Defaults to the HF_TOKEN environment variable if unset.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Custom directory for model/tokenizer cache (ensure it has ample free space).",
    )
    parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Override the temp directory used during downloads (sets TMPDIR/TMP/TEMP).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_id = args.model_id
    device = "cuda"
    prompt = args.prompt
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    cache_dir = args.cache_dir
    tmp_dir = args.tmp_dir

    if cache_dir:
        cache_path = Path(cache_dir).expanduser()
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_dir = str(cache_path)
    if tmp_dir:
        tmp_path = Path(tmp_dir).expanduser()
        tmp_path.mkdir(parents=True, exist_ok=True)
        tmp_dir = str(tmp_path)
        os.environ["TMPDIR"] = tmp_dir
        os.environ["TMP"] = tmp_dir
        os.environ["TEMP"] = tmp_dir

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; make sure you are on a GPU instance.")

    dtype = pick_dtype()
    print(f"Loading {model_id} with dtype={dtype}...")
    try:
        load_kwargs = {}
        if hf_token:
            load_kwargs["token"] = hf_token
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        tokenizer = AutoTokenizer.from_pretrained(model_id, **load_kwargs)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=None,
                **load_kwargs,
            )
        except TypeError:
            # Fallback for older transformers versions that still expect torch_dtype.
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=None,
                **load_kwargs,
            )
    except OSError as exc:
        hint = (
            "Confirm the model ID is correct (e.g. `allenai/OLMo-2-1124-13B`) "
            "and that your Hugging Face credentials grant access.\n"
            "Use `huggingface-cli login` or set `HF_TOKEN` before running this script.\n"
            "If you are hitting disk or tmp issues, pass `--cache-dir` and/or `--tmp-dir` "
            "to point at locations with enough free space."
        )
        raise SystemExit(f"Failed to load model: {exc}\n{hint}") from exc
    except RuntimeError as exc:
        hint = (
            "If you encountered an out-of-disk error, point the cache to a larger volume "
            "using `--cache-dir /path/with/space` or by setting `HF_HOME`/`TRANSFORMERS_CACHE`."
        )
        raise SystemExit(f"Failed to load model: {exc}\n{hint}") from exc
    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print("Generating text...")
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
        )

    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== Generated Text ===")
    print(completion)

    torch.cuda.synchronize()
    memory_gib = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"Allocated GPU memory: {memory_gib:.2f} GiB")

    if hasattr(torch.cuda, "utilization_rate"):
        utilization = torch.cuda.utilization_rate(device)
        print(f"Torch reported GPU utilization: {utilization}%")
    else:
        print("torch.cuda.utilization_rate is unavailable in this PyTorch build.")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        print("nvidia-smi snapshot:")
        try:
            subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"nvidia-smi command failed: {exc}")
    else:
        print("nvidia-smi not found; skipping driver-level utilization check.")


if __name__ == "__main__":
    main()

