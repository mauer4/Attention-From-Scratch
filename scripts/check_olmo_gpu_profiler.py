#!/usr/bin/env python3
"""
Like `check_olmo_gpu.py` but wraps the generation call with
torch.profiler to produce a chrome trace (`trace.json`) and a
small summary table sorted by CUDA time.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, ProfilerActivity


def pick_dtype() -> torch.dtype:
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate text with an OLMo checkpoint and profile the generation step."
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
    parser.add_argument(
        "--prof-trace",
        default="trace.json",
        help="Filename for the chrome trace output (default: %(default)s).",
    )
    return parser.parse_args()


def check_cuda_kernel_support(device: str) -> None:
    if device != "cuda":
        return

    capability = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name()
    arch = f"sm_{capability[0]}{capability[1]}"
    compiled_arches = {arch_name.replace("+PTX", "") for arch_name in torch.cuda.get_arch_list()}
    print(f"Using {device_name} with compute capability {capability[0]}.{capability[1]} ({arch}).")
    if arch not in compiled_arches:
        raise SystemExit(
            (
                f"Torch {torch.__version__} lacks CUDA kernels for {arch}.\n"
                "Reinstall PyTorch with a build that targets your GPU, e.g.:\n"
                "  TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu124\n"
                "  pip install --upgrade --pre torch torchvision torchaudio --index-url \"$TORCH_INDEX_URL\"\n"
            )
        )

    try:
        torch.randn(1, device=device)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise SystemExit(
            "A simple CUDA kernel launch failed. Double-check driver and PyTorch compatibility.\n"
            f"Original error: {exc}"
        ) from exc


def load_tokenizer(model_id: str, load_kwargs: dict) -> AutoTokenizer:
    local_kwargs = dict(load_kwargs)
    local_kwargs["local_files_only"] = True
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, **local_kwargs)
        print("Loaded tokenizer from local cache.")
        return tokenizer
    except OSError:
        pass

    print("Tokenizer not found in cache; downloading from Hugging Face Hub...")
    return AutoTokenizer.from_pretrained(model_id, **load_kwargs)


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

    check_cuda_kernel_support(device)

    dtype = pick_dtype()
    print(f"Loading {model_id} with dtype={dtype}...")
    try:
        load_kwargs = {}
        if hf_token:
            load_kwargs["token"] = hf_token
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir
        tokenizer = load_tokenizer(model_id, load_kwargs)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=None,
                **load_kwargs,
            )
        except TypeError:
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
    print("Generating text (profiled)...")

    # Profile the generation step and export a chrome trace.
    with torch.inference_mode():
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_stack=True,
            profile_memory=True,
        ) as prof:
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True,
            )

    # Ensure CUDA work is finished before summarizing/exporting the profile.
    torch.cuda.synchronize()

    # Print a small summary table and write a chrome trace for deeper analysis.
    try:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        prof.export_chrome_trace(args.prof_trace)
        print(f"Chrome trace exported to {args.prof_trace}")
    except Exception as exc:  # pragma: no cover - profiling output path
        print(f"Profiler summary/export failed: {exc}")

    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== Generated Text ===")
    print(completion)

    torch.cuda.synchronize()
    memory_gib = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"Allocated GPU memory: {memory_gib:.2f} GiB")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        print("nvidia-smi snapshot:")
        try:
            subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                    "--format=csv",
                ],
                check=False,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            print(f"nvidia-smi command failed: {exc}")
    else:
        print("nvidia-smi not found; skipping driver-level utilization check.")


if __name__ == "__main__":
    main()
