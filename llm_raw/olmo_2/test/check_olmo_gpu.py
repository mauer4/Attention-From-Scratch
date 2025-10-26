#!/usr/bin/env python3
"""
Simple utility to load allenai/OLMo-2-1124-13B, move it to CUDA,
generate a short completion, and report GPU utilization.
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
    parser.add_argument(
        "--analysis",
        action="store_true",
        help="Collect additional latency, throughput, and memory metrics.",
    )
    parser.add_argument(
        "--nsight",
        action="store_true",
        help="Re-run the script under NVIDIA Nsight Systems for profiling (requires nsys).",
    )
    parser.add_argument(
        "--nsight-output",
        default="check_olmo_gpu_profile",
        help="Base name for the Nsight Systems output when --nsight is enabled.",
    )
    return parser.parse_args()


def check_cuda_kernel_support(device: str) -> None:
    """Ensure that the installed torch build has kernels for the active GPU."""
    if device != "cuda":
        return

    capability = torch.cuda.get_device_capability()
    device_name = torch.cuda.get_device_name()
    arch = f"sm_{capability[0]}{capability[1]}"
    compiled_arches = {
        arch_name.replace("+PTX", "") for arch_name in torch.cuda.get_arch_list()
    }
    print(f"Using {device_name} with compute capability {capability[0]}.{capability[1]} ({arch}).")
    if arch not in compiled_arches:
        raise SystemExit(
            (
                f"Torch {torch.__version__} lacks CUDA kernels for {arch}.\n"
                "Reinstall PyTorch with a build that targets your GPU, e.g.:\n"
                "  TORCH_INDEX_URL=https://download.pytorch.org/whl/nightly/cu124\n"
                "  pip install --upgrade --pre torch torchvision torchaudio --index-url \"$TORCH_INDEX_URL\"\n"
                "Or build PyTorch from source with the appropriate CUDA arch list."
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
    """Load the tokenizer, preferring any cached copy to avoid repeat downloads."""
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


def maybe_run_with_nsight(args: argparse.Namespace) -> None:
    """If Nsight profiling is requested, re-run the script under nsys."""
    if not args.nsight:
        return
    if os.environ.get("CHECK_OLMO_GPU_NSIGHT_ACTIVE") == "1":
        return

    nsys = shutil.which("nsys")
    if not nsys:
        print(
            "Requested Nsight profiling, but 'nsys' is not on PATH. "
            "Install it via cuda-nsight-systems-12-8 and re-run the setup script."
        )
        return

    output_base = args.nsight_output
    if output_base.endswith(".nsys-rep"):
        output_base = output_base[: -len(".nsys-rep")]
    final_report = f"{output_base}.nsys-rep"

    env = os.environ.copy()
    env["CHECK_OLMO_GPU_NSIGHT_ACTIVE"] = "1"

    filtered_args = []
    skip_next = False
    for token in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if token == "--nsight":
            continue
        if token.startswith("--nsight-output"):
            if token == "--nsight-output":
                skip_next = True
            continue
        filtered_args.append(token)

    filtered_args.append(f"--nsight-output={final_report}")

    cmd = [
        nsys,
        "profile",
        "--force-overwrite",
        "true",
        "-o",
        output_base,
        "--trace",
        "cuda,nvtx,cublas",
        sys.executable,
        str(Path(__file__).resolve()),
    ] + filtered_args

    print(f"Launching Nsight Systems profile to {final_report} ...")
    result = subprocess.run(cmd, env=env)
    raise SystemExit(result.returncode)


def main() -> None:
    args = parse_args()
    maybe_run_with_nsight(args)
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
    if args.analysis:
        torch.cuda.reset_peak_memory_stats(device)
        load_start_time = time.perf_counter()
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
    if args.analysis:
        load_elapsed = time.perf_counter() - load_start_time
        print(f"Model load time: {load_elapsed:.2f} s")
    model.to(device)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    print("Generating text...")
    if args.analysis:
        torch.cuda.reset_peak_memory_stats(device)
        generate_start_time = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=True,
        )
    if args.analysis:
        generate_elapsed = time.perf_counter() - generate_start_time

    completion = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("=== Generated Text ===")
    print(completion)

    torch.cuda.synchronize()
    memory_gib = torch.cuda.memory_allocated(device) / (1024**3)
    print(f"Allocated GPU memory: {memory_gib:.2f} GiB")
    if args.analysis:
        peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
        generated_tokens = outputs.shape[-1] - inputs.input_ids.shape[-1]
        tokens_per_sec = generated_tokens / generate_elapsed if generate_elapsed > 0 else float("nan")
        print(f"Peak allocated GPU memory: {peak_gib:.2f} GiB")
        print(f"Generation latency: {generate_elapsed:.2f} s")
        print(f"Throughput: {tokens_per_sec:.2f} tokens/s over {generated_tokens} tokens")

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
