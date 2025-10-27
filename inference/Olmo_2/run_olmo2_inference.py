#!/usr/bin/env python3
"""Generate text from the staged OLMo 2 snapshot via AllenAI's inference helpers."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Tuple

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - surfaced at runtime
    torch = None  # type: ignore[assignment]
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SNAPSHOT = ROOT / "llm_raw" / "olmo_2" / "raw_weights"
DEFAULT_TOKENIZER = ROOT / "llm_raw" / "olmo_2" / "raw_tokenizer"
DEFAULT_METADATA = ROOT / "llm_raw" / "olmo_2" / "metadata"
DEFAULT_ENGINE_ROOT = ROOT / "llm_original" / "olmo_2_repo"


def _infer_snapshot_dtype(metadata_dir: Path) -> torch.dtype | None:
    """Return the torch dtype recorded in the snapshot config, if any."""
    if torch is None:
        return None

    config_path = metadata_dir / "config.json"
    if not config_path.exists():
        return None

    try:
        config = json.loads(config_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    dtype_str = config.get("torch_dtype")
    if not isinstance(dtype_str, str):
        return None

    torch_attr = dtype_str.strip().lower().replace(" ", "")
    dtype = getattr(torch, torch_attr, None)
    return dtype if isinstance(dtype, torch.dtype) else None


def _import_eval_utils(engine_root: Path):
    """Import AllenAI's `inference.eval.utils` module as a standalone dependency."""
    inference_dir = engine_root / "inference"

    if not engine_root.exists():
        raise SystemExit(
            f"AllenAI OLMo repository not found at {engine_root}. "
            "Run scripts/fetch_olmo2_repo.sh to clone and install it."
        )
    if not inference_dir.exists():
        raise SystemExit(
            f"{inference_dir} is missing. Ensure scripts/fetch_olmo2_repo.sh completed successfully."
        )

    # Ensure the upstream package path has priority so its relative imports resolve.
    for path in (engine_root, inference_dir):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    try:
        import eval.utils as eval_utils  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - handled at runtime
        missing = exc.name
        if missing == "openai":
            import importlib.machinery
            import types
            import warnings

            warnings.warn(
                "The optional 'openai' package is not installed. "
                "Skipping OpenAI API helpers; local OLMo inference remains available.",
                RuntimeWarning,
                stacklevel=2,
            )

            def _missing_openai(*_args, **_kwargs):
                raise ModuleNotFoundError(
                    "The 'openai' package is required for OpenAI API integrations. "
                    "Install it with `pip install openai` if you need those features."
                )

            stub = types.ModuleType("openai")
            stub.__spec__ = importlib.machinery.ModuleSpec("openai", loader=None)  # type: ignore[attr-defined]
            stub.__path__ = []  # type: ignore[attr-defined]
            stub.__package__ = "openai"
            stub.__file__ = "<openai-shim>"  # type: ignore[attr-defined]
            stub.api_key = None  # type: ignore[attr-defined]
            stub.error = types.SimpleNamespace(OpenAIError=ModuleNotFoundError)  # type: ignore[attr-defined]
            stub.ChatCompletion = types.SimpleNamespace(acreate=_missing_openai, create=_missing_openai)
            stub.Completion = types.SimpleNamespace(acreate=_missing_openai, create=_missing_openai)

            def _stub_getattr(_name):
                if _name.startswith("__"):
                    raise AttributeError(f"module 'openai' has no attribute {_name!r}")
                return _missing_openai

            stub.__getattr__ = _stub_getattr  # type: ignore[attr-defined]
            sys.modules.setdefault("openai", stub)

            try:
                import eval.utils as eval_utils  # type: ignore
            except ModuleNotFoundError as secondary:  # pragma: no cover - handled at runtime
                secondary_missing = secondary.name
                if secondary_missing == "openai":
                    raise SystemExit(
                        "Failed to initialize OpenAI API shims. Install the 'openai' package to continue."
                    ) from secondary
                if secondary_missing == "torch":
                    raise SystemExit(
                        "PyTorch is missing. Activate the `.venv-olmo2` environment created by scripts/setup_olmo2_env.sh."
                    ) from secondary
                raise SystemExit(
                    f"Dependency {secondary_missing!r} is unavailable. "
                    "Re-run scripts/fetch_olmo2_repo.sh to finish installing upstream requirements."
                ) from secondary
        elif missing == "torch":
            hint = "PyTorch is missing. Activate the `.venv-olmo2` environment created by scripts/setup_olmo2_env.sh."
            raise SystemExit(hint) from exc
        else:
            hint = (
                f"Dependency {missing!r} is unavailable. "
                "Re-run scripts/fetch_olmo2_repo.sh to finish installing upstream requirements."
            )
            raise SystemExit(hint) from exc

    required_attrs = ("generate_completions",)
    if not all(hasattr(eval_utils, attr) for attr in required_attrs):
        raise SystemExit(
            "AllenAI inference utilities are missing expected helpers. "
            "Verify that llm_original/olmo_2_repo is up to date."
        )

    return eval_utils


def _resolve_device(preference: str) -> Tuple[str, str | None, bool]:
    """Return (device_label, device_map, can_use_half_precision)."""
    assert torch is not None  # guarded in main
    if preference == "cuda":
        if torch.cuda.is_available():
            return "cuda", "auto", True
        return "cpu", None, False
    if preference == "cpu":
        return "cpu", None, False
    # Auto: prefer CUDA when present.
    if torch.cuda.is_available():
        return "cuda", "auto", True
    return "cpu", None, False


def _prepare_snapshot_view(weights_dir: Path, metadata_dir: Path) -> Tuple[tempfile.TemporaryDirectory, Path]:
    """Create a temporary HF-style snapshot directory composed of the staged assets."""
    temp_dir = tempfile.TemporaryDirectory()
    view_path = Path(temp_dir.name)

    def link(src: Path) -> None:
        if not src.exists() or not src.is_file():
            return
        dest = view_path / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest.unlink()
        try:
            os.symlink(src, dest)
        except OSError as exc:
            raise SystemExit(
                f"Failed to create symlink for {src} -> {dest}. "
                "Ensure the filesystem supports symlinks (consider running with elevated permissions)."
            ) from exc

    for file_path in weights_dir.glob("*"):
        link(file_path)

    for metadata_name in ("config.json", "generation_config.json", "model.safetensors.index.json"):
        link(metadata_dir / metadata_name)

    return temp_dir, view_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Olmo 2 inference against a local snapshot.")
    parser.add_argument(
        "--snapshot-path",
        type=Path,
        default=DEFAULT_SNAPSHOT,
        help="Directory containing the OLMo safetensors shards (default: llm_raw/olmo_2/raw_weights).",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help=(
            "Directory holding tokenizer JSON/merges (default: llm_raw/olmo_2/raw_tokenizer; falls back to the snapshot)."
        ),
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=DEFAULT_METADATA,
        help="Directory containing model metadata such as config.json and safetensor index (default: llm_raw/olmo_2/metadata).",
    )
    parser.add_argument(
        "--engine-root",
        type=Path,
        default=DEFAULT_ENGINE_ROOT,
        help="Path to the cloned AllenAI OLMo repository (default: llm_original/olmo_2_repo).",
    )
    parser.add_argument(
        "--prompt",
        default="Explain why attention mechanisms improved transformer models.",
        help="Prompt text to feed into the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Number of tokens to generate beyond the prompt.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device preference (auto detects CUDA when available).",
    )
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Keep the model in full precision even on CUDA devices.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional file to write the prompt+completion text.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for inference when running on CUDA.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    if torch is None:
        raise SystemExit(
            "PyTorch is required. Activate the `.venv-olmo2` environment created by scripts/setup_olmo2_env.sh."
        )

    eval_utils = _import_eval_utils(args.engine_root)

    snapshot_path = args.snapshot_path.expanduser().resolve()
    if not snapshot_path.exists():
        raise SystemExit(
            f"Snapshot directory {snapshot_path} not found. "
            "Run scripts/download_olmo2_assets.py to stage the weights."
        )

    tokenizer_path = args.tokenizer_path.expanduser().resolve() if args.tokenizer_path else DEFAULT_TOKENIZER
    if not tokenizer_path.exists():
        console.print(
            f"[yellow]Tokenizer assets not found at {tokenizer_path}; falling back to {snapshot_path}.[/yellow]"
        )
        tokenizer_path = snapshot_path

    metadata_path = args.metadata_path.expanduser().resolve()
    if not metadata_path.exists():
        raise SystemExit(
            f"Metadata directory {metadata_path} not found. "
            "Run scripts/download_olmo2_assets.py to stage the configs."
        )

    snapshot_view_obj: tempfile.TemporaryDirectory | None = None
    snapshot_source = snapshot_path
    if not (snapshot_path / "config.json").exists():
        snapshot_view_obj, snapshot_source = _prepare_snapshot_view(snapshot_path, metadata_path)
        console.print(f"Created temporary snapshot view at {snapshot_source} for HF-compatible loading.")

    device_label, _, can_half = _resolve_device(args.device)

    snapshot_dtype = _infer_snapshot_dtype(metadata_path)
    if args.full_precision:
        target_dtype = torch.float32 if torch is not None else None
    else:
        target_dtype = snapshot_dtype
        if target_dtype is None and can_half and torch is not None:
            target_dtype = torch.float16

    if target_dtype is not None:
        console.print(f"Target dtype: {target_dtype}")

    if args.device == "cuda" and device_label != "cuda":
        console.print(
            "[yellow]Requested CUDA but none detected. Falling back to CPU (this will be slow).[/yellow]"
        )

    console.print(
        f"Loading OLMo 2 snapshot from {snapshot_source} "
        f"using Hugging Face transformers (dtype={target_dtype or 'fp32'})..."
    )

    load_start = time.perf_counter()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path),
            use_fast=True,
            local_files_only=True,
        )
    except OSError as exc:
        raise SystemExit(
            f"Failed to load tokenizer assets from {tokenizer_path}: {exc}. "
            "Re-run scripts/download_olmo2_assets.py to ensure tokenizer files are staged."
        ) from exc

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model_load_kwargs = {"device_map": None, "local_files_only": True}
    try:
        if target_dtype is not None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    str(snapshot_source),
                    dtype=target_dtype,
                    **model_load_kwargs,
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    str(snapshot_source),
                    torch_dtype=target_dtype,
                    **model_load_kwargs,
                )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                str(snapshot_source),
                **model_load_kwargs,
            )
    except OSError as exc:
        raise SystemExit(
            f"Failed to load model weights from {snapshot_source}: {exc}. "
            "Run scripts/download_olmo2_assets.py to ensure snapshot shards are present."
        ) from exc

    load_duration = time.perf_counter() - load_start
    console.print(f"Model and tokenizer ready in {load_duration:.2f}s.")

    if device_label == "cuda":
        if torch is None or not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")
        torch.cuda.empty_cache()
        model = model.to(device="cuda")
    else:
        model = model.to(device=device_label)

    model.eval()

    if args.compile and device_label == "cuda" and hasattr(torch, "compile"):
        console.print("Compiling model for optimized inference...")
        model = torch.compile(model)  # type: ignore[arg-type]

    console.print("Generating completion...")
    start = time.perf_counter()
    generations = eval_utils.generate_completions(
        model,
        tokenizer,
        [args.prompt],
        batch_size=1,
        disable_tqdm=True,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    duration = time.perf_counter() - start

    completion = generations[0]
    full_text = f"{args.prompt}{completion}"

    prompt_tokens = tokenizer(
        args.prompt, add_special_tokens=False, return_tensors="pt"
    ).input_ids.shape[-1]
    output_tokens = tokenizer(
        full_text, add_special_tokens=False, return_tensors="pt"
    ).input_ids.shape[-1]
    tokens_generated = max(output_tokens - prompt_tokens, 0)
    tokens_per_sec = tokens_generated / duration if duration > 0 else float("nan")

    console.rule("Generation Result")
    console.print(full_text)
    console.rule()
    console.print(
        f"Generated {tokens_generated} tokens in {duration:.2f}s "
        f"({tokens_per_sec:.2f} tok/s)."
    )

    if args.output_path:
        args.output_path.write_text(full_text)
        console.print(f"Wrote output to {args.output_path}")

    if snapshot_view_obj is not None:
        snapshot_view_obj.cleanup()


if __name__ == "__main__":
    main()
