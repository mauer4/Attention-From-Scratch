#!/usr/bin/env python3
"""
Generate text with an Olmo 2 checkpoint on Vast.AI.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Olmo 2 inference locally.")
    parser.add_argument(
        "--model",
        default="allenai/OLMo-2-7B",
        help="Hugging Face model ID to load.",
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
        "--output-path",
        type=Path,
        default=None,
        help="Optional file to write the generated text.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for inference (PyTorch 2.0+).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    console = Console()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    console.print(f"Loading model {args.model!r} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
    )

    if args.compile and hasattr(torch, "compile"):
        console.print("Compiling model for optimized inference...")
        model = torch.compile(model)  # type: ignore[arg-type]

    inputs = tokenizer(args.prompt, return_tensors="pt").to(model.device)

    console.print("Generating...")
    start = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
    duration = time.perf_counter() - start
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    tokens_generated = output.shape[-1] - inputs["input_ids"].shape[-1]
    tokens_per_sec = tokens_generated / duration if duration > 0 else float("nan")

    console.rule("Generation Result")
    console.print(generated_text)
    console.rule()
    console.print(
        f"Generated {tokens_generated} tokens in {duration:.2f}s "
        f"({tokens_per_sec:.2f} tok/s)."
    )

    if args.output_path:
        args.output_path.write_text(generated_text)
        console.print(f"Wrote output to {args.output_path}")


if __name__ == "__main__":
    main()
