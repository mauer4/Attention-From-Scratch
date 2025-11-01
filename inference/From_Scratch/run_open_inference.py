#!/usr/bin/env python3
"""Quick wrapper around Hugging Face transformers for sanity checks."""

from __future__ import annotations

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short generation with transformers.")
    parser.add_argument("--model", default="sshleifer/tiny-gpt2", help="Model identifier.")
    parser.add_argument("--prompt", default="Hello world", help="Prompt to feed the model.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device)
    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    output = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
