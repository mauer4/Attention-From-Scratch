#!/usr/bin/env python3
"""
Analyse the OLMo 2 architecture from a Hugging Face snapshot and print a text
report covering dimensions, layer structure, and parameter counts. The script
does not modify any model weights; it only reads config files (and optionally
the safetensors index) to describe the layout.
"""

import argparse
import csv
import json
import os
from typing import Iterable, List, Sequence, Tuple


class Metadata:
    def __init__(self, config: dict):
        arch = config.get("architectures", [""])[0]
        if arch != "Olmo2ForCausalLM":
            raise ValueError(f"Expected Olmo2ForCausalLM architecture, found {arch!r}")

        self.arch = arch
        self.hidden_size = config["hidden_size"]
        self.intermediate_size = config["intermediate_size"]
        self.head_dim = self.hidden_size // config["num_attention_heads"]
        self.num_layers = config["num_hidden_layers"]
        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.vocab_size = config["vocab_size"]
        self.max_position_embeddings = config["max_position_embeddings"]
        self.bos_token_id = config.get("bos_token_id")
        self.eos_token_id = config.get("eos_token_id")
        self.pad_token_id = config.get("pad_token_id")
        self.rope_theta = config.get("rope_theta", 10000.0)
        self.partial_rotary_factor = config.get("partial_rotary_factor", 1)
        self.rotary_dim = int(self.head_dim * self.partial_rotary_factor)
        if self.rotary_dim > self.head_dim:
            raise ValueError("Rotary dimension exceeds head dimension")
        self.norm_eps = config["rms_norm_eps"]
        self.hidden_act = config["hidden_act"]
        self.tie_embeddings = config.get("tie_word_embeddings", False)


def render_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    widths = [len(h) for h in headers]
    materialized_rows: List[List[str]] = []
    for row in rows:
        str_row = [str(cell) for cell in row]
        materialized_rows.append(str_row)
        for idx, cell in enumerate(str_row):
            widths[idx] = max(widths[idx], len(cell))

    border = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"

    body_lines = [
        "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"
        for row in materialized_rows
    ]

    return "\n".join([border, header_line, border] + body_lines + [border])


def format_int(number: int) -> str:
    if number >= 1_000_000_000:
        return f"{number/1_000_000_000:.2f}B"
    if number >= 1_000_000:
        return f"{number/1_000_000:.2f}M"
    if number >= 1_000:
        return f"{number/1_000:.2f}K"
    return str(number)


def print_overview(metadata: Metadata):
    headers = ["Field", "Value"]
    rows = [
        ("Architecture", metadata.arch),
        ("Hidden size", metadata.hidden_size),
        ("Intermediate size", metadata.intermediate_size),
        ("Heads", metadata.num_heads),
        ("Key/Value heads", metadata.num_kv_heads),
        ("Head dim", metadata.head_dim),
        ("Rotary dim", metadata.rotary_dim),
        ("Hidden activation", metadata.hidden_act),
        ("Layers", metadata.num_layers),
        ("Vocabulary size", metadata.vocab_size),
        ("Max position embeddings", metadata.max_position_embeddings),
        ("Bos token id", metadata.bos_token_id),
        ("Eos token id", metadata.eos_token_id),
        ("Pad token id", metadata.pad_token_id),
        ("RMSNorm eps", metadata.norm_eps),
        ("Tie embeddings", metadata.tie_embeddings),
        ("RoPE theta", metadata.rope_theta),
    ]
    print("=== Architecture Overview ===")
    print(render_table(headers, rows))
    print()


def compute_parameter_counts(metadata: Metadata) -> dict:
    hidden = metadata.hidden_size
    ffn = metadata.intermediate_size
    vocab = metadata.vocab_size

    params = {}
    params["embeddings"] = hidden * vocab
    params["attention_proj"] = 4 * hidden * hidden  # q, k, v, o
    params["attention_norms"] = 2 * hidden  # q_norm + k_norm
    params["residual_norms"] = 2 * hidden  # post attention + post feedforward
    params["mlp"] = 3 * hidden * ffn  # gate, up, down
    per_layer = params["attention_proj"] + params["attention_norms"] + params["residual_norms"] + params["mlp"]
    params["per_layer_total"] = per_layer
    params["all_layers"] = per_layer * metadata.num_layers
    params["final_norm"] = hidden
    params["lm_head"] = 0 if metadata.tie_embeddings else hidden * vocab
    params["total"] = params["embeddings"] + params["all_layers"] + params["final_norm"] + params["lm_head"]
    return params


def print_parameter_summary(metadata: Metadata):
    counts = compute_parameter_counts(metadata)
    headers = ["Component", "Parameters (approx)"]
    rows = [
        ("Token embeddings", format_int(counts["embeddings"])),
        ("All transformer layers", format_int(counts["all_layers"])),
        ("  per layer", format_int(counts["per_layer_total"])),
        ("Final RMSNorm", format_int(counts["final_norm"])),
        ("LM head", format_int(counts["lm_head"])),
        ("Total", format_int(counts["total"])),
    ]
    print("=== Parameter Summary ===")
    print(render_table(headers, rows))
    print()


def build_tensor_rows(metadata: Metadata) -> Tuple[Sequence[str], List[Sequence[str]]]:
    hidden = metadata.hidden_size
    ffn = metadata.intermediate_size
    heads = metadata.num_heads
    head_dim = metadata.head_dim
    rotary = metadata.rotary_dim

    headers = ["Scope", "Tensor", "Shape / Params", "Details"]

    rows: List[Sequence[str]] = [
        (
            "Embeddings",
            "model.embed_tokens.weight",
            f"{metadata.vocab_size} x {hidden}",
            "Token embedding matrix",
        ),
        (
            "Rotary",
            "rope",
            f"rotary_dim={rotary}",
            f"theta={metadata.rope_theta}, heads={heads}, head_dim={head_dim}",
        ),
    ]

    for layer in range(metadata.num_layers):
        scope = f"Layer {layer}"
        rows.extend(
            [
                (scope, "self_attn.q_norm.weight", f"{hidden}", "Query RMSNorm scale"),
                (scope, "self_attn.k_norm.weight", f"{hidden}", "Key RMSNorm scale"),
                (
                    scope,
                    "self_attn.q_proj.weight",
                    f"{hidden} x {hidden}",
                    f"Projects hidden -> heads ({heads}x{head_dim}), rotary_dim={rotary}",
                ),
                (
                    scope,
                    "self_attn.k_proj.weight",
                    f"{hidden} x {hidden}",
                    f"KV projection, heads={metadata.num_kv_heads}",
                ),
                (
                    scope,
                    "self_attn.v_proj.weight",
                    f"{hidden} x {hidden}",
                    "Value projection",
                ),
                (
                    scope,
                    "self_attn.o_proj.weight",
                    f"{hidden} x {hidden}",
                    "Output projection (concats heads)",
                ),
                (
                    scope,
                    "post_attention_layernorm.weight",
                    f"{hidden}",
                    "RMSNorm after attention residual",
                ),
                (
                    scope,
                    "mlp.gate_proj.weight",
                    f"{ffn} x {hidden}",
                    f"FFN gate ({metadata.hidden_act})",
                ),
                (
                    scope,
                    "mlp.up_proj.weight",
                    f"{ffn} x {hidden}",
                    "FFN up projection",
                ),
                (
                    scope,
                    "mlp.down_proj.weight",
                    f"{hidden} x {ffn}",
                    "FFN down projection",
                ),
                (
                    scope,
                    "post_feedforward_layernorm.weight",
                    f"{hidden}",
                    "RMSNorm after FFN residual",
                ),
            ]
        )

    rows.append(
        (
            "Output",
            "model.norm.weight",
            f"{hidden}",
            "Final RMSNorm before logits",
        )
    )

    if metadata.tie_embeddings:
        rows.append(
            (
                "Output",
                "lm_head.weight",
                f"{metadata.vocab_size} x {hidden}",
                "Tied with embeddings",
            )
        )
    else:
        rows.append(
            (
                "Output",
                "lm_head.weight",
                f"{metadata.vocab_size} x {hidden}",
                "Logit projection",
            )
        )

    return headers, rows


def print_tensor_table(headers: Sequence[str], rows: Iterable[Sequence[str]]):
    print("=== Tensor Summary ===")
    print(render_table(headers, rows))
    print()


def print_layer_graph(metadata: Metadata):
    hidden = metadata.hidden_size
    heads = metadata.num_heads
    head_dim = metadata.head_dim
    ffn = metadata.intermediate_size
    rotary = metadata.rotary_dim

    print("=== Layer Graph ===")
    print("Embeddings")
    print(f"  +-- model.embed_tokens.weight [{metadata.vocab_size} x {hidden}]")
    print(f"  +-- RoPE rotary_dim={rotary}, theta={metadata.rope_theta}")
    for layer in range(metadata.num_layers):
        print(f"Layer {layer}")
        print(f"  +-- SelfAttention (heads={heads}, head_dim={head_dim}, rotary_dim={rotary})")
        print(f"  |   +-- q_norm.weight [{hidden}]")
        print(f"  |   +-- k_norm.weight [{hidden}]")
        print(f"  |   +-- q_proj.weight [{hidden} x {hidden}]")
        print(f"  |   +-- k_proj.weight [{hidden} x {hidden}]")
        print(f"  |   +-- v_proj.weight [{hidden} x {hidden}]")
        print(f"  |   +-- o_proj.weight [{hidden} x {hidden}]")
        print(f"  +-- post_attention_layernorm.weight [{hidden}]")
        print(f"  +-- MLP (activation={metadata.hidden_act}, intermediate={ffn})")
        print(f"  |   +-- gate_proj.weight [{ffn} x {hidden}]")
        print(f"  |   +-- up_proj.weight   [{ffn} x {hidden}]")
        print(f"  |   +-- down_proj.weight [{hidden} x {ffn}]")
        print(f"  +-- post_feedforward_layernorm.weight [{hidden}]")
    print("Output")
    print(f"  +-- model.norm.weight [{hidden}]")
    if metadata.tie_embeddings:
        print("  +-- lm_head.weight shares parameters with embeddings")
    else:
        print(f"  +-- lm_head.weight [{metadata.vocab_size} x {hidden}]")
    print()


def main():
    parser = argparse.ArgumentParser(description="Print OLMo 2 architecture information from a HF snapshot.")
    parser.add_argument("input", type=str, help="Path to the Hugging Face snapshot directory")
    parser.add_argument(
        "--format",
        choices=["table", "graph", "both"],
        default="both",
        help="Select which report sections to render",
    )
    parser.add_argument(
        "--layer-table-csv",
        type=str,
        help="If set, also write the tensor summary table to this CSV file",
    )
    args = parser.parse_args()

    config_path = os.path.join(args.input, "config.json")
    if not os.path.exists(config_path):
        parser.error(f"config.json not found in {args.input}")

    with open(config_path, "r") as f:
        config = json.load(f)

    metadata = Metadata(config)

    print_overview(metadata)
    print_parameter_summary(metadata)
    tensor_headers, tensor_rows = build_tensor_rows(metadata)
    if args.format in ("table", "both"):
        print_tensor_table(tensor_headers, tensor_rows)
    if args.format in ("graph", "both"):
        print_layer_graph(metadata)
    if args.layer_table_csv:
        with open(args.layer_table_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(tensor_headers)
            writer.writerows(tensor_rows)
        print(f"Wrote tensor summary table to {args.layer_table_csv}")


if __name__ == "__main__":
    main()
