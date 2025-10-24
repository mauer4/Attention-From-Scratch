#!/usr/bin/env python3
"""
Conversion script tailored for OLMo 2 checkpoints hosted on Hugging Face.

The script mirrors the overall flow of convert.py but handles the extra
normalization layers that OLMo 2 introduces (query/key norms and a second
residual norm after the feed-forward block).
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from safetensors.torch import safe_open, save_file

SUPPORTED_ARCHITECTURES = ["Olmo2ForCausalLM"]
SUPPORTED_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "fp8": torch.float8_e5m2}


class Metadata:
    def __init__(self, config, dtype):
        arch = config["architectures"][0]
        if arch not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture {arch}. Expected one of {SUPPORTED_ARCHITECTURES}")
        if dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype {dtype}. Expected one of {list(SUPPORTED_DTYPES)}")

        self.arch = arch
        self.dtype = dtype
        self.dim = config["hidden_size"]
        self.hidden_dim = config["intermediate_size"]
        self.head_dim = config["hidden_size"] // config["num_attention_heads"]
        self.n_layers = config["num_hidden_layers"]
        self.n_heads = config["num_attention_heads"]
        self.n_kv_heads = config["num_key_value_heads"]
        self.vocab_size = config["vocab_size"]
        self.max_seq_len = config["max_position_embeddings"]
        self.bos_token_id = config.get("bos_token_id")
        self.eos_token_id = config.get("eos_token_id")
        self.pad_token_id = config.get("pad_token_id")
        self.rope_theta = config.get("rope_theta", 10000.0)
        self.rotary_dim = int(self.head_dim * config.get("partial_rotary_factor", 1))
        if self.rotary_dim > self.head_dim:
            raise ValueError("rotary dimension exceeds head dimension")
        self.norm_eps = config["rms_norm_eps"]
        self.norm_type = "rmsnorm"
        self.act_type = config["hidden_act"]

        if config.get("attention_bias", False):
            raise ValueError("Attention bias is not supported")
        if config.get("mlp_bias", False):
            raise ValueError("MLP bias is not supported")

    def to_dict(self):
        meta = {
            "arch": self.arch,
            "dtype": self.dtype,
            "dim": str(self.dim),
            "hidden_dim": str(self.hidden_dim),
            "head_dim": str(self.head_dim),
            "n_layers": str(self.n_layers),
            "n_heads": str(self.n_heads),
            "n_kv_heads": str(self.n_kv_heads),
            "vocab_size": str(self.vocab_size),
            "max_seq_len": str(self.max_seq_len),
            "rope_theta": str(self.rope_theta),
            "rotary_dim": str(self.rotary_dim),
            "norm_eps": str(self.norm_eps),
            "norm_type": str(self.norm_type),
            "act_type": str(self.act_type),
        }
        if self.bos_token_id is not None:
            meta["bos_token_id"] = str(self.bos_token_id)
        if self.eos_token_id is not None:
            meta["eos_token_id"] = str(self.eos_token_id)
        if self.pad_token_id is not None:
            meta["pad_token_id"] = str(self.pad_token_id)
        return meta


def gpt2_bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def load_tokens(tokenizer_path, vocab_size):
    tokens = [""] * vocab_size
    with open(tokenizer_path, "r") as f:
        tokenizer = json.load(f)

    use_gpt2_byte_preprocessing = not tokenizer["model"].get("byte_fallback", False)
    vocab = tokenizer["model"]["vocab"]
    assert len(vocab) <= vocab_size

    for token, idx in vocab.items():
        tokens[idx] = token

    for added in tokenizer.get("added_tokens", []):
        tokens[added["id"]] = added["content"]

    gpt2_decode = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    for i, token in enumerate(tokens):
        if use_gpt2_byte_preprocessing:
            byte_values = bytes([gpt2_decode.get(c, 0) for c in token])
        else:
            token = token.replace("\u2581", " ")
            byte_values = token.encode("utf-8")
        byte_values = byte_values.replace(b"\0", b"\7")
        assert byte_values.count(0) == 0
        tokens[i] = byte_values

    return tokens


def load_weight_map(index_path):
    with open(index_path, "r") as f:
        index = json.load(f)
    return index["weight_map"]


def load_sharded_weights(model_dir, weight_map):
    grouped = defaultdict(list)
    for weight_name, shard_filename in weight_map.items():
        grouped[shard_filename].append(weight_name)

    weights = {}
    for shard_filename, key_list in grouped.items():
        shard_path = os.path.join(model_dir, shard_filename)
        with safe_open(shard_path, framework="pt") as shard:
            for name in key_list:
                weights[name] = shard.get_tensor(name)
    return weights


def permute_reverse(w, heads, rotary_dim):
    head_dim = w.shape[0] // heads
    assert rotary_dim <= head_dim
    w = torch.unflatten(w, 0, (-1, head_dim))
    rotary = w[:, :rotary_dim]
    remainder = w[:, rotary_dim:]
    rotary = torch.unflatten(rotary, 1, (2, -1))
    rotary = rotary.transpose(1, 2)
    rotary = rotary.flatten(1, 2)
    w = torch.cat([rotary, remainder], dim=1)
    return torch.flatten(w, 0, 1)


def convert_weights(model_dir, dtype_str, metadata, tie_word_embeddings):
    dtype = SUPPORTED_DTYPES[dtype_str]
    weight_map = load_weight_map(os.path.join(model_dir, "model.safetensors.index.json"))
    weights = load_sharded_weights(model_dir, weight_map)

    progress = 0

    def conv(tensor):
        nonlocal progress
        progress += 1
        print(f"\rConverting tensor {progress}: {tuple(tensor.shape)}", end="", flush=True)
        return tensor.to(dtype)

    tensors = {}
    tensors["model.embed.weight"] = conv(weights["model.embed_tokens.weight"])

    for layer in range(metadata.n_layers):
        base = f"model.layers.{layer}"
        attn = f"{base}.attn"
        mlp = f"{base}.mlp"

        tensors[f"{attn}.q_norm.weight"] = weights[f"{base}.self_attn.q_norm.weight"].float()
        tensors[f"{attn}.k_norm.weight"] = weights[f"{base}.self_attn.k_norm.weight"].float()

        rotary_dim = metadata.rotary_dim
        tensors[f"{attn}.wq.weight"] = conv(permute_reverse(weights[f"{base}.self_attn.q_proj.weight"], metadata.n_heads, rotary_dim))
        tensors[f"{attn}.wk.weight"] = conv(permute_reverse(weights[f"{base}.self_attn.k_proj.weight"], metadata.n_kv_heads, rotary_dim))
        tensors[f"{attn}.wv.weight"] = conv(weights[f"{base}.self_attn.v_proj.weight"])
        tensors[f"{attn}.wo.weight"] = conv(weights[f"{base}.self_attn.o_proj.weight"])

        tensors[f"{attn}.post_norm.weight"] = weights[f"{base}.post_attention_layernorm.weight"].float()
        tensors[f"{mlp}.post_norm.weight"] = weights[f"{base}.post_feedforward_layernorm.weight"].float()

        tensors[f"{mlp}.w1.weight"] = conv(weights[f"{base}.mlp.gate_proj.weight"])
        tensors[f"{mlp}.w2.weight"] = conv(weights[f"{base}.mlp.down_proj.weight"])
        tensors[f"{mlp}.w3.weight"] = conv(weights[f"{base}.mlp.up_proj.weight"])

    tensors["model.norm.weight"] = weights["model.norm.weight"].float()
    if not tie_word_embeddings:
        tensors["model.output.weight"] = conv(weights["lm_head.weight"])

    print()
    return tensors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=str, help="Destination .safetensors file")
    parser.add_argument("input", type=str, help="Directory containing the OLMo 2 Hugging Face snapshot")
    parser.add_argument("--dtype", type=str, default="fp16", choices=SUPPORTED_DTYPES.keys())
    args = parser.parse_args()

    config_path = os.path.join(args.input, "config.json")
    tokenizer_path = os.path.join(args.input, "tokenizer.json")
    index_path = os.path.join(args.input, "model.safetensors.index.json")

    for path in (config_path, tokenizer_path, index_path):
        if not os.path.exists(path):
            parser.error(f"Missing required file: {path}")

    with open(config_path, "r") as f:
        config = json.load(f)
    metadata = Metadata(config, args.dtype)

    tokens = load_tokens(tokenizer_path, metadata.vocab_size)
    tensors = convert_weights(args.input, args.dtype, metadata, config.get("tie_word_embeddings", False))

    tokenizer_tensor = torch.cat(
        [torch.tensor([byte for byte in token] + [0], dtype=torch.uint8) for token in tokens]
    )
    tensors["tokenizer.tokens"] = tokenizer_tensor

    print(f"Saving {len(tensors)} tensors...")
    save_file(tensors, args.output, metadata.to_dict())


if __name__ == "__main__":
    main()
