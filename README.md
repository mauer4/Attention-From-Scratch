# Attention-From-Scratch

[Project microsite](https://mauer4.github.io/attention-from-scratch/)

## Project Vision

I'm mapping the landscape of large language model inference so an individual developer can run production-grade open weights. The goal is not to pretrain a new network, but to understand and rebuild the inference stack, from environment provisioning and weight inspection through custom CUDA kernels that rival modern serving engines. Olmo 2 provides the reference model family while the repository evolves toward an in-house attention engine.

## Repository Layout

- `src/`, `include/`, `python_bindings/`, `tests/`, `benchmarks/` - C++/CUDA engine code, bindings, test scaffolding, and performance harnesses.
- `inference/Olmo_2/` - experimental Hugging Face wrappers and GPU health checks retained for troubleshooting during the transition to the custom engine.
- `inference/From_Scratch/` - staging area for the bespoke attention implementation (currently a placeholder awaiting engine bring-up).
- `llm_raw/olmo_2/` - canonical storage for downloaded checkpoints, tokenizer assets, and upstream metadata.
- `llm_setup/analysis/` - safetensor inspection utilities (`get_tensor_shapes_form_safetensors.py`, `verify_tensor_extraction.py`, etc.) used during asset validation.
- `llm_original/olmo_2_repo/` - pristine clone of AllenAI's official OLMo repository, fetched via `scripts/fetch_olmo2_repo.sh`.
- `setup/` - light wrapper notes that point at the unified bootstrap script.
- `scripts/` - automation for environment setup (`bootstrap_host.sh`), asset download, and architecture/state-dict analysis.
- `docs/` - project planning (`PROJECT_PLAN.md`), provisioning guide (`ENVIRONMENT_AND_FLOWS.md`), and profiling notes (`OLMO2_BASELINE.md`).

## Environment & Inference Flows

The complete decision matrix and step-by-step instructions now live in
[`docs/ENVIRONMENT_AND_FLOWS.md`](docs/ENVIRONMENT_AND_FLOWS.md). Highlights:

- `bash setup/bootstrap_host.sh` handles system prerequisites, CUDA toolkit
  provisioning, and locked Python dependencies. Use `--help` for options such as
  `SKIP_SYSTEM_PACKAGES=1` on constrained hosts or `--with-cutlass` when you
  start working on custom kernels. The script creates (or reuses) the venv but
  leaves activation (`source <venv>/bin/activate`) to you after it finishes.
- Flow 1 — **AllenAI reference inference**  
  `make fetch-olmo` clones/updates `allenai/OLMo` and installs its inference
  stack. Pair it with `make download-assets` if you want to keep weights cached
  under `llm_raw/olmo_2/`, then use `make allenai-help` (or follow the upstream
  README) for CLI usage.
- Flow 2 — **Custom engine (in development)**  
  Shares the same environment but will execute code under `inference/From_Scratch/`
  once the kernels land. Until then, use Flow 1 for baselines and refer to
  `docs/PROJECT_PLAN.md` for roadmap status.

## Custom Engine Roadmap

The bespoke attention implementation is under active development:

- `src/`, `include/`, and `python_bindings/` will host the CUDA kernels, runtime graph, and pybind11 surface once Week 2 of the roadmap is complete.
- `inference/From_Scratch/` is reserved for the future driver script that exercises the custom engine.
- `docs/PROJECT_PLAN.md` lays out the 12-week schedule covering tokenizer support, FlashAttention-style kernels, paged KV caches, quantisation, scheduler features, and serving APIs.

Until the custom pipeline is checked in, use the baseline flows to gather performance targets and reference outputs.

## Analysis and Benchmarking Toolkit

- `scripts/analyse_architecture.py` summarises model geometry (console tables plus optional CSV).
- `scripts/dump_state_dict_summary.py` enumerates every tensor in the safetensors shards.
- `llm_setup/analysis/` scripts assist with manual tensor extraction and consistency checks.
- `docs/OLMO2_BASELINE.md` captures the Vast.AI workflow for profiling with Nsight tools.
- `benchmarks/` is the landing zone for throughput reports, Nsight traces, and comparison notes.

## Next Steps

1. Stand up the baseline and record key metrics (tokens/s, latency, memory) for prompts you care about.
2. Track implementation progress against `docs/PROJECT_PLAN.md`, using the analysis scripts to verify weight handling as kernels come online.
3. When custom kernels land, add regression tests under `tests/` and compare results with the established Hugging Face runner to ensure parity.
