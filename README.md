# Attention-From-Scratch

[Project microsite](https://mauer4.github.io/attention-from-scratch/)

## Project Vision

I'm mapping the landscape of large language model inference so an individual developer can run production-grade open weights. The goal is not to pretrain a new network, but to understand and rebuild the inference stack, from environment provisioning and weight inspection through custom CUDA kernels that rival modern serving engines. Olmo 2 provides the reference model family while the repository evolves toward an in-house attention engine.

## Repository Layout

- `src/`, `include/`, `python_bindings/`, `tests/`, `benchmarks/` - C++/CUDA engine code, bindings, test scaffolding, and performance harnesses.
- `inference/Olmo_2/` - Hugging Face based runners (`run_from_snapshot.py`, `check_gpu.py`, profiler hook) that exercise locally staged weights.
- `inference/From_Scratch/` - staging area for the bespoke attention implementation (currently a placeholder awaiting engine bring-up).
- `llm_raw/olmo_2/` - canonical storage for downloaded checkpoints, tokenizer assets, and upstream metadata.
- `llm_setup/analysis/` - safetensor inspection utilities (`get_tensor_shapes_form_safetensors.py`, `verify_tensor_extraction.py`, etc.) used during asset validation.
- `llm_original/olmo_2_repo/` - pristine clone of AllenAI's official OLMo repository, fetched via `scripts/fetch_olmo2_repo.sh`.
- `setup/` - host bootstrapping scripts (`bare_metal_setup.sh`, venv helpers) plus dependency manifests.
- `scripts/` - automation for Vast.AI provisioning, environment setup, asset download, and architecture/state-dict analysis.
- `docs/` - project planning (`PROJECT_PLAN.md`) and the step-by-step Vast.AI baseline guide (`OLMO2_BASELINE.md`).

## Setup Overview

1. **Provision system dependencies**
   - `bash setup/bare_metal_setup.sh [.venv]` installs CUDA 12.8 toolkits, Nsight Systems, build essentials, and creates a Python virtual environment on a developer workstation.
   - `bash scripts/bootstrap_vast_ai.sh` executes the equivalent flow inside a Vast.AI instance, including CUTLASS cloning under `third_party/`.
2. **Optional container workflow** - Docker users can run `docker compose build` followed by `docker compose run --rm dev` for a reproducible CUDA-enabled environment (sources mounted at `/workspace`).
3. **Create the Olmo 2 Python environment**  
   `bash scripts/setup_olmo2_env.sh [.venv-olmo2]` wraps the bare-metal setup, verifies CUDA support in the selected PyTorch wheel, and leaves an activation-ready `.venv-olmo2/`.
4. **Activate the environment whenever you work with the baseline tools**  
   `source .venv-olmo2/bin/activate`

## Retrieve Olmo 2 Assets

Staging the model weights and metadata is automated:

```bash
source .venv-olmo2/bin/activate
python scripts/download_olmo2_assets.py
```

The script downloads `allenai/OLMo-2-1124-13B-Instruct` (override with `--model-id` when needed), mirrors the Hugging Face snapshot under `llm_raw/olmo_2/`, and runs the validation suite:

- `llm_setup/analysis/test_analysis.py` confirms required files exist.
- `get_tensor_shapes_form_safetensors.py` regenerates `tensor_inventory.csv`.
- `verify_tensor_extraction.py` spot-checks manual tensor reads.
- `inference/Olmo_2/check_gpu.py` runs when CUDA is available to ensure the checkpoint loads and generates successfully.

Weights are duplicated into `llm_raw/olmo_2/raw_weights/`, tokenizer assets land in `llm_raw/olmo_2/raw_tokenizer/`, and metadata (including the upstream README) is copied to `llm_raw/olmo_2/metadata/`.

## Baseline Inference Flows

### Flow A: Hugging Face runner (this repository)

1. Activate `.venv-olmo2`.
2. Optionally sanity-check the GPU:  
   `python inference/Olmo_2/check_gpu.py --prompt "GPU health check" --analysis`
3. Generate text from the staged snapshot:

   ```bash
   python inference/Olmo_2/run_from_snapshot.py \
     --model allenai/OLMo-2-1124-13B-Instruct \
     --prompt "Summarize the Olmo 2 architecture." \
     --max-new-tokens 128 \
     --trust-remote-code
   ```

`check_gpu_profiler.py` wraps the same workload with Nsight tracing so you can capture kernels for later analysis. Output artefacts can live beside the scripts under `inference/Olmo_2/`.

### Flow B: AllenAI reference repository

Run `bash scripts/fetch_olmo2_repo.sh [.venv-olmo2]` to clone `allenai/OLMo` into `llm_original/olmo_2_repo/`, refresh the virtual environment, and install the repo in editable mode along with its `inference/requirements.txt`. Activate the environment (`source .venv-olmo2/bin/activate`) and follow `llm_original/olmo_2_repo/README.md` for the official CLI and configuration examples. Use this flow to cross-check behaviour against upstream commits.

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
