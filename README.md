# Attention-From-Scratch

[Project microsite](https://mauer4.github.io/attention-from-scratch/)

## Project Vision

I'm mapping the landscape of large language model inference so an individual developer can run production-grade open weights. The goal is not to pretrain a new network, but to understand and rebuild the inference stack, from environment provisioning and weight inspection through custom CUDA kernels that rival modern serving engines. The work splits into two flows: first, validating AllenAI's released inference engine end to end; second, building a bespoke attention-from-scratch engine that eventually replaces the upstream stack for targeted workloads.

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

## Common Setup Flow

Both inference paths share the same bootstrap pipeline (mirrors the setup-flow diagram in the planning notes).

1. **Provision the compute environment**
   - Bare metal: `bash setup/bare_metal_setup.sh [.venv]` installs CUDA 12.8 toolkits, Nsight Systems, build tools, and creates a Python virtual environment.
   - Vast.AI: `bash scripts/bootstrap_vast_ai.sh` prepares the rented instance (APT dependencies, CUDA repos, PyTorch wheel, CUTLASS checkout).
   - Docker: `docker compose build` then `docker compose run --rm dev` spawns a GPU-enabled container with the repository mounted at `/workspace`.
2. **Create and activate the shared Python environment**  
   Run `bash scripts/setup_olmo2_env.sh [.venv-olmo2]` (inside the container or on the host). The script wraps the bare-metal provisioning logic, ensures the selected PyTorch build matches your GPU, and leaves an activation-ready `.venv-olmo2/`. Activate it when working with either flow: `source .venv-olmo2/bin/activate`.
3. **Ensure weights are staged under `llm_raw/`**  
   With the environment active, execute:
   ```bash
   python scripts/download_olmo2_assets.py
   ```
   The helper only downloads `allenai/OLMo-2-1124-13B-Instruct` when the required files are missing (or when `--force` is passed). It mirrors the Hugging Face snapshot into `llm_raw/olmo_2/hf_snapshot/` and copies the usable assets into `llm_raw/olmo_2/raw_weights/`, `raw_tokenizer/`, and `metadata/`.
4. **Validate resources and wiring**  
   The download script automatically runs:
   - `llm_setup/analysis/test_analysis.py` to confirm expected files exist.
   - `get_tensor_shapes_form_safetensors.py` and `verify_tensor_extraction.py` to regenerate tensor inventories and check shard offsets.
   - `inference/Olmo_2/check_gpu.py` (when CUDA is available) to load the model and perform a short generation.
   Re-run these checks after modifying paths or swapping checkpoints to ensure both the AllenAI and custom engines see the same staged assets.

## Inference Flows

### 1. AllenAI OLMo inference engine

This flow leans on AllenAI's official repository so the baseline behaviour matches the published release.

1. Activate the Olmo environment: `source .venv-olmo2/bin/activate`.
2. Clone or refresh the repo with `bash scripts/fetch_olmo2_repo.sh [.venv-olmo2]`. The script:
   - Downloads the `allenai/OLMo` repository into `llm_original/olmo_2_repo/`.
   - Reuses `.venv-olmo2` (creating it if necessary) and installs the repo in editable mode alongside its `inference/requirements.txt`.
3. Populate weights via `python scripts/download_olmo2_assets.py` so the snapshot is staged under `llm_raw/olmo_2/`.
4. Follow `llm_original/olmo_2_repo/README.md` (and the repo's `inference/` folder) to launch AllenAI's CLI or scripts. Point them at the staged snapshot when running locally or use the default Hugging Face download path in cloud environments.

Supporting utilities in this repository include:

- `inference/Olmo_2/run_from_snapshot.py`, `check_gpu.py`, and `check_gpu_profiler.py` for quick health checks against the staged weights.
- `scripts/analyse_architecture.py` and `scripts/dump_state_dict_summary.py` to catalog the downloaded tensors.

Treat these helpers as diagnostics around the AllenAI engine rather than a replacement for it.

### 2. Custom attention-from-scratch engine (WIP)

The second flow is the bespoke CUDA/C++ inference stack under development.

- Source layout: core kernels and runtime live under `src/` and `include/`, with `python_bindings/` exposing a Python surface for tests and integration once the engine stabilises.
- Build tooling: `CMakeLists.txt`, `Dockerfile`, `docker-compose.yml`, and the scripts in `setup/` provide repeatable build environments on bare metal, Docker, and Vast.AI.
- Execution entry point: `inference/From_Scratch/` currently holds placeholder scripts; they will be replaced with the production driver when the engine reaches parity.
- Roadmap: `docs/PROJECT_PLAN.md` sequences the milestones from tokenizer validation through FlashAttention-style kernels, paged KV caches, quantisation, and serving.
- Benchmark targets: store Nsight traces and throughput data in `benchmarks/` to compare against the AllenAI baseline as new features land.

For now, use the plan to guide implementation tasks and rely on the AllenAI flow for functional inference. As kernels come online, keep the staged Olmo 2 weights in `llm_raw/` to validate numerical parity and performance.

## Custom Engine Roadmap

`docs/PROJECT_PLAN.md` captures the 12-week schedule that brings the custom engine online (tokenizer parity, FlashAttention-style kernels, paged KV caches, quantisation, batching, serving). Use it to plan implementation work while comparing against the AllenAI baseline for functional and performance validation.

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
