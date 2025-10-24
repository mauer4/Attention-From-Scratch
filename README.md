# Attention-From-Scratch

[Project microsite](https://mauer4.github.io/attention-from-scratch/)

## Project Vision

I'm mapping the landscape of large language models to see what's possible for an individual, not just for massive tech teams. The exciting news is that fully open-source LLMs exist: you can inspect the weights, study the architecture, run the inference code, and even examine the training stack. As someone who loves marrying hardware, software, and math (classic HW/SW co-design), I'm inspired by the breakthroughs that pushed LLM performance forward: from FlashAttention and its fused kernels, to asynchronous attention in FlashAttention 2/3, and the paged-attention serving tricks in engines like vLLM. With today's cloud options, renting enterprise-class GPUs for inference is also within reach. So I've set a personal roadmap that takes me from beginner to building production-grade inference infrastructure on top of open models like Olmo 2. I'm not aiming to train new models; I'm aiming to run them exceptionally well. Along the way, I hope to understand the algorithms behind one of the most impactful technologies of our time and maybe even find new ways to push them further.

## Repository Structure

- `src/` - core C++ and CUDA source code.
- `include/` - public headers for the inference engine.
- `python_bindings/` - pybind11 bindings and packaging glue.
- `benchmarks/` - performance measurement utilities.
- `scripts/` - automation for environment setup and workflows.
- `tests/` - unit and integration tests.
- `docs/` - design notes, technical reports, and planning.

See `docs/PROJECT_PLAN.md` for the 12-week roadmap guiding development.

## Containerized Environment

Prerequisites:

- Docker with the NVIDIA Container Toolkit (`nvidia-smi` works inside containers).
- `docker compose` plugin.

Quick start:

1. Build the image: `docker compose build`
2. Launch an interactive session with GPU access: `docker compose run --rm dev`
3. Inside the container, verify CUDA: `nvidia-smi`
4. Build the project (already run during image build) or rerun as needed: `cmake --build build`

The working directory is mounted at `/workspace`, so changes sync to your host. Use the container for consistent dependency management across GPUs and environments.

> During `docker compose build` the NVIDIA CUTLASS sources are cloned into `third_party/cutlass`. The same happens when running `scripts/bootstrap_vast_ai.sh`; no manual install is required.

## Olmo 2 Baseline on Vast.AI

Use the dedicated scripts to prepare and exercise an isolated Olmo 2 environment:

1. `bash scripts/setup_olmo2_env.sh` to create a `.venv-olmo2` virtualenv and install dependencies.
2. `source .venv-olmo2/bin/activate`
3. `python scripts/run_olmo2_inference.py --trust-remote-code --help` to explore generation options (OLMo 2 repositories supply custom code that must be trusted). The setup script also pulls AllenAI's `hf_olmo` helper package directly from GitHub to satisfy Hugging Face's dynamic import.

Refer to `docs/OLMO2_BASELINE.md` for detailed instructions and upcoming profiling steps with Nsight tools.

### GPU Sanity Check & Profiling

`python scripts/check_olmo_gpu.py` loads `allenai/OLMo-2-1124-13B`, generates a short sample, and prints CUDA diagnostics. Useful flags:

- `--analysis` records model-load / generation latency, peak GPU memory, and throughput.
- `--nsight` re-runs the script under Nsight Systems (requires `cuda-nsight-systems-12-8` from the NVIDIA apt repo). Use `--nsight-output=my_run` to rename the resulting `my_run.nsys-rep`.

Both `scripts/setup_olmo2_env.sh` and `scripts/bootstrap_vast_ai.sh` ensure CUDA Toolkit 12.8, Nsight Systems, and a compatible PyTorch build are present. If you previously ran either script before these changes, rerun it to refresh the toolchain and gain Nsight profiling support.

### Inspecting Model Structure

Once you have an Olmo 2 snapshot cached locally (for example the files the Hugging Face downloader places under `~/.cache/huggingface/hub/`), you can generate descriptive tables that summarize the architecture and the exact tensors present in the checkpoint:

```bash
# High-level architecture summary (console + CSV)
python scripts/analyse_architecture.py /path/to/snapshot \
    --layer-table-csv scripts/layer_summary.csv

# Low-level state-dict tensor listing (console + CSV)
python scripts/dump_state_dict_summary.py /path/to/snapshot \
    --output scripts/state_dict_summary.csv
```

`analyse_architecture.py` reads `config.json` to report layer geometry, embeddings, norms, and RoPE settings; the optional CSV makes it easy to compare runs. `dump_state_dict_summary.py` walks the safetensor shards named in `model.safetensors.index.json` and records the true shapes/dtypes for every weight, which is useful for cross-checking against the architectural view.
