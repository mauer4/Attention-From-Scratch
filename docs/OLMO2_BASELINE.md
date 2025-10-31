# Olmo 2 Baseline Inference on Vast.AI

This guide sets up an isolated Python environment to run AllenAI's Olmo 2 checkpoints on a Vast.AI GPU instance. It keeps dependency management separate from the main project bootstrap so you can iterate and benchmark independently.

## 1. Prepare the Environment

Follow the decision matrix in [`docs/ENVIRONMENT_AND_FLOWS.md`](ENVIRONMENT_AND_FLOWS.md).
On Vast.ai we typically disable package/toolkit installation and reuse the host
driver stack:

```bash
git clone https://github.com/mauer4/Attention-From-Scratch.git
cd Attention-From-Scratch

SKIP_SYSTEM_PACKAGES=1 SKIP_CUDA_TOOLKIT=1 \
  bash setup/bootstrap_host.sh --python-env .venv-olmo2

source .venv-olmo2/bin/activate
python scripts/download_olmo2_assets.py
```

The bootstrap script installs from the locked dependency set (PyTorch, HF stack,
etc.) while leaving the system Python untouched.

## 2. Run a Baseline Generation

```bash
cd llm_original/olmo_2_repo
python -m olmo.generate --help               # inspect CLI entrypoints
# Example (see upstream README for full option set):
# python -m olmo.generate --model allenai/OLMo-2-1124-13B-Instruct --prompt "Vast.ai smoke test"
```

The upstream CLI loads weights from the Hugging Face Hub by default. To prefer
the staged snapshot, set the appropriate flags documented in AllenAI's README,
pointing them at `llm_raw/olmo_2/`.

## 3. Next Steps Toward Profiling

The goal is to transition from “baseline hf-inference” into detailed kernel profiling with Nsight:

- Record consistent prompts and batch sizes to compare throughput vs. the forthcoming custom engine.
- Capture GPU utilization (`nvidia-smi dmon`) and memory footprints during runs.
- Install Nsight Systems / Nsight Compute inside the Vast container or stream the session back to a local GUI. NVIDIA’s container images already include the CLI tools (`nsys`, `ncu`).
- Re-run the AllenAI CLI under Nsight to collect timeline traces and kernel metrics (e.g., `nsys profile python -m olmo.generate --config <...>`).
- Store reports under `benchmarks/` (keep traces outside Git or add to `.gitignore`).

These measurements will anchor the performance targets for the from-scratch implementation described in `docs/PROJECT_PLAN.md`.
\n\n\n
