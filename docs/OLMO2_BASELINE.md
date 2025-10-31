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
python inference/Olmo_2/run_from_snapshot.py \
  --prompt "Vast.ai smoke test" \
  --max-new-tokens 128 \
  --device cuda
```

This uses the locally staged snapshot via `transformers` (v4.48+). Prefer the
`--device cuda` flag on GPUs to avoid CPU fallback. Alternatively, use the Make
wrapper: `make run-olmo ARGS='--prompt "Vast.ai smoke test" --max-new-tokens 128 --device cuda'`.

If you need to compare against AllenAI’s CLI, run `make fetch-olmo` and follow
`llm_original/olmo_2_repo/README.md`, but that flow is optional for the baseline.

## 3. Next Steps Toward Profiling

The goal is to transition from “baseline hf-inference” into detailed kernel profiling with Nsight:

- Record consistent prompts and batch sizes to compare throughput vs. the forthcoming custom engine.
- Capture GPU utilization (`nvidia-smi dmon`) and memory footprints during runs.
- Install Nsight Systems / Nsight Compute inside the Vast container or stream the session back to a local GUI. NVIDIA’s container images already include the CLI tools (`nsys`, `ncu`).
- Re-run the AllenAI CLI under Nsight to collect timeline traces and kernel metrics (e.g., `nsys profile python -m olmo.generate --config <...>`).
- Store reports under `benchmarks/` (keep traces outside Git or add to `.gitignore`).

These measurements will anchor the performance targets for the from-scratch implementation described in `docs/PROJECT_PLAN.md`.
\n\n\n
