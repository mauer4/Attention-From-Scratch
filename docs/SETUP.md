# Setup Guide

This guide explains how to bring up the Attention-From-Scratch environment on both local machines and Vast.ai GPU rentals. It uses the layered automation shipped under `setup_env/` so every run is reproducible and self-verifying.

## Prerequisites

- NVIDIA GPU with drivers installed (see [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md) for fixes).
- CUDA-capable container or host (Ubuntu 22.04, WSL2, or macOS with eGPU passthrough for experimentation).
- Python 3.12+ and `python3 -m venv` available. Conda is optional but supported.

## Local Ubuntu / WSL Workflow

1. Clone the repository and ensure submodules (if any) are initialised.
2. Run the orchestrator:
   ```bash
   bash setup_env/install_all.sh
   ```
3. Download weights and tokenizer assets (runs idempotently):
   ```bash
   python scripts/download_weights.py --model-name olmo2
   ```
4. Review the generated report at `reports/environment_report.md` (the header lists the last verification timestamp).
5. Activate the virtual environment for interactive work:
   ```bash
   source .venv/bin/activate
   ```
6. Launch quick validation:
   ```bash
   python scripts/test_inference.py
   ```

## Vast.ai Containers

The scripts auto-detect Vast.ai hosts via `/etc/vastai-release` or the presence of `/run/nvidia`. Docker instructions are skipped and `.venv/` is created inside the container filesystem. Steps:

1. Deploy a CUDA-capable image (e.g. `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime`).
2. Mount your project storage, then clone this repository.
3. Execute the same command:
   ```bash
   bash setup_env/install_all.sh
   ```
4. Download weights once the environment is ready:
   ```bash
   python scripts/download_weights.py --model-name olmo2
   ```
5. If checkpoints live on another volume, bind-mount them to `weights/` or update the paths in `configs/default.yaml`.

## Torch Wheel Compatibility

| GPU Architecture | Recommended CUDA | Torch Wheel Index URL |
| ---------------- | ---------------- | --------------------- |
| RTX 30 series (Ampere) | 11.8 | `https://download.pytorch.org/whl/cu118` |
| RTX 40/50 series (Ada/Blackwell) | 12.1+ | `https://download.pytorch.org/whl/cu121` |
| A100 | 11.8 / 12.1 | `https://download.pytorch.org/whl/cu118` or `https://download.pytorch.org/whl/cu121` |
| H100 | 12.1+ | `https://download.pytorch.org/whl/cu121` |

`setup_env/install_deps.sh` automatically reconciles the wheel index with the detected CUDA runtime. Override by exporting `TORCH_WHEEL_INDEX_URL` before running the script if you need a preview build.

## After Setup

- `source .venv/bin/activate` whenever you open a new shell.
- Use `pip install -e .` (already invoked by the installer) to keep module changes live.
- Re-run `bash setup_env/install_all.sh` after modifying drivers, CUDA toolkits, or dependency pins to refresh the report.

For troubleshooting tips, visit [docs/TROUBLESHOOTING.md](TROUBLESHOOTING.md).
