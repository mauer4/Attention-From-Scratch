# Environment Architecture

The Attention-From-Scratch environment is intentionally layered so issues are isolated and reproducible across hosts. Each step emits machine-readable artefacts under `reports/` and human-readable summaries in `docs/`. The latest verification timestamp is recorded inside `reports/environment_report.md`.

## Layer Overview

1. **Hardware & Drivers** (`setup_env/check_gpu.py`)
 - Captures GPU inventory, driver versions, CUDA runtime, cuDNN, and NCCL.
  - Writes `reports/system_gpu.json`, exports `.env.autodetected`, and surfaces actionable warnings (e.g. mismatched CUDA).
2. **Virtual Environment** (`setup_env/create_venv.sh`)
   - Creates `.venv/` on all supported platforms with conda fallback.
   - Ensures shell activation is easy to script (`source setup_env/create_venv.sh`).
3. **Python Dependencies** (`setup_env/install_deps.sh`)
   - Installs `requirements.txt` and performs `pip install -e .`.
   - Validates torch CUDA compatibility, regenerates `requirements.lock`, and records `reports/pip_tree.txt`.
4. **Project Code Validation**
   - Editable install guarantees that `inference.engine`, `custom_engine.core`, and `analysis.benchmark_runtime` import cleanly.
   - Missing files or packages surface immediately with actionable errors.
5. **Configuration Verification** (`setup_env/verify_config.py`)
 - Loads `configs/default.yaml` plus overrides from `.env`.
  - Checks for `model-*.safetensors` shards and tokenizer assets under `weights/<model>/`.
6. **Sanity Inference** (`scripts/test_inference.py`)
   - Executes a minimal GPU-backed inference path, writing `reports/test_summary.json` and `reports/test_summary.md`.
7. **Reporting** (`setup_env/run_env_report.py`)
   - Collates the previous layers into Markdown and HTML summaries and records the last verification timestamp.

## Lockfile Strategy

- `requirements.txt` remains the authoritative list of direct dependencies.
- `setup_env/install_deps.sh` creates `requirements.lock` using `pip freeze | sort` so the exact wheel set is reproducible.
- `reports/pip_tree.txt` captures the dependency graph at install time for audit trails.
- Commit `requirements.txt` and `requirements.lock` to version control. Regenerate the lockfile whenever you upgrade packages.

## Configuration Sources

- `configs/default.yaml` tracks model name, weight directory, and tokenizer location.
- `configs/vast.yaml` inherits defaults but tweaks runtime dtype for typical Vast.ai GPUs.
- `.env` holds overrides such as `MODEL_WEIGHTS_ROOT` or prod secrets (never commit secrets).

## Verification Workflow

```bash
bash setup_env/install_all.sh
python scripts/download_weights.py --model-name olmo2
python scripts/test_inference.py  # optional rerun
python setup_env/run_env_report.py  # rebuild report if needed
```

All scripts are idempotent. If you change CUDA drivers, torch version pins, or config paths, rerun the installers to regenerate artefacts and catch regressions early.
