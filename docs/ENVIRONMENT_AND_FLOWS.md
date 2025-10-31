# Environment & Inference Flows

This guide unifies the environment bootstrap story for Attention-From-Scratch
and walks through the two supported inference flows:

1. **Flow 1** – run AllenAI's reference OLMo repository.
2. **Flow 2** – (planned) run the custom from-scratch attention engine.

Use the decision table below to choose the right bootstrap configuration, then
follow the flow-specific checklists.

## 1. Environment decision matrix

| Host scenario | Recommended command | Notes |
| ------------- | ------------------ | ----- |
| Local workstation with sudo access | `bash setup/bootstrap_host.sh` | Installs system packages, detects the NVIDIA driver, provisions a compatible CUDA toolkit, and creates `.venv-olmo2`. |
| Vast.ai / managed container with limited permissions | `SKIP_SYSTEM_PACKAGES=1 SKIP_CUDA_TOOLKIT=1 bash setup/bootstrap_host.sh --python-env ~/.venvs/olmo` | Reuses preinstalled drivers/toolkit and only manages the virtual environment. |
| Docker image build (Dockerfile already handles this) | `bash setup/bootstrap_host.sh --system none --cuda-toolkit none --python-env /opt/venv` | Used inside the bundled Dockerfile; can be run manually in other container workflows. |
| Hosts without GPUs (CPU-only dev) | `TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu bash setup/bootstrap_host.sh` | Installs CPU torch wheels; useful for validating the AllenAI CLI without CUDA. |

`setup/bootstrap_host.sh --help` documents additional switches such as
`--lock` for custom lockfiles or `--with-cutlass` to clone NVIDIA CUTLASS when
you start working on the custom kernels. Every run streams to
`logs/bootstrap_<timestamp>.log`, which makes post-mortem debugging easy on
shared machines. When driver detection is unavailable the script defaults to
CUDA 12.8, matching the repository’s primary toolchain.
During Python dependency installation the bootstrapper also double-checks the
PyTorch wheels: if CPU-only builds were pinned in the lock file it
automatically reinstalls the matching CUDA wheels from the detected PyTorch
index so a subsequent `make setup` keeps the venv GPU-ready.

### Dependency locking

Python dependencies are pinned in `pyproject.toml` and resolved into
`requirements/locks/olmo.lock` via `pip-compile`. Both the bootstrap script and
the Dockerfile install from this lock so environments stay in sync. Regenerate
the lock after dependency changes:

```bash
python3 -m pip install --user --upgrade pip-tools
python3 -m piptools compile --extra olmo --output-file requirements/locks/olmo.lock pyproject.toml
```

### Bootstrap behaviour

The script creates (or reuses) the requested virtual environment but does **not**
leave it activated. Follow the final log line—typically
`source .venv-olmo2/bin/activate`—in every new shell. All runs emit a detailed
record under `logs/bootstrap_<timestamp>.log` so you can audit package installs
afterwards.

## 2. Flow 1 – `run_from_snapshot.py` (Hugging Face loader)

This flow drives the legacy Hugging Face bridge so you can generate text from
the locally staged snapshot without touching the upstream repository.

1. **Bootstrap and activate the environment**
   ```bash
   bash setup/bootstrap_host.sh --python-env .venv-olmo2
   source .venv-olmo2/bin/activate
   ```
2. **Stage weights, tokenizer, and metadata**
   ```bash
   python scripts/download_olmo2_assets.py
   ```
   Files land under `llm_raw/olmo_2/` and can be reused across subsequent runs.
3. **Generate text**
   ```bash
   python inference/Olmo_2/run_from_snapshot.py \
     --prompt "Summarize the Olmo 2 architecture." \
     --max-new-tokens 64
   ```
   or use the Make shortcut: `make run-olmo ARGS='--prompt "..." --max-new-tokens 64'`.
   The script wires the staged assets into `transformers` (v4.48+) and prints the
   completion plus throughput metrics.

## 3. Flow 2 – Custom attention engine (planned)

The from-scratch engine will live under `src/`, `python_bindings/`, and
`inference/From_Scratch/`. Until those components land:

- Bootstrap the environment the same way as Flow 1.
- Keep assets staged under `llm_raw/olmo_2/` using `scripts/download_olmo2_assets.py`.
- Track progress in `docs/PROJECT_PLAN.md`; once kernels and drivers are merged,
  this section will include build and execution notes for the bespoke runner.

## 4. Optional – AllenAI reference repository

When you need parity with the upstream CLI, clone the official repo:

```bash
make fetch-olmo          # set FETCH_OLMO_UPDATE=1 to pull new commits
source .venv-olmo2/bin/activate
cd llm_original/olmo_2_repo
python -m olmo.generate --help
```

The helper keeps the repo synced under `llm_original/olmo_2_repo/` while
preserving your staged assets.

## 5. Vast.ai / restricted environment tips

- Prefer the project bootstrap script even when you cannot elevate privileges;
  set `SKIP_SYSTEM_PACKAGES=1` and `SKIP_CUDA_TOOLKIT=1` to avoid failing on
  locked-down instances.
- If `nvidia-smi` is unavailable because the provider hides it, specify your
  desired torch wheel index explicitly (`TORCH_INDEX_URL=<cu118|cu121|cpu>`).
- Cache the `.venv-olmo2` directory across reboots to avoid repeated wheel
  downloads.

## 6. Optional components

- **CUTLASS** – Clone only when you start building the custom CUDA kernels:  
  `ATTN_INCLUDE_CUTLASS=1 bash setup/bootstrap_host.sh --with-cutlass`
- **C++/CUDA engine build** – The unified bootstrap does not run CMake by
  default. Trigger builds when needed:
  ```bash
  cmake -S . -B build -G Ninja
  cmake --build build
  ```

Common developer shortcuts live in the top-level `Makefile` (`make setup`,
`make setup-vast`, `make download-assets`, `make fetch-olmo`, etc.), so you can
reuse the same flows without retyping full commands.
