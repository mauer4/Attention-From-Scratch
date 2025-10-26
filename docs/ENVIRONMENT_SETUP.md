# Environment Setup Guide

This project supports two main workflows:

1. **Docker/NVIDIA container** - easiest when you can run Docker with GPU passthrough (local workstation, some cloud providers).
2. **Bare-metal virtual environments** - when containers are unavailable or restricted (e.g., certain Vast.ai offerings).

Follow the section that matches your host capabilities.

---

## 1. Docker-based workflow

**Prerequisites**: Docker 20+, NVIDIA Container Toolkit (for GPU access).

The container provisions a virtual environment at `/opt/venv` and installs the
same Python requirements used for bare-metal setups. The environment is active
by default (`PATH` points at `/opt/venv/bin`).

```bash
# Build the CUDA-enabled image (runs CMake build during the image build step)
docker build -t attention-from-scratch .

# Launch an interactive session with GPU access
# (replace $PWD with %cd% on PowerShell)
docker run --gpus all --rm -it -v "$(pwd)":/workspace attention-from-scratch
```

Inside the container the code lives at `/workspace`, the CMake build has already run during `docker build`, and Python dependencies are installed system-wide. Re-run `cmake --build build` if you change C++ sources.

If you want a lighter-weight image (e.g., skipping the automatic build), remove or adjust the `cmake --build` line in `Dockerfile` before building.

---

## 2. Bare-metal setup (no Docker)

Use these steps on hosts where containers are unavailable. The examples assume a CUDA-capable Linux box; adapt the package manager as needed. A convenience script (`setup/bare_metal_setup.sh`) wraps the steps below for apt-based distributions; run it with `bash setup/bare_metal_setup.sh [venv_path]`.

### 2.1 System packages

```bash
sudo apt-get update
sudo apt-get install -y build-essential git cmake ninja-build python3 python3-dev \
    python3-venv python3-pip clang curl ca-certificates
```

When CUDA libraries are not preinstalled, install a compatible CUDA toolkit (the project currently targets CUDA 12.8, as reflected in the Dockerfile).

### 2.2 Python virtual environment

The project keeps `.venv` at the repository root (listed in `.gitignore`). Use one of the helper scripts under `setup/venv/` or run the commands manually.

| Shell | Command |
| ----- | ------- |
| PowerShell (Windows) | `powershell -ExecutionPolicy Bypass -File setup/venv/create_venv.ps1` |
| Bash / Zsh (Linux/macOS) | `bash setup/venv/create_venv.sh` |
| csh / tcsh | `csh setup/venv/create_venv.csh` |

Each script accepts an optional custom path as the first argument if you do not want to use `.venv`.

Manual equivalent:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r setup/requirements/requirements.txt
```

### 2.3 Build the C++/CUDA components (optional)

Run only if you need to build the native attention library or Python bindings.

```bash
cmake -S . -B build -G Ninja
cmake --build build
```

Add `-DATTN_BUILD_PYTHON=OFF` if you want to skip the pybind11 module.

### 2.4 Validate the analysis tooling

After the Python environment is active, run the smoke tests to ensure raw assets are discoverable:

```bash
python llm_setup/analysis/test_analysis.py
python llm_setup/analysis/verify_tensor_extraction.py --tensor-name lm_head.weight
```

To fetch weights/metadata the project relies on `python scripts/download_olmo2_assets.py`, which stages files under `llm_raw/olmo_2/` and re-runs the analysis checks automatically.

---

## 3. Notes for Vast.ai and similar providers

- If the instance already launches inside a container (common for Vast.ai base images), treat it like the bare-metal workflow: install missing system packages inside the existing container and use the helper scripts for virtual environments. Spawning nested Docker containers is usually not allowed.
- If Docker is enabled (some dedicated GPU servers), follow the Docker instructions instead.
- Ensure the host exposes `/dev/nvidia*` devices and CUDA drivers compatible with CUDA 12.x when running the native build.

---

## 4. Quick reference

| Scenario | Steps |
| -------- | ----- |
| Docker with GPU available | `docker build -t attention-from-scratch .`, `docker run --gpus all --rm -it -v "$(pwd)":/workspace attention-from-scratch` |
| Linux/macOS without Docker | `bash setup/venv/create_venv.sh`, `cmake --build build` |
| Windows | `powershell -ExecutionPolicy Bypass -File setup/venv/create_venv.ps1` |
| csh shells | `csh setup/venv/create_venv.csh` |
| Fetch OLMo snapshot | `python scripts/download_olmo2_assets.py` |
| Clone upstream repo (optional) | `bash scripts/fetch_olmo2_repo.sh` |

After any setup path, activate the environment, then use scripts under
`llm_setup/analysis/` or run CMake builds as needed.


