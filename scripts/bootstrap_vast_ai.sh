#!/usr/bin/env bash
set -euo pipefail

# Bootstrap script for a Vast.AI GPU instance.

apt-get update
apt-get install -y --no-install-recommends \
    build-essential \
    git \
    cmake \
    ninja-build \
    python3-dev \
    python3-pip \
    clang \
    curl

ensure_cuda_repo() {
    if dpkg -s cuda-keyring >/dev/null 2>&1; then
        # Ensure repo file has signed-by directive even if keyring already present.
        if [ -f /etc/os-release ]; then
            # shellcheck disable=SC1091
            . /etc/os-release
            if [ "${ID:-}" = "ubuntu" ] && [ -n "${VERSION_ID:-}" ]; then
                local version_nodot="${VERSION_ID//./}"
                local repo_id="ubuntu${version_nodot}"
                local repo_line="deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/x86_64/ /"
                local list_dir="/etc/apt/sources.list.d"
                mkdir -p "${list_dir}"
                for file in "${list_dir}"/*.list; do
                    if [ -f "${file}" ] && grep -q "developer.download.nvidia.com/compute/cuda" "${file}"; then
                        rm -f "${file}"
                    fi
                done
                echo "${repo_line}" > "${list_dir}/cuda-${repo_id}.list"
            fi
        fi
        return
    fi

    if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        if [ "${ID:-}" = "ubuntu" ] && [ -n "${VERSION_ID:-}" ]; then
            local version_nodot="${VERSION_ID//./}"
            local repo_id="ubuntu${version_nodot}"
            local tmp_deb="/tmp/cuda-keyring_1.1-1_all.deb"
            curl -fsSL \
                "https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/x86_64/cuda-keyring_1.1-1_all.deb" \
                -o "${tmp_deb}"
            dpkg -i "${tmp_deb}"
            rm -f "${tmp_deb}"
            local repo_line="deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/x86_64/ /"
            local list_dir="/etc/apt/sources.list.d"
            mkdir -p "${list_dir}"
            for file in "${list_dir}"/*.list; do
                if [ -f "${file}" ] && grep -q "developer.download.nvidia.com/compute/cuda" "${file}"; then
                    rm -f "${file}"
                fi
            done
            echo "${repo_line}" > "${list_dir}/cuda-${repo_id}.list"
            apt-get update
            return
        fi
    fi

    echo "Warning: Unable to determine CUDA repository for this distribution; skipping cuda-keyring install."
}

ensure_cuda_repo
apt-get update
apt-get install -y --no-install-recommends \
    cuda-toolkit-12-8 \
    cuda-nsight-systems-12-8

python3 -m pip install --upgrade pip
TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

detect_compute_capability() {
    local cap
    if command -v nvidia-smi >/dev/null 2>&1; then
        cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
        if [[ "${cap}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
            echo "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
        fi
    fi
}

pick_torch_index() {
    local cap="$1"
    if [ -n "${TORCH_INDEX_URL}" ]; then
        echo "${TORCH_INDEX_URL}"
        return
    fi

    if [ -z "${cap}" ]; then
        echo "https://download.pytorch.org/whl/cu121"
        return
    fi

    local major="${cap%%.*}"
    if (( major >= 12 )); then
        echo "https://download.pytorch.org/whl/nightly/cu128"
    else
        echo "https://download.pytorch.org/whl/cu121"
    fi
}

COMPUTE_CAPABILITY="$(detect_compute_capability || true)"
SELECTED_INDEX_URL="$(pick_torch_index "${COMPUTE_CAPABILITY}")"

if [ -n "${COMPUTE_CAPABILITY}" ]; then
    echo "Detected GPU compute capability ${COMPUTE_CAPABILITY}. Installing PyTorch from ${SELECTED_INDEX_URL}"
else
    echo "Unable to detect GPU compute capability; defaulting to PyTorch from ${SELECTED_INDEX_URL}"
fi

PYTHON3_PIP_PRE_FLAG=""
if [[ "${SELECTED_INDEX_URL}" == *"nightly"* ]]; then
    PYTHON3_PIP_PRE_FLAG="--pre"
fi

python3 -m pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
python3 -m pip install --upgrade ${PYTHON3_PIP_PRE_FLAG} torch --index-url "${SELECTED_INDEX_URL}"
if ! python3 -m pip install --upgrade ${PYTHON3_PIP_PRE_FLAG} torchvision torchaudio --index-url "${SELECTED_INDEX_URL}"; then
    echo "Warning: torchvision/torchaudio install failed; continuing without them."
fi
python3 -m pip install \
    transformers \
    safetensors \
    sentencepiece \
    numpy \
    pybind11 \
    tqdm

python3 - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"Failed to import torch after installation: {exc}") from exc

if not torch.cuda.is_available():
    raise SystemExit("The installed torch build does not detect CUDA. Check driver/toolkit compatibility.")

device_capability = torch.cuda.get_device_capability()
arch = f"sm_{device_capability[0]}{device_capability[1]}"
compiled_arches = {arch_name.replace('+PTX', '') for arch_name in torch.cuda.get_arch_list()}
if arch not in compiled_arches:
    msg = (
        f"Torch {torch.__version__} is missing kernels for {arch}. "
        "Install a build that targets your GPU architecture "
        "(set TORCH_INDEX_URL to the appropriate wheel or build from source)."
    )
    raise SystemExit(msg)

print(f"Cuda capability check passed for {arch} with torch {torch.__version__}.")
PY

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
CUTLASS_DIR="${THIRD_PARTY_DIR}/cutlass"

mkdir -p "${THIRD_PARTY_DIR}"
if [ ! -d "${CUTLASS_DIR}" ]; then
    git clone --depth 1 https://github.com/NVIDIA/cutlass.git "${CUTLASS_DIR}"
fi

echo "Bootstrap complete."
