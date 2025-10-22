#!/usr/bin/env bash
set -euo pipefail

# Set up a dedicated virtual environment for running Olmo 2 inference.

ENV_DIR="${1:-.venv-olmo2}"
PYTHON_BIN="${PYTHON:-python3}"

version_ge() {
    # Returns success if $1 >= $2 (semantic version compare)
    [ "$(printf '%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

ensure_cuda_prereqs() {
    local have_nvcc=0
    if command -v nvcc >/dev/null 2>&1; then
        local version_line
        version_line="$(nvcc --version | grep -i 'release' || true)"
        if [[ "${version_line}" =~ release[[:space:]]+([0-9]+\.[0-9]+) ]]; then
            local version="${BASH_REMATCH[1]}"
            if version_ge "${version}" "12.8"; then
                have_nvcc=1
            fi
        fi
    fi

    local need_install=0
    if [ "${have_nvcc}" -eq 0 ]; then
        need_install=1
    elif ! command -v nsys >/dev/null 2>&1; then
        need_install=1
    fi

    if [ "${need_install}" -eq 0 ]; then
        return
    fi

    if [ "$(id -u)" -ne 0 ] && ! command -v sudo >/dev/null 2>&1; then
        cat <<'MSG'
CUDA 12.8 toolkit or Nsight Systems is missing, but elevated privileges are required to install them.
Install manually with:
  sudo apt-get update
  sudo apt-get install cuda-toolkit-12-8 cuda-nsight-systems-12-8
MSG
        return
    fi

    local apt_cmd=(apt-get)
    if [ "$(id -u)" -ne 0 ]; then
        apt_cmd=(sudo "${apt_cmd[@]}")
    fi

    ${apt_cmd[@]} update
    if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
        ${apt_cmd[@]} install -y curl
    fi

    local priv_cmd=()
    if [ "$(id -u)" -ne 0 ]; then
        priv_cmd=(sudo)
    fi

    ensure_cuda_repo() {
        if dpkg -s cuda-keyring >/dev/null 2>&1; then
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
                            ${priv_cmd[@]} rm -f "${file}"
                        fi
                    done
                    printf '%s\n' "${repo_line}" | ${priv_cmd[@]} tee "${list_dir}/cuda-${repo_id}.list" >/dev/null
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
                if command -v curl >/dev/null 2>&1; then
                    curl -fsSL \
                        "https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/x86_64/cuda-keyring_1.1-1_all.deb" \
                        -o "${tmp_deb}"
                elif command -v wget >/dev/null 2>&1; then
                    wget -qO "${tmp_deb}" \
                        "https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/x86_64/cuda-keyring_1.1-1_all.deb"
                else
                    echo "Unable to download cuda-keyring (curl/wget unavailable)."
                    return
                fi
                local priv_cmd=()
                if [ "$(id -u)" -ne 0 ]; then
                    priv_cmd=(sudo)
                fi
                ${priv_cmd[@]} dpkg -i "${tmp_deb}"
                rm -f "${tmp_deb}"
                local repo_line="deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/x86_64/ /"
                local list_dir="/etc/apt/sources.list.d"
                mkdir -p "${list_dir}"
                for file in "${list_dir}"/*.list; do
                    if [ -f "${file}" ] && grep -q "developer.download.nvidia.com/compute/cuda" "${file}"; then
                        ${priv_cmd[@]} rm -f "${file}"
                    fi
                done
                printf '%s\n' "${repo_line}" | ${priv_cmd[@]} tee "${list_dir}/cuda-${repo_id}.list" >/dev/null
                ${apt_cmd[@]} update
            fi
        fi
    }

    ensure_cuda_repo
    ${apt_cmd[@]} update
    ${apt_cmd[@]} install -y cuda-toolkit-12-8 cuda-nsight-systems-12-8
}

ensure_cuda_prereqs || true

if [ ! -d "${ENV_DIR}" ]; then
    "${PYTHON_BIN}" -m venv "${ENV_DIR}"
    echo "Created virtual environment at ${ENV_DIR}"
else
    echo "Using existing virtual environment at ${ENV_DIR}"
fi

# shellcheck disable=SC1090
source "${ENV_DIR}/bin/activate"

pip install --upgrade pip
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

PIP_PRE_FLAG=""
if [[ "${SELECTED_INDEX_URL}" == *"nightly"* ]]; then
    PIP_PRE_FLAG="--pre"
fi

pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
pip install --upgrade ${PIP_PRE_FLAG} torch --index-url "${SELECTED_INDEX_URL}"
if ! pip install --upgrade ${PIP_PRE_FLAG} torchvision torchaudio --index-url "${SELECTED_INDEX_URL}"; then
    echo "Warning: torchvision/torchaudio install failed; continuing without them."
fi

pip install \
    transformers \
    accelerate \
    sentencepiece \
    safetensors \
    bitsandbytes \
    huggingface_hub \
    numpy \
    rich \
    typer

pip install \
    "git+https://github.com/allenai/OLMo.git#subdirectory=hf_olmo"

python - <<'PY'
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

echo
echo "Olmo 2 environment ready."
echo "Activate it with: source ${ENV_DIR}/bin/activate"
echo "Then run inference via: python scripts/run_olmo2_inference.py --help"
