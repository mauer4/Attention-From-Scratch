#!/usr/bin/env bash
# Convenience script for bare-metal hosts (no Docker) to provision system
# packages, create a virtual environment, and install Python dependencies.

set -euo pipefail

APT_PACKAGES=(
  build-essential
  git
  cmake
  ninja-build
  python3
  python3-dev
  python3-venv
  python3-pip
  clang
  curl
  ca-certificates
  cuda-toolkit-12-8
  cuda-nsight-systems-12-8
)

ensure_cuda_repo() {
  if ! command -v apt-get >/dev/null 2>&1; then
    return
  fi

  if dpkg -s cuda-toolkit-12-8 >/dev/null 2>&1; then
    return
  fi

  if [ -f /etc/os-release ]; then
    . /etc/os-release
    if [ "${ID:-}" = "ubuntu" ] && [ -n "${VERSION_ID:-}" ]; then
      local version_nodot="${VERSION_ID//./}"
      local repo_id="ubuntu${version_nodot}"
      local keyring_pkg="cuda-keyring_1.1-1_all.deb"
      local url="https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/x86_64/${keyring_pkg}"
      local tmp_deb="/tmp/${keyring_pkg}"
      echo "[setup] Adding NVIDIA CUDA APT repository for ${repo_id}"
      if command -v curl >/dev/null 2>&1; then
        curl -fsSL "${url}" -o "${tmp_deb}"
      elif command -v wget >/dev/null 2>&1; then
        wget -qO "${tmp_deb}" "${url}"
      else
        echo "[setup] Neither curl nor wget available; install cuda-keyring manually from ${url}."
        return
      fi
      sudo dpkg -i "${tmp_deb}"
      rm -f "${tmp_deb}"
      sudo apt-get update
    fi
  fi
}

VENV_PATH=${1:-.venv}

if command -v apt-get >/dev/null 2>&1; then
  ensure_cuda_repo
  echo "[setup] Installing system packages via apt-get (sudo required)..."
  sudo apt-get update
  sudo apt-get install -y "${APT_PACKAGES[@]}"
else
  echo "[setup] apt-get not available. Please install the following manually:"
  printf '  %s\n' "${APT_PACKAGES[@]}"
fi

echo "[setup] Creating virtual environment at ${VENV_PATH}"
python3 -m venv "${VENV_PATH}"

# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

echo "[setup] Installing Python dependencies"
"${VENV_PATH}/bin/pip" install --upgrade pip
"${VENV_PATH}/bin/pip" install -r "$(dirname "$0")/requirements/requirements.txt"

echo "[setup] Environment ready. Run 'source ${VENV_PATH}/bin/activate' in new shells."\n\n