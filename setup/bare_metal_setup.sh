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
)

VENV_PATH=${1:-.venv}

if command -v apt-get >/dev/null 2>&1; then
  echo "[setup] Installing system packages via apt-get (sudo required)…"
  sudo apt-get update
  sudo apt-get install -y "${APT_PACKAGES[@]}"
else
  echo "[setup] apt-get not available. Please install the following manually:"
  printf '  %s\n' "${APT_PACKAGES[@]}"
fi

echo "[setup] Creating virtual environment at ${VENV_PATH}"
python3 -m venv "${VENV_PATH}"

echo "[setup] Activating virtual environment"
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

echo "[setup] Installing Python dependencies"
pip install --upgrade pip
pip install -r "$(dirname "$0")/requirements/requirements.txt"

echo "[setup] Environment ready. Run 'source ${VENV_PATH}/bin/activate' in new shells."
