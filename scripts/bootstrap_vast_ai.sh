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
    clang

python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install \
    transformers \
    safetensors \
    sentencepiece \
    numpy \
    pybind11 \
    tqdm

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY_DIR="${ROOT_DIR}/third_party"
CUTLASS_DIR="${THIRD_PARTY_DIR}/cutlass"

mkdir -p "${THIRD_PARTY_DIR}"
if [ ! -d "${CUTLASS_DIR}" ]; then
    git clone --depth 1 https://github.com/NVIDIA/cutlass.git "${CUTLASS_DIR}"
fi

echo "Bootstrap complete."
