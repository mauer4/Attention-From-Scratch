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
    cutlass \
    tqdm

echo "Bootstrap complete."
