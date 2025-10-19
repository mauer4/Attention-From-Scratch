#!/usr/bin/env bash
set -euo pipefail

# Set up a dedicated virtual environment for running Olmo 2 inference.

ENV_DIR="${1:-.venv-olmo2}"
PYTHON_BIN="${PYTHON:-python3}"

if [ ! -d "${ENV_DIR}" ]; then
    "${PYTHON_BIN}" -m venv "${ENV_DIR}"
    echo "Created virtual environment at ${ENV_DIR}"
else
    echo "Using existing virtual environment at ${ENV_DIR}"
fi

# shellcheck disable=SC1090
source "${ENV_DIR}/bin/activate"

pip install --upgrade pip
pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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

echo
echo "Olmo 2 environment ready."
echo "Activate it with: source ${ENV_DIR}/bin/activate"
echo "Then run inference via: python scripts/run_olmo2_inference.py --help"
