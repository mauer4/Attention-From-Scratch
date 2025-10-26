#!/usr/bin/env bash
# Create a Python virtual environment for the analysis tooling.
set -euo pipefail
VENV_PATH=${1:-.venv}
python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"
pip install --upgrade pip
pip install -r "$(dirname "$0")/../requirements/requirements.txt"
echo "Virtual environment ready at ${VENV_PATH}"
