#!/usr/bin/env bash
#set -euxo pipefail
set -u
trap 'echo "Error on line $LINENO"; exit 1' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH=${1:-.venv}

python3 -m venv "${VENV_PATH}"
source "${VENV_PATH}/bin/activate"

pip install --upgrade pip
pip install -r "${SCRIPT_DIR}/../requirements/requirements.txt"

echo "Virtual environment ready at ${VENV_PATH}"
