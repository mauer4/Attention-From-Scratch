#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${ROOT_DIR}/llm_original/olmo_2_repo"
ENV_DIR="${1:-${ROOT_DIR}/.venv-olmo2}"

if [ ! -d "${REPO_DIR}/.git" ]; then
    git clone --depth 1 https://github.com/allenai/OLMo.git "${REPO_DIR}"
else
    git -C "${REPO_DIR}" pull --ff-only
fi

echo "[setup] Preparing virtual environment via scripts/setup_olmo2_env.sh"
"${ROOT_DIR}/scripts/setup_olmo2_env.sh" "${ENV_DIR}"

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

pip install -e "${REPO_DIR}"
TMP_REQ=$(mktemp)
sed 's|git+ssh://git@github.com/|git+https://github.com/|' "${REPO_DIR}/inference/requirements.txt" > "${TMP_REQ}"
pip install -r "${TMP_REQ}"
rm -f "${TMP_REQ}"

echo
echo "OLMo repository cloned to: ${REPO_DIR}"
echo "Virtual environment available at: ${ENV_DIR}"
echo "To activate: source ${ENV_DIR}/bin/activate"\necho "Run inference via: python inference/Olmo_2/run_from_snapshot.py --help"\n\n