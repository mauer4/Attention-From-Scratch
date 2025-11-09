ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

source ${VENV_DIR}/bin/activate
export HF_HUB_OFFLINE=1