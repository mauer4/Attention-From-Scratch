#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_DIR="${ROOT_DIR}/llm_repos/Olmo"
ENV_DIR="${1:-${ROOT_DIR}/.venv}"

log() {
  echo "[fetch-olmo] $*"
}

UPDATE_REPO=${FETCH_OLMO_UPDATE:-0}

mkdir -p "$(dirname "${REPO_DIR}")"

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  log "Cloning allenai/OLMo into ${REPO_DIR}."
  git clone --depth 1 https://github.com/allenai/OLMo.git "${REPO_DIR}"
elif [[ "${UPDATE_REPO}" == "1" ]]; then
  log "Updating existing repository at ${REPO_DIR} (FETCH_OLMO_UPDATE=1)."
  git -C "${REPO_DIR}" pull --ff-only
else
  log "Repository already present at ${REPO_DIR}; skipping git pull."
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
    log "Python environment missing at ${ENV_DIR}; run setup_env/create_venv.sh first."
    exit 1
  fi
  log "Activating Python environment at ${ENV_DIR}"
  source "${ENV_DIR}/bin/activate"
else
  log "Using active Python environment at ${VIRTUAL_ENV}"
fi

python -m pip install --no-deps -e "${REPO_DIR}"
TMP_REQ="$(mktemp "${REPO_DIR}/inference/requirements.XXXXXX.txt")"
sed -E 's|git+ssh://git@github.com/|git+https://github.com/|; /^(compression|efficiency)\//d' \
  "${REPO_DIR}/inference/requirements.txt" > "${TMP_REQ}"
python -m pip install --no-deps -r "${TMP_REQ}"
rm -f "${TMP_REQ}"

cat <<EOF

OLMo repository cloned to: ${REPO_DIR}
Virtual environment available at: ${ENV_DIR}
Activate it with: source ${ENV_DIR}/bin/activate
Repo CLI usage: see ${REPO_DIR}/README.md

EOF
