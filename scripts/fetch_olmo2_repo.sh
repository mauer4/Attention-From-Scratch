#!/usr/bin/env bash

# Detect whether sourced to avoid exiting the caller's shell
IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  IS_SOURCED=1
fi

# Only set pipefail when executed directly (avoid altering caller options)
if [[ "${IS_SOURCED}" -eq 0 ]]; then
  set -o pipefail
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_DIR="${ROOT_DIR}/llm_repos/Olmo"
ENV_DIR="${1:-${ROOT_DIR}/.venv}"

log() {
  echo "[fetch-olmo] $*"
}

die() {
  echo "[fetch-olmo] $*" >&2
  return 1
}

main() {
  UPDATE_REPO=${FETCH_OLMO_UPDATE:-0}

  mkdir -p "$(dirname "${REPO_DIR}")"

  ensure_repo() {
    if [[ -f "${REPO_DIR}/pyproject.toml" || -f "${REPO_DIR}/setup.py" ]]; then
      return 0
    fi
    log "Repository at ${REPO_DIR} looks incomplete; re-cloning."
    rm -rf "${REPO_DIR}"
    git clone --depth 1 https://github.com/allenai/OLMo.git "${REPO_DIR}"
  }

  if [[ ! -d "${REPO_DIR}/.git" ]]; then
    log "Cloning allenai/OLMo into ${REPO_DIR}."
    if ! git clone --depth 1 https://github.com/allenai/OLMo.git "${REPO_DIR}"; then
      die "❌ Failed to clone repository." || return 1
    fi
  elif [[ "${UPDATE_REPO}" == "1" ]]; then
    log "Updating existing repository at ${REPO_DIR} (FETCH_OLMO_UPDATE=1)."
    if ! git -C "${REPO_DIR}" pull --ff-only; then
      die "❌ git pull failed." || return 1
    fi
  else
    log "Repository already present at ${REPO_DIR}; skipping git pull."
  fi

  if ! ensure_repo; then
    die "❌ Repository re-clone failed; cannot proceed." || return 1
  fi

  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    if [[ ! -f "${ENV_DIR}/bin/activate" ]]; then
      die "Python environment missing at ${ENV_DIR}; run setup_env/create_venv.sh first." || return 1
    fi
    log "Activating Python environment at ${ENV_DIR}"
    # shellcheck disable=SC1090
    if ! source "${ENV_DIR}/bin/activate"; then
      die "❌ Failed to activate environment at ${ENV_DIR}" || return 1
    fi
  else
    log "Using active Python environment at ${VIRTUAL_ENV}"
  fi

  if ! python -m pip install --no-deps -e "${REPO_DIR}"; then
    die "❌ Failed to install OLMo in editable mode." || return 1
  fi
  TMP_REQ="$(mktemp "${REPO_DIR}/inference/requirements.XXXXXX.txt")"
  # Normalize any ssh-based Git URLs to https to avoid auth prompts
  SRC_REQ="${REPO_DIR}/inference/requirements.txt"
  SRC_REQ="${SRC_REQ}" DST_REQ="${TMP_REQ}" python - <<'PY'
import os, re, pathlib

src = pathlib.Path(os.environ["SRC_REQ"])
dst = pathlib.Path(os.environ["DST_REQ"])
text = src.read_text()
text = re.sub(r"git\+ssh://[^\s]*github.com/", "git+https://github.com/", text)
lines = [line for line in text.splitlines() if not line.startswith(("compression/", "efficiency/"))]
dst.write_text("\n".join(lines) + "\n")
PY
  if ! python -m pip install --no-deps -r "${TMP_REQ}"; then
    rm -f "${TMP_REQ}"
    die "❌ Failed to install inference requirements." || return 1
  fi
  rm -f "${TMP_REQ}"

  cat <<EOF

OLMo repository cloned to: ${REPO_DIR}
Virtual environment available at: ${ENV_DIR}
Activate it with: source ${ENV_DIR}/bin/activate
Repo CLI usage: see ${REPO_DIR}/README.md

EOF
}

main "$@"
status=$?
if [[ "${IS_SOURCED}" -eq 1 ]]; then
  return "${status}"
fi
exit "${status}"
