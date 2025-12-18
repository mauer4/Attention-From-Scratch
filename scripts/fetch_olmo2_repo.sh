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
FETCH_INFO="${REPO_DIR}/.fetch_info.json"

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
    ACTION="cloned"
  elif [[ "${UPDATE_REPO}" == "1" ]]; then
    log "Updating existing repository at ${REPO_DIR} (FETCH_OLMO_UPDATE=1)."
    if ! git -C "${REPO_DIR}" pull --ff-only; then
      die "❌ git pull failed." || return 1
    fi
    ACTION="updated"
  else
    log "Repository already present at ${REPO_DIR}; skipping git pull."
    ACTION="present"
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

  # Record fetch/update status and repository version information so callers can
  # make informed decisions (install_all.sh will consume this file).
  if command -v git >/dev/null 2>&1; then
    LOCAL_COMMIT="$(git -C "${REPO_DIR}" rev-parse --verify HEAD 2>/dev/null || true)"
    # Try to query the origin for the most recent HEAD commit.
    REMOTE_COMMIT="$(git -C "${REPO_DIR}" ls-remote origin HEAD 2>/dev/null | awk '{print $1}' || true)"
    if [[ -z "${REMOTE_COMMIT}" ]]; then
      REMOTE_COMMIT="$(git ls-remote https://github.com/allenai/OLMo.git HEAD 2>/dev/null | awk '{print $1}' || true)"
    fi
    TAG="$(git -C "${REPO_DIR}" describe --tags --exact-match HEAD 2>/dev/null || true)"
    REMOTE_URL="$(git -C "${REPO_DIR}" remote get-url origin 2>/dev/null || echo 'https://github.com/allenai/OLMo.git')"

    # If we only found the repo present and didn't explicitly update, mark as
    # up-to-date when local == remote to avoid unnecessary pulls.
    if [[ "${ACTION}" == "present" ]]; then
      if [[ -n "${LOCAL_COMMIT}" && -n "${REMOTE_COMMIT}" && "${LOCAL_COMMIT}" == "${REMOTE_COMMIT}" ]]; then
        ACTION="up-to-date"
      else
        ACTION="outdated"
      fi
    fi

    # Write a small JSON summary so callers (and CI) can inspect status.
    if [[ -n "${FETCH_INFO}" ]]; then
      python - <<PY > "${FETCH_INFO}.tmp"
import json, time, os
info = {
    'action': os.environ.get('ACTION', ''),
    'local_commit': os.environ.get('LOCAL_COMMIT', ''),
    'remote_commit': os.environ.get('REMOTE_COMMIT', ''),
    'tag': os.environ.get('TAG', ''),
    'remote_url': os.environ.get('REMOTE_URL', ''),
    'timestamp': time.time(),
}
print(json.dumps(info, indent=2))
PY
      # Atomically move into place.
      mv "${FETCH_INFO}.tmp" "${FETCH_INFO}" 2>/dev/null || true
    fi
  fi
}

main "$@"
status=$?
if [[ "${IS_SOURCED}" -eq 1 ]]; then
  return "${status}"
fi
exit "${status}"
