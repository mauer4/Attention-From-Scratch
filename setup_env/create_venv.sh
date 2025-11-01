#!/usr/bin/env bash
# shellcheck disable=SC1091
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  IS_SOURCED=1
fi

detect_platform() {
  local uname_out
  uname_out="$(uname -s)"
  case "${uname_out}" in
    Linux*)
      if [[ -f /etc/wsl.conf || -n "${WSL_DISTRO_NAME:-}" ]]; then
        printf "wsl"
      else
        printf "linux"
      fi
      ;;
    Darwin*)
      printf "macos"
      ;;
    *)
      printf "unknown"
      ;;
  esac
}

log() {
  printf "[create_venv] %s\n" "$*"
}

platform="$(detect_platform)"
log "Detected host platform: ${platform}"

if [[ -f /etc/vastai-release || -d /run/nvidia ]]; then
  log "Running inside a Vast.ai-style container (no nested Docker assumed)."
fi

if [[ -d "${VENV_DIR}" ]]; then
  echo "✅ Virtual environment found at ${VENV_DIR}"
else
  echo "⚙️  Creating virtual environment at ${VENV_DIR}"
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv "${VENV_DIR}" || {
      echo "⚠️  python3 -m venv failed; attempting fallback." >&2
    }
  else
    echo "❌ python3 command not found." >&2
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    if command -v conda >/dev/null 2>&1; then
      echo "⚠️  Falling back to conda environment 'infer'."
      conda create -y -n infer python=3.12
      echo "✅ Activate with: conda activate infer"
      exit 0
    else
      echo "❌ Unable to create virtual environment. Install Python 3.12+ or conda." >&2
      exit 1
    fi
  fi
fi

ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"
if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
  echo "❌ Activation script missing at ${ACTIVATE_SCRIPT}" >&2
  exit 1
fi

log "Activating ${VENV_DIR}"
source "${ACTIVATE_SCRIPT}"

echo "✅ Python: $(python --version 2>&1)"
echo "✅ pip: $(pip --version 2>&1)"

if [[ "${IS_SOURCED}" -eq 0 ]]; then
  echo "⚠️  This script was executed, not sourced. Activate manually via 'source .venv/bin/activate' for interactive use."
fi
