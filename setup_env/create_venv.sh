#!/usr/bin/env bash
# shellcheck disable=SC1091

# If this script is sourced, avoid changing the caller's shell options
IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  IS_SOURCED=1
fi

# Only set strict options when executed directly (not when sourced)
if [[ "${IS_SOURCED}" -eq 0 ]]; then
  set -euo pipefail
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

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

# Helper to fail safely: return when sourced, exit when executed
die() {
  local msg="$1"
  echo "${msg}" >&2
  if [[ "${IS_SOURCED}" -eq 1 ]]; then
    return 1
  else
    exit 1
  fi
}

CONFIG_EXPORTER="${ROOT_DIR}/setup_env/export_model_env.py"
PYTHON_BIN="python"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    die "❌ python or python3 command is required."
  fi
fi
if ! ENV_SETTINGS="$("${PYTHON_BIN}" "${CONFIG_EXPORTER}")"; then
  die "❌ Failed to load model configuration from config/config.yaml"
fi
eval "${ENV_SETTINGS}"
log "Project root: ${PROJECT_ROOT}"
log "Snapshot directory: ${MODEL_SNAPSHOT_DIR}"

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
    die "❌ python3 command not found."
  fi

  if [[ ! -d "${VENV_DIR}" ]]; then
    if command -v conda >/dev/null 2>&1; then
      echo "⚠️  Falling back to conda environment 'infer'."
      conda create -y -n infer python=3.12
      echo "✅ Activate with: conda activate infer"
      if [[ "${IS_SOURCED}" -eq 1 ]]; then
        return 0
      else
        exit 0
      fi
    else
      die "❌ Unable to create virtual environment. Install Python 3.12+ or conda."
    fi
  fi
fi

ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"
if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
  die "❌ Activation script missing at ${ACTIVATE_SCRIPT}"
fi

log "Activating ${VENV_DIR}"
source "${ACTIVATE_SCRIPT}"

echo "✅ Python: $(python --version 2>&1)"
echo "✅ pip: $(pip --version 2>&1)"

if [[ "${IS_SOURCED}" -eq 0 ]]; then
  echo "⚠️  This script was executed, not sourced. Activate manually via 'source .venv/bin/activate' for interactive use."
fi
