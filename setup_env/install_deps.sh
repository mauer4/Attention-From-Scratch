#!/usr/bin/env bash
# shellcheck disable=SC1091

# Detect whether sourced to avoid exiting the caller's shell
IS_SOURCED=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  IS_SOURCED=1
fi

# Only set pipefail when executed directly (avoid altering caller options)
if [[ "${IS_SOURCED}" -eq 0 ]]; then
  set -o pipefail
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQUIREMENTS_FILE="${ROOT_DIR}/requirements.txt"
LOCK_FILE="${ROOT_DIR}/requirements.lock"
REPORTS_DIR="${ROOT_DIR}/reports"
mkdir -p "${REPORTS_DIR}"

die() {
  echo "$*" >&2
  return 1
}

main() {
  if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    die "❌ requirements.txt missing at ${REQUIREMENTS_FILE}" || return 1
  fi

  if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"
    if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
      die "❌ Virtual environment missing. Run setup_env/create_venv.sh first." || return 1
    fi
    # shellcheck disable=SC1090
    if ! source "${ACTIVATE_SCRIPT}"; then
      die "❌ Failed to activate environment at ${VENV_DIR}" || return 1
    fi
  fi

  echo "⚙️  Upgrading pip tooling"
  if ! python -m pip install --upgrade pip wheel setuptools >/dev/null; then
    die "❌ Failed to upgrade pip tooling." || return 1
  fi

  echo "⚙️  Installing dependencies from ${REQUIREMENTS_FILE}"
  if ! pip install -r "${REQUIREMENTS_FILE}" >/dev/null; then
    die "❌ Failed to install dependencies from ${REQUIREMENTS_FILE}." || return 1
  fi

  echo "⚙️  Installing project in editable mode"
  if ! pip install -e "${ROOT_DIR}" >/dev/null; then
    die "❌ Failed to install project in editable mode." || return 1
  fi

  VERIFY_JSON="$(python "${ROOT_DIR}/setup_env/verify_install.py" --json)"
  needs_reinstall=$(VERIFY_JSON="${VERIFY_JSON}" python - <<'PY'
import json
import os

report = json.loads(os.environ["VERIFY_JSON"])
print("yes" if report.get("needs_reinstall") else "no")
PY
)

  if [[ "${needs_reinstall}" == "yes" ]]; then
    index_url=$(VERIFY_JSON="${VERIFY_JSON}" python - <<'PY'
import json
import os

report = json.loads(os.environ["VERIFY_JSON"])
print(report.get("suggested_index_url") or "")
PY
)
    if [[ -n "${index_url}" ]]; then
      echo "⚙️  Reinstalling torch stack from ${index_url}"
      pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
      if ! pip install --upgrade torch torchvision torchaudio --index-url "${index_url}" >/dev/null; then
        die "❌ Failed to reinstall torch stack from ${index_url}." || return 1
      fi
      VERIFY_JSON="$(python "${ROOT_DIR}/setup_env/verify_install.py" --json)"
    else
      echo "⚠️  Torch/CUDA mismatch detected; review reports/verify_install.json." >&2
    fi
  fi

  echo "⚙️  Freezing dependencies"
  if ! pip freeze | sort > "${LOCK_FILE}"; then
    die "❌ Failed to write lock file at ${LOCK_FILE}." || return 1
  fi

  if ! command -v pipdeptree >/dev/null 2>&1; then
    echo "⚙️  Installing pipdeptree"
    if ! pip install pipdeptree >/dev/null; then
      die "❌ Failed to install pipdeptree." || return 1
    fi
  fi

  if ! pipdeptree --freeze > "${REPORTS_DIR}/pip_tree.txt"; then
    die "❌ Failed to generate dependency tree." || return 1
  fi

  if ! python - <<'PY'
missing = []
for module in (
    "torch",
    "transformers",
    "safetensors",
    "inference.engine",
    "custom_engine.core",
    "analysis.benchmark_runtime",
    "model_env",
):
    try:
        __import__(module)
    except ModuleNotFoundError:
        missing.append(module)

if missing:
    raise SystemExit(f"Missing required modules: {', '.join(missing)}")
PY
  then
    die "❌ Missing required modules (see message above)." || return 1
  fi

  if ! python "${ROOT_DIR}/setup_env/verify_install.py"; then
    die "❌ verify_install.py reported issues." || return 1
  fi

  echo "✅ Dependencies installed and verified"
  echo "✅ requirements.lock updated"
  echo "✅ reports/pip_tree.txt refreshed"
}

main "$@"
status=$?
if [[ "${IS_SOURCED}" -eq 1 ]]; then
  return "${status}"
fi
exit "${status}"
