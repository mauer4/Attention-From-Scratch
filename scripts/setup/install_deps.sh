#!/usr/bin/env bash
# shellcheck disable=SC1091
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
REQUIREMENTS_FILE="${ROOT_DIR}/requirements.txt"
LOCK_FILE="${ROOT_DIR}/requirements.lock"
REPORTS_DIR="${ROOT_DIR}/reports"
mkdir -p "${REPORTS_DIR}"

if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
  echo "❌ requirements.txt missing at ${REQUIREMENTS_FILE}" >&2
  exit 1
fi

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  ACTIVATE_SCRIPT="${VENV_DIR}/bin/activate"
  if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
    echo "❌ Virtual environment missing. Run setup_env/create_venv.sh first." >&2
    exit 1
  fi
  source "${ACTIVATE_SCRIPT}"
fi

echo "⚙️  Upgrading pip tooling"
python -m pip install --upgrade pip wheel setuptools >/dev/null

echo "⚙️  Installing dependencies from ${REQUIREMENTS_FILE}"
pip install -r "${REQUIREMENTS_FILE}" >/dev/null

echo "⚙️  Installing project in editable mode"
pip install -e "${ROOT_DIR}" >/dev/null

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
    pip install --upgrade torch torchvision torchaudio --index-url "${index_url}" >/dev/null
    VERIFY_JSON="$(python "${ROOT_DIR}/setup_env/verify_install.py" --json)"
  else
    echo "⚠️  Torch/CUDA mismatch detected; review reports/verify_install.json." >&2
  fi
fi

echo "⚙️  Freezing dependencies"
pip freeze | sort > "${LOCK_FILE}"

if ! command -v pipdeptree >/dev/null 2>&1; then
  echo "⚙️  Installing pipdeptree"
  pip install pipdeptree >/dev/null
fi

pipdeptree --freeze > "${REPORTS_DIR}/pip_tree.txt"

python - <<'PY'
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

python "${ROOT_DIR}/setup_env/verify_install.py"

echo "✅ Dependencies installed and verified"
echo "✅ requirements.lock updated"
echo "✅ reports/pip_tree.txt refreshed"
