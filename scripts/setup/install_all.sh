#!/usr/bin/env bash
# shellcheck disable=SC1091
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )/../.." && pwd)"
REPORTS_DIR="${ROOT_DIR}/reports"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_BIN="${VENV_DIR}/bin/python"
PIP_BIN="${VENV_DIR}/bin/pip"

mkdir -p "${REPORTS_DIR}"

step() {
  printf '\n%s\n' "$1"
}

overall_status=0

step "⚙️  Step 1: Prepare virtual environment"
bash "${ROOT_DIR}/scripts/setup/create_venv.sh"

step "⚙️  Step 2: Install Python dependencies"
"${PIP_BIN}" install -r "${ROOT_DIR}/requirements.txt"

step "⚙️  Step 3: Load model configuration"
CONFIG_EXPORTER="${ROOT_DIR}/scripts/setup/export_model_env.py"

if ! ENV_SETTINGS="$("${PYTHON_BIN}" "${CONFIG_EXPORTER}")"; then
  echo "❌ Failed to load model configuration from config/config.yaml" >&2
  exit 1
fi
eval "${ENV_SETTINGS}"
echo "✅ Model configuration: ${MODEL_NAME} → ${MODEL_REPO_ID}"
echo "✅ Snapshot directory: ${MODEL_SNAPSHOT_DIR}"

step "⚙️  Step 4: GPU and driver probe"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/setup/check_gpu.py"

step "⚙️  Step 5: Clone AllenAI OLMo repository"
bash "${ROOT_DIR}/scripts/setup/fetch_olmo2_repo.sh" "${VENV_DIR}"

step "⚙️  Step 6: Validate configuration"
if "${PYTHON_BIN}" "${ROOT_DIR}/scripts/setup/verify_config.py"; then
  echo "✅ Configuration validated"
else
  echo "⚠️  Configuration has warnings (see reports/config_report.json)"
  echo "   → Run 'python scripts/utils/download_weights.py --model-name olmo2' if assets are missing."
  overall_status=1
fi

step "⚙️  Step 7: Run sanity inference"
if "${PYTHON_BIN}" "${ROOT_DIR}/scripts/inference/test_inference.py"; then
  echo "✅ Sanity inference executed"
else
  echo "⚠️  Sanity inference recorded issues (see reports/test_summary.json)"
  overall_status=1
fi

step "⚙️  Step 8: Generate environment report"
"${PYTHON_BIN}" "${ROOT_DIR}/scripts/setup/run_env_report.py"

if [[ "${overall_status}" -eq 0 ]]; then
  cat <<SUMMARY

✔ GPU probe complete (reports/system_gpu.json)
✔ .venv ready at .venv/
✔ Dependencies installed & requirements.txt refreshed
✔ CUDA version recorded in config/config.yaml
✔ Sanity summary written to reports/test_summary.json
✔ Environment report at reports/environment_report.md
→ Ready for inference and benchmarking
SUMMARY
else
  cat <<SUMMARY

⚠️  Setup finished with warnings
• Review reports/config_report.json and reports/test_summary.json for details.
• Run 'python scripts/download_weights.py --model-name olmo2' to stage assets if needed.
• Environment report (reports/environment_report.md) captures the warnings above.
SUMMARY
  exit 1
fi
