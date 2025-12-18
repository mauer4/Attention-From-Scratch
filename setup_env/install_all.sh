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
REPORTS_DIR="${ROOT_DIR}/reports"
mkdir -p "${REPORTS_DIR}"

step() {
  printf '\n%s\n' "$1"
}

# Print error and return non-zero so sourcing users keep their shell
die() {
  echo "$*" >&2
  return 1
}

main() {
  overall_status=0

  step "⚙️  Step 0: Load model configuration"
  CONFIG_EXPORTER="${ROOT_DIR}/setup_env/export_model_env.py"
  PYTHON_BIN="python"
  if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="python3"
    else
      die "❌ python or python3 command is required." || return 1
    fi
  fi
  if ! ENV_SETTINGS="$("${PYTHON_BIN}" "${CONFIG_EXPORTER}")"; then
    die "❌ Failed to load model configuration from config/config.yaml" || return 1
  fi
  eval "${ENV_SETTINGS}"
  echo "✅ Model configuration: ${MODEL_NAME} → ${MODEL_REPO_ID}"
  echo "✅ Snapshot directory: ${MODEL_SNAPSHOT_DIR}"

  step "⚙️  Step 1: GPU and driver probe"
  if ! python "${ROOT_DIR}/setup_env/check_gpu.py"; then
    echo "⚠️  GPU probe reported issues (see above)." >&2
    overall_status=1
  fi
  if [[ -f "${ROOT_DIR}/.env.autodetected" ]]; then
    CUDA_VERSION_VALUE="$(grep '^CUDA_VERSION=' "${ROOT_DIR}/.env.autodetected" | cut -d'=' -f2)"
    if [[ -n "${CUDA_VERSION_VALUE}" ]]; then
      export CUDA_VERSION="${CUDA_VERSION_VALUE}"
      echo "✅ Loaded CUDA version ${CUDA_VERSION_VALUE} from .env.autodetected"
    fi
  fi

  step "⚙️  Step 2: Prepare virtual environment"
  if ! source "${ROOT_DIR}/setup_env/create_venv.sh"; then
    echo "⚠️  Virtual environment setup encountered issues." >&2
    overall_status=1
  fi

  step "⚙️  Step 3: Clone AllenAI OLMo repository"
  if ! bash "${ROOT_DIR}/scripts/fetch_olmo2_repo.sh" "${ROOT_DIR}/.venv"; then
    echo "⚠️  OLMo repository setup encountered issues." >&2
    overall_status=1
  fi

  step "⚙️  Step 4: Install Python dependencies"
  if ! bash "${ROOT_DIR}/setup_env/install_deps.sh"; then
    echo "⚠️  Dependency installation encountered issues." >&2
    overall_status=1
  fi

  step "⚙️  Step 5: Validate configuration"
  if python "${ROOT_DIR}/setup_env/verify_config.py"; then
    echo "✅ Configuration validated"
  else
    echo "⚠️  Configuration has warnings (see reports/config_report.json)"
    echo "   → Run 'python scripts/download_weights.py --model-name olmo2' if assets are missing."
    overall_status=1
  fi

  step "⚙️  Step 6: Run sanity inference"
  if python "${ROOT_DIR}/inference/Olmo_2/test_inference.py"; then
    echo "✅ Sanity inference executed"
  else
    echo "⚠️  Sanity inference recorded issues (see reports/test_summary.json)"
    overall_status=1
  fi

  step "⚙️  Step 7: Generate environment report"
  if ! python "${ROOT_DIR}/setup_env/run_env_report.py"; then
    echo "⚠️  Environment report generation encountered issues." >&2
    overall_status=1
  fi

  if [[ "${overall_status}" -eq 0 ]]; then
    cat <<SUMMARY

✔ GPU probe complete (reports/system_gpu.json)
✔ .venv ready at .venv/
✔ Dependencies installed & requirements.lock refreshed
✔ CUDA version recorded in .env.autodetected
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
    return 1
  fi
}

main "$@"
status=$?
if [[ "${IS_SOURCED}" -eq 1 ]]; then
  return "${status}"
fi
exit "${status}"
