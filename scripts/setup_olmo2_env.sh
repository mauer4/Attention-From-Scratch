#!/usr/bin/env bash
set -euo pipefail

# Wrapper around setup/bare_metal_setup.sh with a small CUDA sanity check.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REQUESTED_ENV="${1:-.venv-olmo2}"
ENV_DIR="$(python3 - <<PY
import os, sys
root = sys.argv[1]
requested = sys.argv[2]
print(os.path.abspath(os.path.join(root, requested)))
PY
"${ROOT_DIR}" "${REQUESTED_ENV}")"

bash "${ROOT_DIR}/setup/bare_metal_setup.sh" "${ENV_DIR}"

# shellcheck disable=SC1091
source "${ENV_DIR}/bin/activate"

python - <<'PY'
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover - defensive guard
    raise SystemExit(f"Failed to import torch after installation: {exc}") from exc

if not torch.cuda.is_available():
    raise SystemExit("The installed torch build does not detect CUDA. Check driver/toolkit compatibility.")

device_capability = torch.cuda.get_device_capability()
arch = f"sm_{device_capability[0]}{device_capability[1]}"
compiled_arches = {arch_name.replace('+PTX', '') for arch_name in torch.cuda.get_arch_list()}
if arch not in compiled_arches:
    raise SystemExit(
        f"Torch {torch.__version__} is missing kernels for {arch}. "
        "Install a build that targets your GPU architecture (set TORCH_INDEX_URL) or build from source."
    )

print(f"CUDA capability check passed for {arch} with torch {torch.__version__}.")
PY

echo
printf 'Olmo 2 environment ready at %s\n' "${ENV_DIR}"
echo "Activate it with: source ${ENV_DIR}/bin/activate"
echo "Then run inference via: python inference/Olmo_2/run_from_snapshot.py --help"\n\n\n