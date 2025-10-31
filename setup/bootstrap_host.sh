#!/usr/bin/env bash
set -euo pipefail

# Unified host bootstrapper for Attention-From-Scratch.
# - Installs optional system packages (apt/dnf/pacman/brew).
# - Optionally provisions a CUDA toolkit aligned with the detected driver.
# - Creates/updates a Python virtual environment and installs locked deps.
# - Can clone optional third-party sources (e.g., CUTLASS) on demand.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/bootstrap_$(date +%Y%m%d-%H%M%S).log"
exec > >(tee -a "${LOG_FILE}")
exec 2>&1
echo "[bootstrap] Logging to ${LOG_FILE}"
DEFAULT_ENV="${ROOT_DIR}/.venv-olmo2"

SYSTEM_MODE="auto"
REQUESTED_ENV="${DEFAULT_ENV}"
CUDA_SELECTION="auto"
EXTRAS="olmo"
LOCK_FILE=""
INCLUDE_CUTLASS="${ATTN_INCLUDE_CUTLASS:-0}"

usage() {
  cat <<'USAGE'
Usage: setup/bootstrap_host.sh [options]

Options:
  --system <auto|apt|dnf|yum|pacman|brew|none>
      Package manager to use for system prerequisites (default: auto).
  --python-env <path>
      Virtual environment location (default: .venv-olmo2 at repo root).
  --cuda-toolkit <auto|none|<major.minor>>
      Install CUDA toolkit matching detected driver or specified version.
  --extras <name[,name...]>
      Project extras to install (default: olmo).
  --lock <path>
      Override lockfile path (defaults to requirements/locks/<extras>.lock when present).
  --with-cutlass | --without-cutlass
      Opt in/out of cloning NVIDIA CUTLASS under third_party/ (default: off).
  -h, --help
      Show this help message and exit.

Environment overrides:
  SKIP_SYSTEM_PACKAGES=1        Skip system package installation regardless of --system.
  SKIP_CUDA_TOOLKIT=1           Skip CUDA toolkit installation.
  TORCH_INDEX_URL=<url>         Force PyTorch wheel index (overrides auto detection).
  ATTN_INCLUDE_CUTLASS=1        Same as passing --with-cutlass.

Examples:
  # Full bootstrap on an Ubuntu workstation
  bash setup/bootstrap_host.sh

  # Reuse existing system packages and install into a custom venv without CUDA
  SKIP_SYSTEM_PACKAGES=1 bash setup/bootstrap_host.sh --python-env ~/envs/olmo --cuda-toolkit none

USAGE
}

log() {
  echo "[bootstrap] $*"
}

warn() {
  echo "[bootstrap][warn] $*" >&2
}

die() {
  echo "[bootstrap][error] $*" >&2
  exit 1
}

resolve_path() {
  python3 - <<'PY' "$1" "$2"
import os, sys
base, rel = sys.argv[1], sys.argv[2]
if os.path.isabs(rel):
    print(os.path.abspath(rel))
else:
    print(os.path.abspath(os.path.join(base, rel)))
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --system)
      [[ $# -lt 2 ]] && die "--system requires a value"
      SYSTEM_MODE="$2"
      shift 2
      ;;
    --python-env)
      [[ $# -lt 2 ]] && die "--python-env requires a value"
      REQUESTED_ENV="$(resolve_path "${PWD}" "$2")"
      shift 2
      ;;
    --cuda-toolkit)
      [[ $# -lt 2 ]] && die "--cuda-toolkit requires a value"
      CUDA_SELECTION="$2"
      shift 2
      ;;
    --extras)
      [[ $# -lt 2 ]] && die "--extras requires a value"
      EXTRAS="$2"
      shift 2
      ;;
    --lock)
      [[ $# -lt 2 ]] && die "--lock requires a value"
      LOCK_FILE="$(resolve_path "${PWD}" "$2")"
      shift 2
      ;;
    --with-cutlass)
      INCLUDE_CUTLASS=1
      shift
      ;;
    --without-cutlass)
      INCLUDE_CUTLASS=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "Unknown argument: $1"
      ;;
  esac
done

if [[ -z "${LOCK_FILE}" ]]; then
  extras_key="${EXTRAS//,/-}"
  candidate="${ROOT_DIR}/requirements/locks/${extras_key}.lock"
  if [[ -f "${candidate}" ]]; then
    LOCK_FILE="${candidate}"
  else
    LOCK_FILE="${ROOT_DIR}/requirements/locks/olmo.lock"
    warn "Lockfile for extras='${EXTRAS}' not found; using ${LOCK_FILE}"
  fi
fi

if [[ ! -f "${LOCK_FILE}" ]]; then
  die "Lockfile ${LOCK_FILE} not found. Generate it with pip-compile or uv."
fi

if [[ "${INCLUDE_CUTLASS}" != "0" && "${INCLUDE_CUTLASS}" != "1" ]]; then
  die "ATTN_INCLUDE_CUTLASS/--with-cutlass expects 0 or 1"
fi

if [[ "${SYSTEM_MODE}" == "auto" ]]; then
  if command -v apt-get >/dev/null 2>&1; then
    SYSTEM_MODE="apt"
  elif command -v dnf >/dev/null 2>&1; then
    SYSTEM_MODE="dnf"
  elif command -v yum >/dev/null 2>&1; then
    SYSTEM_MODE="yum"
  elif command -v pacman >/dev/null 2>&1; then
    SYSTEM_MODE="pacman"
  elif [[ "$(uname -s)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    SYSTEM_MODE="brew"
  else
    SYSTEM_MODE="none"
  fi
fi

version_gte() {
  local left="$1" right="$2"
  [[ "${left}" == "${right}" ]] && return 0
  if [[ "$(printf '%s\n%s\n' "${right}" "${left}" | sort -V | tail -n1)" == "${left}" ]]; then
    return 0
  fi
  return 1
}

detect_driver_version() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local version
    version="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
    if [[ -n "${version}" ]]; then
      echo "${version}"
      return 0
    fi
  fi
  return 1
}

choose_cuda_version() {
  local driver="$1"
  local -a matrix=(
    "560.35.03:12.8"
    "555.42.02:12.5"
    "550.40.07:12.4"
    "545.23.08:12.3"
    "535.54.03:12.2"
    "525.60.13:12.1"
    "515.65.01:11.7"
    "510.47.03:11.6"
    "470.57.02:11.4"
  )
  local default="12.8"
  for entry in "${matrix[@]}"; do
    local min="${entry%%:*}"
    local cuda="${entry##*:}"
    if version_gte "${driver}" "${min}"; then
      echo "${cuda}"
      return
    fi
  done
  echo "${default}"
}

require_sudo() {
  if [[ "${EUID}" -eq 0 ]]; then
    "$@"
  elif command -v sudo >/dev/null 2>&1; then
    sudo "$@"
  else
    die "Need root privileges to run: $*"
  fi
}

install_system_packages() {
  if [[ "${SYSTEM_MODE}" == "none" ]]; then
    log "Skipping system packages (--system none)."
    return
  fi
  if [[ "${SKIP_SYSTEM_PACKAGES:-0}" == "1" ]]; then
    log "SKIP_SYSTEM_PACKAGES set; skipping system packages."
    return
  fi

  log "Installing system packages via ${SYSTEM_MODE}."
  case "${SYSTEM_MODE}" in
    apt)
      local packages=(
        build-essential
        git
        cmake
        ninja-build
        python3
        python3-dev
        python3-pip
        python3-venv
        clang
        curl
        ca-certificates
        pkg-config
        unzip
      )
      require_sudo apt-get update
      require_sudo apt-get install -y --no-install-recommends "${packages[@]}"
      ;;
    dnf|yum)
      local packages=(
        @development-tools
        git
        cmake
        ninja-build
        python3
        python3-devel
        python3-pip
        clang
        curl
        ca-certificates
      )
      require_sudo "${SYSTEM_MODE}" install -y "${packages[@]}"
      ;;
    pacman)
      local packages=(
        base-devel
        git
        cmake
        ninja
        python
        python-pip
        python-virtualenv
        clang
        curl
        ca-certificates
      )
      require_sudo pacman -Sy --noconfirm "${packages[@]}"
      ;;
    brew)
      brew update
      brew install cmake ninja python git llvm curl
      ;;
    *)
      warn "Unsupported package manager '${SYSTEM_MODE}'. Install prerequisites manually."
      ;;
  esac
}

ensure_cuda_repo_apt() {
  local version="$1"
  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    warn "Neither curl nor wget installed; cannot configure CUDA repository."
    return 1
  fi
  if [[ ! -f /etc/os-release ]]; then
    warn "Cannot determine distribution to configure CUDA repository."
    return 1
  fi
  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || -z "${VERSION_ID:-}" ]]; then
    warn "CUDA repository helper currently supports Ubuntu only."
    return 1
  fi
  local version_nodot="${VERSION_ID//./}"
  local repo_id="ubuntu${version_nodot}"
  local keyring_pkg="cuda-keyring_1.1-1_all.deb"
  local url="https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/x86_64/${keyring_pkg}"
  local tmp_deb="/tmp/${keyring_pkg}"
  log "Configuring NVIDIA CUDA APT repository for ${repo_id}."
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "${url}" -o "${tmp_deb}"
  else
    wget -qO "${tmp_deb}" "${url}"
  fi
  require_sudo dpkg -i "${tmp_deb}"
  rm -f "${tmp_deb}"
  local repo_line="deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/${repo_id}/x86_64/ /"
  local list_dir="/etc/apt/sources.list.d"
  require_sudo mkdir -p "${list_dir}"
  for file in "${list_dir}"/*.list; do
    if [[ -f "${file}" ]] && grep -q "developer.download.nvidia.com/compute/cuda" "${file}"; then
      require_sudo rm -f "${file}"
    fi
  done
  echo "${repo_line}" | require_sudo tee "${list_dir}/cuda-${repo_id}.list" >/dev/null
  require_sudo apt-get update
}

install_cuda_toolkit() {
  if [[ "${CUDA_SELECTION}" == "none" || "${SKIP_CUDA_TOOLKIT:-0}" == "1" ]]; then
    log "Skipping CUDA toolkit installation."
    return
  fi

  local desired_version=""
  if [[ "${CUDA_SELECTION}" == "auto" ]]; then
    if driver_version="$(detect_driver_version)"; then
      desired_version="$(choose_cuda_version "${driver_version}")"
      log "Detected NVIDIA driver ${driver_version}; targeting CUDA ${desired_version}."
    else
      desired_version="12.8"
      warn "Unable to detect NVIDIA driver; defaulting CUDA toolkit to ${desired_version}."
    fi
  else
    desired_version="${CUDA_SELECTION}"
  fi

  if [[ "${SYSTEM_MODE}" != "apt" ]]; then
    warn "Automatic CUDA toolkit install currently supported only for apt-based systems."
    warn "Install CUDA ${desired_version} manually or set SKIP_CUDA_TOOLKIT=1."
    return
  fi

  local pkg_version="${desired_version//./-}"
  local toolkit_pkg="cuda-toolkit-${pkg_version}"
  local nsight_pkg="cuda-nsight-systems-${pkg_version}"
  if dpkg -s "${toolkit_pkg}" >/dev/null 2>&1; then
    log "CUDA toolkit ${desired_version} already installed."
    return
  fi

  ensure_cuda_repo_apt "${desired_version}" || warn "Continuing without ensuring CUDA repository."
  log "Installing ${toolkit_pkg} and ${nsight_pkg}."
  require_sudo apt-get install -y --no-install-recommends "${toolkit_pkg}" "${nsight_pkg}"
}

detect_compute_capability() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cap
    cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 | tr -d '[:space:]')"
    if [[ "${cap}" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
      echo "${BASH_REMATCH[1]}.${BASH_REMATCH[2]}"
      return 0
    fi
  fi
  return 1
}

select_torch_index() {
  if [[ -n "${TORCH_INDEX_URL:-}" ]]; then
    echo "${TORCH_INDEX_URL}"
    return
  fi
  local chosen_cuda=""
  if [[ "${CUDA_SELECTION}" == "auto" ]]; then
    if driver_version="$(detect_driver_version)"; then
      chosen_cuda="$(choose_cuda_version "${driver_version}")"
    fi
  elif [[ "${CUDA_SELECTION}" != "none" ]]; then
    chosen_cuda="${CUDA_SELECTION}"
  fi

  local capability
  capability="$(detect_compute_capability || true)"
  if [[ -n "${capability}" ]]; then
    log "Detected GPU compute capability ${capability}."
  fi

  case "${chosen_cuda}" in
    12.*) echo "https://download.pytorch.org/whl/cu121" ;;
    11.8) echo "https://download.pytorch.org/whl/cu118" ;;
    ""|none) echo "" ;;
    *) echo "https://download.pytorch.org/whl/cu121" ;;
  esac
}

install_python_dependencies() {
  local env_path="$1"
  local venv_python="${env_path}/bin/python"
  local venv_pip="${env_path}/bin/pip"

  if [[ ! -x "${venv_python}" ]]; then
    die "Python venv missing at ${env_path}; expected ${venv_python}."
  fi

  "${venv_python}" -m pip install --upgrade pip setuptools wheel

  local -a pip_args=("--requirement" "${LOCK_FILE}")
  local torch_index
  torch_index="$(select_torch_index)"
  if [[ -n "${torch_index}" ]]; then
    pip_args=("--extra-index-url" "${torch_index}" "${pip_args[@]}")
    log "Using PyTorch wheel index: ${torch_index}"
  else
    warn "Proceeding without a dedicated PyTorch index; CPU wheels may be installed."
  fi

  "${venv_pip}" install "${pip_args[@]}"
}

ensure_python_env() {
  local env_path="$1"
  if [[ -d "${env_path}" ]]; then
    log "Reusing existing virtual environment at ${env_path}."
  else
    log "Creating virtual environment at ${env_path}."
    python3 -m venv "${env_path}"
  fi
}

maybe_clone_cutlass() {
  if [[ "${INCLUDE_CUTLASS}" != "1" ]]; then
    return
  fi
  local dest="${ROOT_DIR}/third_party/cutlass"
  if [[ -d "${dest}/.git" ]]; then
    log "CUTLASS repository already present at ${dest}."
    return
  fi
  log "Cloning NVIDIA CUTLASS into ${dest}."
  mkdir -p "$(dirname "${dest}")"
  git clone --depth 1 https://github.com/NVIDIA/cutlass.git "${dest}"
}

install_system_packages
install_cuda_toolkit
ensure_python_env "${REQUESTED_ENV}"

source "${REQUESTED_ENV}/bin/activate"

install_python_dependencies "${REQUESTED_ENV}"
maybe_clone_cutlass

log "Bootstrap complete."
log "Activate the environment with: source \"${REQUESTED_ENV}/bin/activate\""
log "Bootstrap log saved to ${LOG_FILE}"
