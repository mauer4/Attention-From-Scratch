#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REQUIREMENTS_FILE="${ROOT_DIR}/requirements.txt"

echo "📦 Installing Python dependencies from ${REQUIREMENTS_FILE}..."
pip install --no-cache-dir -r "${REQUIREMENTS_FILE}"

echo "🔧 Installing project in editable mode..."
pip install -e "${ROOT_DIR}"

echo "✅ Dependencies installed."
