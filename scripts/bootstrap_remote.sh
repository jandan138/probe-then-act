#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${1:-${REPO_ROOT}/.env.dsw}"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi

PTA_REPO_URL="${PTA_REPO_URL:-git@github.com:jandan138/probe-then-act.git}"
GENESIS_REPO_URL="${GENESIS_REPO_URL:-git@github.com:jandan138/Genesis.git}"
PTA_CODE_ROOT="${PTA_CODE_ROOT:-${REPO_ROOT}}"
GENESIS_ROOT="${GENESIS_ROOT:-$(dirname "${PTA_CODE_ROOT}")/Genesis}"
GENESIS_VENV="${GENESIS_VENV:-${GENESIS_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BOOTSTRAP_SKIP_INSTALL="${BOOTSTRAP_SKIP_INSTALL:-0}"

if ! command -v git >/dev/null 2>&1; then
    echo "git is required" >&2
    exit 2
fi

if [[ ! -d "${PTA_CODE_ROOT}/.git" ]]; then
    mkdir -p "$(dirname "${PTA_CODE_ROOT}")"
    git clone "${PTA_REPO_URL}" "${PTA_CODE_ROOT}"
fi

if [[ ! -d "${GENESIS_ROOT}/.git" ]]; then
    mkdir -p "$(dirname "${GENESIS_ROOT}")"
    git clone "${GENESIS_REPO_URL}" "${GENESIS_ROOT}"
fi

if [[ "${BOOTSTRAP_SKIP_INSTALL}" != "1" ]]; then
    "${PYTHON_BIN}" -m venv "${GENESIS_VENV}"
    # shellcheck disable=SC1091
    source "${GENESIS_VENV}/bin/activate"
    python -m pip install --upgrade pip
    python -m pip install -e "${GENESIS_ROOT}"
    python -m pip install -r "${PTA_CODE_ROOT}/requirements.txt"
fi

cat <<EOF
Remote bootstrap paths:
  PTA_CODE_ROOT=${PTA_CODE_ROOT}
  GENESIS_ROOT=${GENESIS_ROOT}
  GENESIS_VENV=${GENESIS_VENV}

Next:
  cd ${PTA_CODE_ROOT}
  cp .env.dsw.example .env.dsw
  scripts/download_artifacts.sh .env.dsw
  scripts/smoke_remote.sh .env.dsw
EOF
