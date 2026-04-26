#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTA_CODE_ROOT="${PTA_CODE_ROOT:-$(cd "${SCRIPT_DIR}/../../.." && pwd)}"
GENESIS_ROOT="${GENESIS_ROOT:-$(dirname "${PTA_CODE_ROOT}")/Genesis}"
GENESIS_VENV="${GENESIS_VENV:-${GENESIS_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
DLC_RESULTS_ROOT="${DLC_RESULTS_ROOT:-${PTA_CODE_ROOT}/results/dlc}"
CHECKPOINT_STRICT="${CHECKPOINT_STRICT:-0}"
REQUIRE_DLC_CLI="${REQUIRE_DLC_CLI:-0}"

if [[ ! -d "${PTA_CODE_ROOT}" ]]; then
    echo "PTA_CODE_ROOT not found: ${PTA_CODE_ROOT}" >&2
    exit 2
fi

if [[ ! -d "${GENESIS_ROOT}" ]]; then
    echo "GENESIS_ROOT not found: ${GENESIS_ROOT}" >&2
    exit 2
fi

if [[ ! -x "${GENESIS_VENV}/bin/python" && "${PYTHON_BIN}" == "${GENESIS_VENV}/bin/python" ]]; then
    echo "GENESIS_VENV python not executable: ${GENESIS_VENV}/bin/python" >&2
    exit 2
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1 && [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "PYTHON_BIN not executable: ${PYTHON_BIN}" >&2
    exit 2
fi

if [[ "${REQUIRE_DLC_CLI}" == "1" ]] && ! command -v dlc >/dev/null 2>&1; then
    echo "DLC CLI not found on PATH" >&2
    exit 2
fi

mkdir -p "${DLC_RESULTS_ROOT}"

"${PYTHON_BIN}" "${PTA_CODE_ROOT}/scripts/build_checkpoint_manifest.py" \
    --repo-root "${PTA_CODE_ROOT}" \
    --manifest "${DLC_RESULTS_ROOT}/checkpoint_manifest.json" \
    $(if [[ "${CHECKPOINT_STRICT}" == "1" ]]; then printf '%s' "--strict"; fi)

cat <<EOF
Remote preflight paths:
  PTA_CODE_ROOT=${PTA_CODE_ROOT}
  GENESIS_ROOT=${GENESIS_ROOT}
  GENESIS_VENV=${GENESIS_VENV}
  PYTHON_BIN=${PYTHON_BIN}
  DLC_RESULTS_ROOT=${DLC_RESULTS_ROOT}
EOF
