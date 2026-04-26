#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${1:-${REPO_ROOT}/.env.local}"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi

PTA_CODE_ROOT="${PTA_CODE_ROOT:-${REPO_ROOT}}"
GENESIS_ROOT="${GENESIS_ROOT:-$(dirname "${PTA_CODE_ROOT}")/Genesis}"
GENESIS_VENV="${GENESIS_VENV:-${GENESIS_ROOT}/.venv}"
DLC_RESULTS_ROOT="${DLC_RESULTS_ROOT:-${PTA_CODE_ROOT}/results/dlc}"

if [[ -x "${GENESIS_VENV}/bin/python" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-${GENESIS_VENV}/bin/python}"
else
    PYTHON_BIN="${PYTHON_BIN:-python3}"
fi

export PYTHONPATH="${GENESIS_ROOT}:${PTA_CODE_ROOT}:${PYTHONPATH:-}"
export PTA_CODE_ROOT
export GENESIS_ROOT
export GENESIS_VENV
export DLC_RESULTS_ROOT

"${PYTHON_BIN}" - <<'PY'
import importlib
import os

for module_name in ("genesis", "pta"):
    module = importlib.import_module(module_name)
    print(f"{module_name}: {getattr(module, '__file__', '<namespace>')}")

print("PTA_CODE_ROOT=" + os.environ["PTA_CODE_ROOT"])
print("GENESIS_ROOT=" + os.environ["GENESIS_ROOT"])
PY

bash "${PTA_CODE_ROOT}/pta/scripts/dlc/preflight_remote.sh"

DLC_DRY_RUN=1 DLC_SKIP_PREFLIGHT=1 DLC_RUN_ID=remote_smoke \
    bash "${PTA_CODE_ROOT}/pta/scripts/dlc/run_task.sh" eval_ood
