#!/bin/bash
set -euo pipefail

ORIGINAL_ARGS=("$@")
CODE_ROOT=${PTA_CODE_ROOT:-${DLC_CODE_ROOT:-"/cpfs/shared/simulation/zhuzihou/dev/probe-then-act"}}
GENESIS_ROOT=${GENESIS_ROOT:-"/cpfs/shared/simulation/zhuzihou/dev/Genesis"}
GENESIS_VENV=${GENESIS_VENV:-"$GENESIS_ROOT/.venv"}
DLC_RESULTS_ROOT=${DLC_RESULTS_ROOT:-"$CODE_ROOT/results/dlc"}
RUN_ID=${DLC_RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)_${1:-no_mode}_${HOSTNAME:-worker}"}
RUN_START_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
RUN_MODE=${1:-}
RUN_RECORD="$DLC_RESULTS_ROOT/runs/$RUN_ID.json"
RUN_COMMAND_TEXT="bash pta/scripts/dlc/run_task.sh ${ORIGINAL_ARGS[*]:-}"
CHECKPOINT_HINT=""
RESULT_HINT=""
PYTHON_BIN=${PYTHON_BIN:-python}

export PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export PYTHONPATH="$CODE_ROOT:$GENESIS_ROOT:${PYTHONPATH:-}"
export RUN_ID RUN_MODE RUN_START_TIME RUN_COMMAND_TEXT CHECKPOINT_HINT RESULT_HINT RUN_RECORD PYTHON_BIN

mkdir -p "$DLC_RESULTS_ROOT/runs" "$DLC_RESULTS_ROOT/logs"

resolve_python_bin() {
    if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    elif command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "ERROR: neither python nor python3 is available on PATH" >&2
        exit 2
    fi
    export PYTHON_BIN
}

resolve_python_bin

write_record() {
    local exit_code=$1
    RUN_EXIT_CODE=$exit_code "$PYTHON_BIN" - <<'PY'
import json
import os
import socket
from datetime import datetime, timezone

path = os.environ["RUN_RECORD"]
payload = {
    "run_id": os.environ.get("RUN_ID", ""),
    "mode": os.environ.get("RUN_MODE", ""),
    "command": os.environ.get("RUN_COMMAND_TEXT", ""),
    "cwd": os.getcwd(),
    "start_time": os.environ.get("RUN_START_TIME", ""),
    "end_time": datetime.now(timezone.utc).isoformat(),
    "exit_code": int(os.environ.get("RUN_EXIT_CODE", "1")),
    "hostname": socket.gethostname(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "checkpoint_hint": os.environ.get("CHECKPOINT_HINT", ""),
    "result_hint": os.environ.get("RESULT_HINT", ""),
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
PY
}

on_exit() {
    local exit_code=$?
    write_record "$exit_code" || true
    exit "$exit_code"
}
trap on_exit EXIT

if [ ! -d "$CODE_ROOT" ]; then
    echo "ERROR: PTA_CODE_ROOT not found: $CODE_ROOT" >&2
    exit 2
fi
cd "$CODE_ROOT"

if [ -f "$GENESIS_VENV/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$GENESIS_VENV/bin/activate"
fi

if [ "${DLC_SKIP_PREFLIGHT:-0}" != "1" ]; then
    "$PYTHON_BIN" - <<'PY'
import importlib.util
import sys

required = ["genesis", "torch", "stable_baselines3", "gymnasium", "numpy", "pandas"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
print("Python:", sys.executable, flush=True)
if missing:
    raise SystemExit("missing required modules: " + ", ".join(missing))
for name in ["sb3_contrib"]:
    print(f"optional {name}: {'ok' if importlib.util.find_spec(name) else 'missing'}", flush=True)
PY
fi

if [ $# -eq 0 ]; then
    echo "Usage: run_task.sh <smoke_env|train_ablation|eval_ood|custom> [args...]" >&2
    exit 2
fi

MODE=$1
shift
RUN_MODE=$MODE
export RUN_MODE

run_or_print() {
    if [ "${DLC_DRY_RUN:-0}" = "1" ]; then
        printf 'DRY RUN:'
        printf ' %q' "$@"
        printf '\n'
        return 0
    fi
    "$@"
}

case "$MODE" in
    smoke_env)
        run_or_print "$PYTHON_BIN" - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available(), flush=True)
print("cuda_device_count", torch.cuda.device_count(), flush=True)
PY
        run_or_print "$PYTHON_BIN" pta/scripts/train_m7.py --help >/dev/null
        RESULT_HINT="smoke_env"
        export RESULT_HINT
        ;;
    train_ablation)
        if [ $# -ne 2 ]; then
            echo "Usage: run_task.sh train_ablation <no_probe|no_belief> <seed>" >&2
            exit 2
        fi
        VARIANT=$1
        SEED=$2
        case "$VARIANT" in
            no_probe)
                CKPT_DIR="m7_pta_noprobe_seed${SEED}"
                ;;
            no_belief)
                CKPT_DIR="m7_pta_nobelief_seed${SEED}"
                ;;
            *)
                echo "ERROR: unsupported ablation variant: $VARIANT" >&2
                exit 2
                ;;
        esac
        case "$SEED" in
            ''|*[!0-9]*)
                echo "ERROR: seed must be a non-negative integer: $SEED" >&2
                exit 2
                ;;
        esac
        CHECKPOINT_HINT="checkpoints/${CKPT_DIR}/m7_pta_final.zip"
        export CHECKPOINT_HINT
        run_or_print "$PYTHON_BIN" -u pta/scripts/train_m7.py \
            --ablation "$VARIANT" \
            --seed "$SEED" \
            --total-timesteps 500000 \
            --residual-scale 0.05
        if [ "${DLC_DRY_RUN:-0}" != "1" ] && [ ! -f "$CHECKPOINT_HINT" ]; then
            echo "ERROR: expected checkpoint missing: $CHECKPOINT_HINT" >&2
            exit 1
        fi
        ;;
    eval_ood)
        RESULT_HINT="results/main_results.csv"
        export RESULT_HINT
        if [ $# -eq 0 ]; then
            run_or_print "$PYTHON_BIN" -u pta/scripts/run_ood_eval_v2.py \
                --residual-scale 0.05 \
                --methods m7_noprobe m7_nobelief
        else
            run_or_print "$PYTHON_BIN" -u pta/scripts/run_ood_eval_v2.py "$@"
        fi
        ;;
    custom)
        if [ $# -eq 0 ]; then
            echo "Usage: run_task.sh custom <command...>" >&2
            exit 2
        fi
        CUSTOM_TEXT="$*"
        CUSTOM_TEXT_LC=$(printf '%s' "$CUSTOM_TEXT" | tr '[:upper:]' '[:lower:]')
        case "$CUSTOM_TEXT_LC" in
            *cron_aris_orchestrator*|*run_cron_aris_orchestrator*|*aris*|*opencode*|*claude*|*codex*|*auto-claude-code-research-in-sleep*)
                echo "ERROR: disallowed custom command for DLC worker: $CUSTOM_TEXT" >&2
                exit 2
                ;;
        esac
        run_or_print "$@"
        ;;
    *)
        echo "ERROR: unsupported mode: $MODE" >&2
        exit 2
        ;;
esac
