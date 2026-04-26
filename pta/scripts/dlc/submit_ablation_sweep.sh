#!/bin/bash
set -euo pipefail

# Full sweep includes m7_noprobe seed 42. Use only after local R001 is abandoned
# or isolated; otherwise submit the remaining seeds with submit_jobs.py directly.
PYTHON_BIN=${PYTHON_BIN:-python}
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    else
        echo "ERROR: neither python nor python3 is available on PATH" >&2
        exit 2
    fi
fi

"$PYTHON_BIN" pta/scripts/dlc/submit_jobs.py \
    --suite ablation \
    --variants no_probe no_belief \
    --seeds 42 0 1 \
    "$@"
