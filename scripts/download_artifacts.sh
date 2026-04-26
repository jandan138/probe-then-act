#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_FILE="${1:-${REPO_ROOT}/.env.dsw}"

if [[ -f "${ENV_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${ENV_FILE}"
fi

PTA_CODE_ROOT="${PTA_CODE_ROOT:-${REPO_ROOT}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CHECKPOINT_BUNDLE_URL="${CHECKPOINT_BUNDLE_URL:-}"
CHECKPOINT_BUNDLE_SHA256="${CHECKPOINT_BUNDLE_SHA256:-}"
CHECKPOINT_BUNDLE_PATH="${CHECKPOINT_BUNDLE_PATH:-${PTA_CODE_ROOT}/results/dlc/checkpoint_bundle.tar.gz}"
CHECKPOINT_MANIFEST_PATH="${CHECKPOINT_MANIFEST_PATH:-${PTA_CODE_ROOT}/results/dlc/checkpoint_manifest.json}"

mkdir -p "$(dirname "${CHECKPOINT_BUNDLE_PATH}")"

if [[ -n "${CHECKPOINT_BUNDLE_URL}" ]]; then
    case "${CHECKPOINT_BUNDLE_URL}" in
        file://*)
            cp "${CHECKPOINT_BUNDLE_URL#file://}" "${CHECKPOINT_BUNDLE_PATH}"
            ;;
        http://*|https://*)
            if command -v curl >/dev/null 2>&1; then
                curl -L --fail --output "${CHECKPOINT_BUNDLE_PATH}" "${CHECKPOINT_BUNDLE_URL}"
            elif command -v wget >/dev/null 2>&1; then
                wget -O "${CHECKPOINT_BUNDLE_PATH}" "${CHECKPOINT_BUNDLE_URL}"
            else
                echo "curl or wget is required for HTTP downloads" >&2
                exit 2
            fi
            ;;
        *)
            cp "${CHECKPOINT_BUNDLE_URL}" "${CHECKPOINT_BUNDLE_PATH}"
            ;;
    esac

    if [[ -n "${CHECKPOINT_BUNDLE_SHA256}" ]]; then
        actual_sha="$("${PYTHON_BIN}" - "${CHECKPOINT_BUNDLE_PATH}" <<'PY'
import hashlib
import sys

path = sys.argv[1]
digest = hashlib.sha256()
with open(path, "rb") as handle:
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
        if [[ "${actual_sha}" != "${CHECKPOINT_BUNDLE_SHA256}" ]]; then
            echo "checkpoint bundle sha256 mismatch" >&2
            echo "expected: ${CHECKPOINT_BUNDLE_SHA256}" >&2
            echo "actual:   ${actual_sha}" >&2
            exit 3
        fi
    fi

    tar -xzf "${CHECKPOINT_BUNDLE_PATH}" -C "${PTA_CODE_ROOT}"
else
    echo "CHECKPOINT_BUNDLE_URL is empty; writing manifest for currently present files only."
fi

"${PYTHON_BIN}" "${PTA_CODE_ROOT}/scripts/build_checkpoint_manifest.py" \
    --repo-root "${PTA_CODE_ROOT}" \
    --manifest "${CHECKPOINT_MANIFEST_PATH}"
