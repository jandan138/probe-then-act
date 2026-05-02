#!/bin/bash
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: bash launch_job.sh <TASK_NAME> <CHUNK_ID> <CHUNK_TOTAL> [DATA_SOURCES] [COMMAND_ARGS]" >&2
    exit 1
fi

TASK_NAME=$1
CHUNK_ID=$2
CHUNK_TOTAL=$3
if [ $# -ge 4 ]; then
    DATA_SOURCES=$4
    shift 4
else
    DATA_SOURCES="d-mzps5b7joy2axmqpa8,d-d49o5g0h2818sw8j1g,d-8wz4emfs21s5ajs9oz"
    shift 3
fi
if [ $# -gt 0 ]; then
    COMMAND_ARGS="$*"
else
    COMMAND_ARGS="smoke_env"
fi

WORKSPACE_ID=${DLC_WORKSPACE_ID:-"270969"}
IMAGE=${DLC_IMAGE:-"pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang"}
CODE_ROOT=${PTA_CODE_ROOT:-${DLC_CODE_ROOT:-"/cpfs/shared/simulation/zhuzihou/dev/probe-then-act"}}
DLC_BIN=${DLC_BIN:-"$CODE_ROOT/dlc"}
GPU_COUNT=${DLC_GPU_COUNT:-1}

case "$GPU_COUNT" in
    1)
        WORKER_GPU=${DLC_WORKER_GPU:-1}
        WORKER_CPU=${DLC_WORKER_CPU:-14}
        WORKER_MEMORY=${DLC_WORKER_MEMORY:-100Gi}
        WORKER_SHARED_MEMORY=${DLC_WORKER_SHARED_MEMORY:-100Gi}
        RESOURCE_ID=${DLC_RESOURCE_ID:-quota1r947pmazvk}
        ;;
    2)
        WORKER_GPU=${DLC_WORKER_GPU:-2}
        WORKER_CPU=${DLC_WORKER_CPU:-28}
        WORKER_MEMORY=${DLC_WORKER_MEMORY:-200Gi}
        WORKER_SHARED_MEMORY=${DLC_WORKER_SHARED_MEMORY:-200Gi}
        RESOURCE_ID=${DLC_RESOURCE_ID:-quota1r947pmazvk}
        ;;
    4)
        WORKER_GPU=${DLC_WORKER_GPU:-4}
        WORKER_CPU=${DLC_WORKER_CPU:-56}
        WORKER_MEMORY=${DLC_WORKER_MEMORY:-400Gi}
        WORKER_SHARED_MEMORY=${DLC_WORKER_SHARED_MEMORY:-400Gi}
        RESOURCE_ID=${DLC_RESOURCE_ID:-quota1r947pmazvk}
        ;;
    8)
        WORKER_GPU=${DLC_WORKER_GPU:-8}
        WORKER_CPU=${DLC_WORKER_CPU:-128}
        WORKER_MEMORY=${DLC_WORKER_MEMORY:-960Gi}
        WORKER_SHARED_MEMORY=${DLC_WORKER_SHARED_MEMORY:-960Gi}
        RESOURCE_ID=${DLC_RESOURCE_ID:-quotaksvqq2oh2pg}
        ;;
    *)
        echo "ERROR: unsupported DLC_GPU_COUNT=$GPU_COUNT" >&2
        exit 1
        ;;
esac

JOB_NAME="${TASK_NAME}_${CHUNK_ID}_${CHUNK_TOTAL}"
WORKER_COMMAND="bash $CODE_ROOT/pta/scripts/dlc/run_task.sh ${COMMAND_ARGS}"

echo "Submitting Job: $JOB_NAME"
echo "Code Root: $CODE_ROOT"
echo "Resolved config -> GPU=$WORKER_GPU CPU=$WORKER_CPU Memory=$WORKER_MEMORY SharedMem=$WORKER_SHARED_MEMORY Resource=$RESOURCE_ID"

if [ "${DLC_DRY_RUN:-0}" = "1" ]; then
    echo "$DLC_BIN submit pytorchjob --name=$JOB_NAME --workers=1 --job_max_running_time_minutes=0 --worker_gpu=$WORKER_GPU --worker_cpu=$WORKER_CPU --worker_memory=$WORKER_MEMORY --worker_shared_memory=$WORKER_SHARED_MEMORY --worker_image=$IMAGE --workspace_id=$WORKSPACE_ID --resource_id=$RESOURCE_ID --data_sources=$DATA_SOURCES --oversold_type=ForbiddenQuotaOverSold --priority 7 --command=\"$WORKER_COMMAND\""
    exit 0
fi

if [ ! -x "$DLC_BIN" ]; then
    echo "ERROR: DLC binary not found or not executable at $DLC_BIN" >&2
    exit 1
fi

"$DLC_BIN" submit pytorchjob --name="$JOB_NAME" \
    --workers=1 \
    --job_max_running_time_minutes=0 \
    --worker_gpu="$WORKER_GPU" \
    --worker_cpu="$WORKER_CPU" \
    --worker_memory="$WORKER_MEMORY" \
    --worker_shared_memory="$WORKER_SHARED_MEMORY" \
    --worker_image="$IMAGE" \
    --workspace_id="$WORKSPACE_ID" \
    --resource_id="$RESOURCE_ID" \
    --data_sources="$DATA_SOURCES" \
    --oversold_type=ForbiddenQuotaOverSold \
    --priority 7 \
    --command="$WORKER_COMMAND"
