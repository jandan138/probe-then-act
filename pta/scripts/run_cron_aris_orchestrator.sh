#!/bin/bash
set -euo pipefail

source /home/zhuzihou/dev/Genesis/.venv/bin/activate
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d:${PYTHONPATH:-}

cd /home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d
python pta/scripts/cron_aris_orchestrator.py >> logs/orchestration/cron_aris_orchestrator.log 2>&1
