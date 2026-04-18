#!/bin/bash
set -euo pipefail

CRON_A="0 */3 * * * /home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d/pta/scripts/run_cron_aris_orchestrator.sh"
CRON_B="30 1-22/3 * * * /home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d/pta/scripts/run_cron_aris_orchestrator.sh"

printf '%s\n%s\n' "$CRON_A" "$CRON_B"
