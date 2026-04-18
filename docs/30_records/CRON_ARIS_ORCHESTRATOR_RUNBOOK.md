# Cron ARIS Orchestrator Runbook

## Install

Print the two-line 90-minute schedule:

`bash pta/scripts/install_cron_aris_orchestrator.sh`

Open the crontab editor:

`crontab -e`

Paste both lines so the job runs every 90 minutes.

## State and Logs

- state: `results/orchestration/aris_state.json`
- coordinator log: `logs/orchestration/cron_aris_orchestrator.log`

## Recovery

If the orchestrator appears stalled after reboot or interruption, Inspect the current state file:

`results/orchestration/aris_state.json`

Inspect the coordinator log:

`logs/orchestration/cron_aris_orchestrator.log`

After inspection, wait for the next cron tick or run the manual recovery command:

`bash pta/scripts/run_cron_aris_orchestrator.sh`

rerunning is safe because the coordinator reconstructs state from artifacts and processes before deciding whether to launch the next step.
