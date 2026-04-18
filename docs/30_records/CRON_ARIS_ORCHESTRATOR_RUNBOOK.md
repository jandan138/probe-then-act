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

After reboot, wait for the next cron tick or run manually:

`bash pta/scripts/run_cron_aris_orchestrator.sh`
