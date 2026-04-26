# Cron ARIS Orchestrator Runbook

This runbook is for the local WSL cron coordinator only. Do not install cron,
run ARIS, or launch opencode/Claude/Codex inside PAI-DLC workers. DLC workers
should use `docs/30_records/DLC_EXECUTION_RUNBOOK.md` and run only deterministic
train/eval commands.

## Install

Print the two-line 90-minute schedule:

`bash pta/scripts/install_cron_aris_orchestrator.sh`

Open the crontab editor:

`crontab -e`

Paste both lines so the job runs every 90 minutes.

## State and Logs

- state: `results/orchestration/aris_state.json`
- coordinator log: `logs/orchestration/cron_aris_orchestrator.log`
- OOD eval log: `logs/orchestration/run_ood_eval.log`

Cron is not a resident worker. It wakes on the installed 90-minute schedule, reconstructs state from running processes and artifacts, then exits.

## OOD Evaluation Policy

The coordinator only marks OOD evaluation complete when both `results/main_results.csv` and `results/ood_eval_per_seed.csv` exist, are newer than the relevant post-hotfix checkpoints, contain the exact expected per-seed keys, and have matching aggregate counts. Stale pre-fix outputs are not valid completion evidence; stale complete outputs trigger a `--no-resume` rerun, while fresh partial outputs resume.

`pta/scripts/run_ood_eval_v2.py` handles Genesis episode-level NaNs by counting that episode as a failure and continuing the sweep. Failed NaN episodes are visible via `n_failed_episodes` in per-seed results and `n_failed_episodes_sum` in aggregate results. Non-NaN exceptions should still stop the evaluator.

**2026-04-26 OOD recovery update:** before the resumable fix, the evaluator failed at the process level, not just at the episode level. Kernel logs show repeated OOM kills of cron-launched Python eval processes at roughly 12 GB anonymous RSS:

- `2026-04-25 21:43 HKT`: killed Python PID `1159227`, anon RSS `12133604kB`
- `2026-04-26 03:45 HKT`: killed Python PID `1241153`, anon RSS `12328420kB`
- `2026-04-26 09:20 HKT`: killed Python PID `1339114`, anon RSS `12361992kB`
- `2026-04-26 17:17 HKT`: killed Python PID `1417113`, anon RSS `12485336kB`; resumable CSVs preserved progress and the next launch completed the sweep

`run_ood_eval_v2.py` now writes `results/ood_eval_per_seed.csv` after every completed split and refreshes `results/main_results.csv`, so OOM kills preserve completed rows. If `ps` shows no `run_ood_eval_v2.py` process but `aris_state.json` still says `ood_eval.running=true`, treat the state file as stale and verify with `dmesg -T | rg -i "killed process|out of memory|oom|python"`.

The resumable OOD evaluator was implemented on 2026-04-26 and completed the corrected OOD matrix (`35/35` per-seed rows, `15` aggregate rows). Result-to-claim found the original broad PTA claims unsupported, so the selected automation path is Option 1: ablation-first diagnosis in `refine-logs/EXPERIMENT_PLAN.md`, starting with `m7_noprobe` and `m7_nobelief` seeds `42/0/1`.

## Recovery

If the orchestrator appears stalled after reboot or interruption, Inspect the current state file:

`cat results/orchestration/aris_state.json`

Inspect the coordinator log:

`tail -n 50 logs/orchestration/cron_aris_orchestrator.log`

After inspection, wait for the next cron tick or run the manual recovery command:

`bash pta/scripts/run_cron_aris_orchestrator.sh`

rerunning is safe while `aris.blocked=false` because the coordinator reconstructs state from artifacts and processes before deciding whether to launch the next step. If `aris.blocked=true`, inspect `failure_reason`, fix the cause, and clear the blocked state deliberately before expecting the handoff stage to advance.
