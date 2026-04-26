# Cron ARIS Orchestrator Design

## Goal

Add a durable local orchestration layer that survives OpenCode session loss and machine restarts.

Every 90 minutes, a cron-triggered coordinator should inspect the current project state and advance the research pipeline by at most one well-defined step. The system should cover the local experiment chain first, then hand off into this repo's existing automatic research workflow once experiment evidence is ready.

## Non-Goals

- No always-on daemon or queue service.
- No platform-specific OpenCode scheduler integration.
- No blind retry loops for failed research stages.
- No rewriting of the existing training scripts into a new framework.
- No direct automation of every ARIS command without persistent state and evidence checks.

## Design Summary

Use a two-layer approach:

1. `cron` provides periodic wake-up every 90 minutes.
2. A new idempotent coordinator script reads local facts, updates a small state file, and launches at most one missing next-stage task.

This keeps scheduling simple and robust while allowing the research pipeline to resume after reboot. The coordinator never assumes success from prior intent; it only trusts concrete evidence such as process presence, checkpoints, evaluation files, and result artifacts.

## Scope

The orchestrator covers this progression:

1. monitor current `M8 seed=42` formal run
2. when complete, launch `M1` seeds `42`, `0`, `1`
3. when `M1` completes, launch `M7` seeds `42`, `0`, `1`
4. when core training artifacts exist, run corrected OOD evaluation
5. when experiment evidence is complete, mark ARIS handoff ready
6. invoke the repo-level automatic research continuation entrypoint for review and paper stages

The first implementation should treat ARIS handoff as a controlled next step, not a free-form infinite automation loop.

## Architecture

### 1. Cron Entry

Add a project-scoped cron entry that runs every 90 minutes.

The cron target should be a single shell wrapper that:

- activates the Genesis virtual environment
- sets required headless rendering environment variables
- enters the worktree path
- runs the coordinator
- appends output to a dedicated coordinator log

The cron line should be deterministic and project-local in behavior, even though installation happens in the user's crontab.

### 2. Coordinator Script

Add a new script under `pta/scripts/` responsible for orchestration.

Responsibilities:

- inspect active processes relevant to current training/eval stages
- inspect run artifacts under `logs/`, `checkpoints/`, and `results/`
- persist a normalized pipeline state JSON
- decide the next legal step
- launch only that next step if no conflicting process is active
- record every decision to a plain-text coordinator log

The coordinator must be idempotent. Running it repeatedly should not duplicate work.

### 3. Persistent State File

Maintain a small JSON state file, for example under `results/orchestration/aris_state.json`.

It should track:

- current stage
- last check time
- active process metadata if known
- completed runs by method/seed
- latest discovered checkpoints
- latest discovered eval/result artifacts
- ARIS handoff readiness
- last failure reason, if any

This file is an optimization and audit trail. The coordinator should recompute truth from disk/process state when possible rather than trusting stale state blindly.

### 4. Stage Boundaries

Each stage advances only when concrete completion evidence exists.

Completion evidence rules:

- `M8` complete: final checkpoint exists for the formal post-hotfix run, or process has exited and the expected completion artifact exists
- `M1` complete: all three seeds have final checkpoints
- `M7` complete: all three seeds have final checkpoints
- OOD eval complete: expected result CSV outputs exist and are newer than the relevant checkpoints
- ARIS handoff ready: all required experiment artifacts and summaries exist

If a process is still active, the coordinator should log `running` and do nothing else.

If a stage has neither a running process nor completion artifacts, the coordinator may launch the stage.

## Task Launch Policy

### M8

The coordinator should treat the current `M8 seed=42` run as already launched and only monitor it.

If the process is gone but final artifacts are missing, the coordinator should restart from the latest known resume point rather than from scratch.

### M1

Launch `M1` sequentially by seed to fit a single local GPU/CPU-heavy workflow and simplify recovery.

Order:

1. seed `42`
2. seed `0`
3. seed `1`

Each seed uses the post-hotfix command:

`python pta/scripts/train_baselines.py --method m1 --seed <seed> --total-timesteps 500000 --residual-scale 0.05`

### M7

Launch `M7` sequentially after `M1` completes.

Order:

1. seed `42`
2. seed `0`
3. seed `1`

Each seed uses:

`python pta/scripts/train_m7.py --seed <seed> --total-timesteps 500000 --residual-scale 0.05`

### OOD Evaluation

After `M1` and `M7` checkpoints exist, run the corrected evaluation path using `pta/scripts/run_ood_eval_v2.py`.

The coordinator must only use the fixed evaluation codepath with:

- correct metric keys: `transfer_efficiency`, `spill_ratio`
- hotfix-aligned `residual_scale=0.05`

The coordinator should never treat pre-fix evaluation outputs as valid evidence for handoff.

The evaluator should count Genesis episode-level NaNs as failed rollouts and continue the sweep, while surfacing `n_failed_episodes` in both per-seed and aggregate outputs. Non-NaN exceptions remain infrastructure or code failures and should stop the run.

2026-04-26 update: this is necessary but not sufficient. The OOD evaluator must also be resumable because the local machine repeatedly OOM-kills long Genesis evaluation processes before final CSVs are written. The scheduler should continue to relaunch OOD eval after crashes, but the evaluator itself must persist per-row progress and skip completed `(method, seed, split)` combinations on restart. This has now been implemented and validated by a complete corrected OOD sweep.

## ARIS Handoff

Once experiment outputs are complete, the coordinator should move into a handoff stage rather than embed all review/writing logic directly.

The handoff contract is:

- write a compact machine-readable readiness record
- write a human-readable summary file describing completed experiments and result locations
- invoke the repo's automatic research continuation entrypoint in a single controlled command

The first implementation should support a configurable handoff command so that the project can point it at the current preferred ARIS entrypoint without changing the core scheduler.

Examples of acceptable handoff targets:

- a local shell wrapper in this repo
- a command file consumed by the Auto-claude-code-research-in-sleep project mirror
- a future stable entrypoint for `/auto-review-loop` and `/paper-writing`

## Failure Handling

The coordinator must prefer explicit stop states over unsafe retries.

If a stage fails:

- record the failure reason in the state JSON
- mark the stage as `blocked`
- do not auto-retry indefinitely
- do not advance to later stages

Examples of blocking conditions:

- training process exits without final checkpoint
- evaluation command exits non-zero
- required checkpoints missing for one or more seeds
- ARIS handoff command missing or returns failure

This keeps the automation auditable and prevents cron from repeatedly launching broken work.

## Logging

Use two logging layers:

1. existing per-run logs stay where they already live
2. coordinator writes a dedicated append-only orchestration log

Coordinator log entries should include:

- timestamp
- observed stage
- evidence found
- decision taken
- launched command if any
- PID if a new process was started

This log is the recovery source after restarts.

## Recovery Model

After reboot or session loss, the next cron invocation should be enough to recover control.

Recovery algorithm:

1. scan active experiment processes
2. scan checkpoints, eval outputs, and result files
3. reconcile them into normalized stage state
4. if something is already running, continue monitoring only
5. if the active stage is complete, advance one stage
6. if the active stage is incomplete and nothing is running, relaunch only the missing stage

The design assumes crashes and restarts are normal and should not require manual bookkeeping in most cases.

## Testing Strategy

Testing should focus on coordinator decisions rather than full long-running experiments.

Required tests:

- state detection from synthetic artifact layouts
- stage transition logic
- skip-if-running behavior
- skip-if-complete behavior
- no duplicate launch when cron fires repeatedly
- ARIS handoff only after all required prerequisites exist

Implementation should isolate filesystem/process inspection behind small functions so these decisions can be tested without real training runs.

## Security and Safety Constraints

- Never install a system-wide service silently; emit the cron line and install it only through an explicit setup command
- Never kill unrelated processes
- Never delete checkpoints or logs automatically
- Never advance past a failed stage without explicit evidence or manual reset

## Rollout Plan

Phase 1:

- add coordinator script
- add state file handling
- add cron wrapper script
- support `M8 -> M1 -> M7 -> OOD eval`

Phase 2:

- add ARIS handoff summary and handoff command execution

Phase 3:

- add optional helper command to install/uninstall the cron entry cleanly

## Open Questions Resolved

- Schedule frequency: every 90 minutes
- Execution style: project-local cron wrapper plus idempotent coordinator
- Advancement scope: continue through local training/eval and then hand off into repo-level automatic research workflow
- Concurrency model: single-path sequential orchestration, not parallel experiment launching
