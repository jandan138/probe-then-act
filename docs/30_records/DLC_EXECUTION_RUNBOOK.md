# PAI-DLC Execution Runbook

This runbook adapts the working DLC submission pattern from
`/home/zhuzihou/dev/usd-scene-physics-prep` to Probe-Then-Act. The selected
architecture is Approach B:

- the probe repo owns the DLC execution layer;
- the DSW machine submits deterministic train/eval jobs to PAI-DLC;
- DLC workers run only bounded experiment commands;
- ARIS, cron, opencode, Claude, Codex, and Auto-repo orchestration stay outside
  the DLC worker.

## Files

- `pta/scripts/dlc/submit_jobs.py` builds job specs and appends
  `results/dlc/jobs.jsonl`.
- `pta/scripts/dlc/launch_job.sh` wraps `dlc submit pytorchjob`.
- `pta/scripts/dlc/run_task.sh` is the worker entrypoint.
- `pta/scripts/dlc/submit_ablation_sweep.sh` submits the default six ablation
  training jobs.
- `docs/superpowers/specs/2026-04-26-dlc-execution-layer-design.md` records the
  design boundary and rationale.
- `docs/superpowers/plans/2026-04-26-dlc-execution-layer.md` records the
  implementation plan.

## DSW Setup

Upload or sync the probe repo and Genesis runtime to the CPFS path visible from
both DSW and DLC workers. The default paths are:

```bash
export PTA_CODE_ROOT=/cpfs/shared/simulation/zhuzihou/dev/probe-then-act
export GENESIS_ROOT=/cpfs/shared/simulation/zhuzihou/dev/Genesis
export GENESIS_VENV=$GENESIS_ROOT/.venv
export DLC_RESULTS_ROOT=$PTA_CODE_ROOT/results/dlc
export DLC_BIN=$PTA_CODE_ROOT/dlc
export DLC_DATA_SOURCES=d-mzps5b7joy2axmqpa8,d-d49o5g0h2818sw8j1g,d-8wz4emfs21s5ajs9oz
```

Override these variables if the DSW upload lands elsewhere. The launch script
also supports:

```bash
export DLC_WORKSPACE_ID=270969
export DLC_IMAGE=pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang
export DLC_GPU_COUNT=1
export DLC_RESOURCE_ID=quota1r947pmazvk
```

`DLC_GPU_COUNT` must be one of `1`, `2`, `4`, or `8`. The default data sources
are inherited from the working `usd-scene-physics-prep` submit pattern:

```text
d-mzps5b7joy2axmqpa8,d-d49o5g0h2818sw8j1g,d-8wz4emfs21s5ajs9oz
```

Use `--data-sources` on `submit_jobs.py` if the DSW workspace requires a
different mount set.

The `launch_job.sh` default image is the verified Genesis/PTA training image above. Do not use the generic PyTorch image for Genesis training jobs unless intentionally running a non-Genesis smoke/debug command.

## Dry Runs

Always dry-run from DSW before real submission:

```bash
cd "$PTA_CODE_ROOT"
DLC_DRY_RUN=1 bash pta/scripts/dlc/launch_job.sh pta_smoke 0 1 "$DLC_DATA_SOURCES" smoke_env
python3 pta/scripts/dlc/submit_jobs.py --suite smoke --dry-run
python3 pta/scripts/dlc/submit_jobs.py --suite ablation --variants no_probe --seeds 0 1 --dry-run
python3 pta/scripts/dlc/submit_jobs.py --suite ablation --variants no_belief --seeds 42 0 1 --dry-run
```

Before treating a submission as valid, run:

```bash
"$DLC_BIN" get job <JOB_ID> \
  --endpoint=pai-dlc.cn-beijing.aliyuncs.com \
  --region=cn-beijing
```

Expected: `JobSpecs[0].Image` is `pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang` for all Genesis/PTA training jobs.

For local contract checks without Genesis preflight:

```bash
cd /home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d
DLC_DRY_RUN=1 DLC_SKIP_PREFLIGHT=1 PTA_CODE_ROOT=$PWD \
  bash pta/scripts/dlc/run_task.sh train_ablation no_probe 42
```

## Submission Order

R001 completed locally as `m7_noprobe seed=42`. Do not submit a duplicate DLC
job for that exact run unless intentionally isolating results in a separate
result root.

Recommended DLC route after local R001 completion:

```bash
cd "$PTA_CODE_ROOT"

# Smoke first; inspect results/dlc/runs/*.json after the worker exits.
python3 pta/scripts/dlc/submit_jobs.py --suite smoke

# Submit the remaining no-probe seeds only.
python3 pta/scripts/dlc/submit_jobs.py --suite ablation --variants no_probe --seeds 0 1

# Submit all no-belief seeds.
python3 pta/scripts/dlc/submit_jobs.py --suite ablation --variants no_belief --seeds 42 0 1
```

If the local R001 result is intentionally discarded or isolated, the full ablation sweep is:

```bash
bash pta/scripts/dlc/submit_ablation_sweep.sh
```

After all required checkpoints exist, run the ablation-only OOD expansion:

```bash
python3 pta/scripts/dlc/submit_jobs.py --suite ood-ablation
```

The OOD job runs:

```bash
python -u pta/scripts/run_ood_eval_v2.py \
  --residual-scale 0.05 \
  --methods m7_noprobe m7_nobelief
```

The ablation OOD rows use the final 500k checkpoints:

```text
checkpoints/m7_pta_noprobe_seed{seed}/m7_pta_final.zip
checkpoints/m7_pta_nobelief_seed{seed}/m7_pta_final.zip
```

This differs from the earlier M1/M7/M8 baseline rows, which used their existing
`best/best_model.zip` checkpoint policy. The final-checkpoint policy is
intentional for DLC-resumed ablations because eval callbacks were disabled during
the final 400k-to-500k resume segment, so `best/best_model.zip` was not refreshed.

## Worker Modes

`run_task.sh` supports only these modes:

- `smoke_env`: verifies CUDA visibility and `train_m7.py --help`.
- `train_ablation <no_probe|no_belief> <seed>`: runs 500K-step M7 ablation
  training and verifies the expected final checkpoint after success.
- `eval_ood [args...]`: runs corrected resumable OOD evaluation; without args it
  evaluates `m7_noprobe` and `m7_nobelief`.
- `custom <command...>`: guarded escape hatch for bounded experiment commands.

The custom mode rejects commands that contain cron/ARIS entrypoints, opencode,
Claude, Codex, or the Auto repo path. DLC workers are compute workers, not
orchestrators.

## Artifacts

Submission records:

```text
results/dlc/jobs.jsonl
```

Worker records:

```text
results/dlc/runs/<run_id>.json
```

Each worker record includes `mode`, `command`, `exit_code`, `hostname`,
`cuda_visible_devices`, `checkpoint_hint`, and `result_hint`.

Expected ablation checkpoints:

```text
checkpoints/m7_pta_noprobe_seed{seed}/m7_pta_final.zip
checkpoints/m7_pta_nobelief_seed{seed}/m7_pta_final.zip
```

Expected OOD outputs:

```text
results/ood_eval_per_seed.csv
results/main_results.csv
```

After DLC jobs finish, update `refine-logs/EXPERIMENT_TRACKER.md`, run
result-to-claim again, and mirror the status in the Auto repo execution log.

## Failure Handling

- If `launch_job.sh` fails before submission, check `DLC_BIN`, workspace ID,
  quota/resource ID, image, and data sources on the DSW machine.
- If `smoke_env` fails inside DLC, fix the CPFS Genesis runtime or image before
  submitting expensive training jobs.
- If a training worker exits non-zero, inspect `results/dlc/runs/*.json` and the
  DLC console logs, then rerun only the missing `(variant, seed)` job.
- If OOD is OOM-killed, rerun the same `ood-ablation` job; the evaluator is
  resumable and persists completed rows.
