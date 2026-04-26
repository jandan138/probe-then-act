# DLC Execution Layer Design

## Decision

Use a project-level Alibaba PAI-DLC execution layer for `probe-then-act`.
The DLC layer will live in the experiment repo and run deterministic training
or evaluation jobs on DLC workers. ARIS remains the research-control and record
layer; it does not run inside every DLC worker.

## Scope

This design covers the immediate ablation-first diagnostic cycle:

- Train `m7_noprobe` for seeds `42`, `0`, and `1`.
- Train `m7_nobelief` for seeds `42`, `0`, and `1`.
- Rerun corrected resumable OOD v2 after the six ablation checkpoints exist.
- Preserve enough job metadata for DSW-side monitoring and ARIS log updates.

The design intentionally does not implement a fully automatic cloud research
controller. Result-to-claim and paper-facing decisions remain outside DLC jobs.

## Repositories

### `probe-then-act`

Role: executable experiment repo.

It owns:

- DLC submission scripts.
- Worker dispatcher scripts.
- Training and evaluation commands.
- Checkpoints, logs, results, and DLC job manifests.

### `Auto-claude-code-research-in-sleep`

Role: ARIS mirror and research memory.

It owns:

- High-level execution plan.
- User-facing runbook.
- Status summaries copied from `probe-then-act` outputs.
- Research decisions after result-to-claim.

It must not duplicate training logic or submit jobs as the default path.

### `Genesis`

Role: runtime dependency.

The DLC worker must have either:

- a usable CPFS-mounted Genesis checkout with a prepared virtual environment, or
- a DLC image that already contains compatible Genesis, PyTorch, and RL deps.

## Directory Layout

Add the DLC layer under:

```text
pta/scripts/dlc/
  launch_job.sh
  run_task.sh
  submit_jobs.py
  submit_ablation_sweep.sh
```

Write job metadata under:

```text
results/dlc/
  jobs.jsonl
  runs/
    <run_id>.json
  logs/
    <run_id>.log
```

The existing training outputs remain unchanged:

```text
checkpoints/m7_pta_noprobe_seed42/
checkpoints/m7_pta_noprobe_seed0/
checkpoints/m7_pta_noprobe_seed1/
checkpoints/m7_pta_nobelief_seed42/
checkpoints/m7_pta_nobelief_seed0/
checkpoints/m7_pta_nobelief_seed1/
results/ood_eval_per_seed.csv
results/main_results.csv
```

## Environment Contract

All paths are configurable so the same scripts work locally, on DSW, and inside
DLC workers.

```bash
PTA_CODE_ROOT=/cpfs/shared/simulation/zhuzihou/dev/probe-then-act
ARIS_CODE_ROOT=/cpfs/shared/simulation/zhuzihou/dev/Auto-claude-code-research-in-sleep
GENESIS_ROOT=/cpfs/shared/simulation/zhuzihou/dev/Genesis
GENESIS_VENV=/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv
DLC_WORKSPACE_ID=270969
DLC_IMAGE=<Genesis-capable training image>
DLC_GPU_COUNT=1
```

`run_task.sh` must set:

```bash
PYTHONUNBUFFERED=1
PYOPENGL_PLATFORM=osmesa
PYTHONPATH=$PTA_CODE_ROOT:$GENESIS_ROOT:$PYTHONPATH
```

If `GENESIS_VENV/bin/activate` exists, `run_task.sh` activates it. If it does
not exist, the script uses the image Python and validates imports explicitly.

Required import preflight:

- `genesis`
- `torch`
- `stable_baselines3`
- `gymnasium`
- `numpy`
- `pandas`

`sb3_contrib` is optional for the immediate ablation path but should be reported
if missing because M2/RNN is a conditional follow-up.

## Submission Model

`submit_jobs.py` runs on DSW and submits one DLC job per requested run.

Supported suites:

- `ablation`: submit one job per `(variant, seed)`.
- `ood-ablation`: submit one OOD eval job after ablation checkpoints exist.
- `smoke`: submit a cheap worker environment check.

`launch_job.sh` wraps `dlc submit pytorchjob`. It follows the known
`usd-scene-physics-prep` pattern:

- one worker per job,
- configurable image, workspace, resource, data sources, GPU count,
- `--command="bash $PTA_CODE_ROOT/pta/scripts/dlc/run_task.sh ..."` inside the
  DLC job.

## Worker Modes

### `smoke_env`

Validate the worker environment and write a run record.

Checks:

- expected roots exist,
- Python executable is printed,
- required imports load,
- CUDA is visible,
- `pta/scripts/train_m7.py --help` exits successfully.

### `train_ablation <variant> <seed>`

Supported variants:

- `no_probe`
- `no_belief`

The worker command is:

```bash
python -u pta/scripts/train_m7.py \
  --ablation <variant> \
  --seed <seed> \
  --total-timesteps 500000 \
  --residual-scale 0.05
```

The mode must reject unsupported variants and non-integer seeds before launching
Python.

### `eval_ood [args...]`

Run the corrected resumable evaluator. For the immediate ablation cycle, the
recommended invocation is:

```bash
python -u pta/scripts/run_ood_eval_v2.py \
  --residual-scale 0.05 \
  --methods m7_noprobe m7_nobelief
```

The evaluator keeps its existing per-row resume behavior. It should be updated
to include seed `1` for `m7_noprobe` and `m7_nobelief`, or to discover optional
ablation seeds from available checkpoint directories.

### `custom <command...>`

Run a one-off command from `PTA_CODE_ROOT`. This is for emergency debugging and
must still write a run record.

## Run Records

Each worker writes a JSON file at:

```text
results/dlc/runs/<run_id>.json
```

Fields:

- `run_id`
- `mode`
- `command`
- `cwd`
- `start_time`
- `end_time`
- `exit_code`
- `hostname`
- `cuda_visible_devices`
- `checkpoint_hint`
- `result_hint`

The DSW submitter appends submission metadata to:

```text
results/dlc/jobs.jsonl
```

Fields:

- `submitted_at`
- `job_name`
- `suite`
- `mode`
- `variant`
- `seed`
- `gpu_count`
- `command_args`

If the DLC CLI exposes a job id in stdout, the submitter should capture it when
practical. The initial implementation may store the raw stdout path instead.

## Resource Defaults

Immediate ablation training defaults:

- `DLC_GPU_COUNT=1`
- one training run per job
- one worker per job
- CPU and memory follow the same template style as `usd-scene-physics-prep`

OOD eval defaults:

- `DLC_GPU_COUNT=1`
- run as one resumable job
- if OOM occurs, resubmit the same OOD job; existing rows are skipped

Do not run multiple ablation seeds inside one DLC job unless quota pressure makes
job count a bigger problem than wall time. Independent one-run jobs are easier
to retry and inspect.

## Failure Handling

The worker script uses `set -euo pipefail` and records the final exit code.

Expected retry behavior:

- Failed training job: resubmit the same `(variant, seed)` after inspecting the
  run log and partial checkpoint directory.
- Failed OOD job: resubmit `eval_ood`; resumable CSV rows should prevent lost
  work.
- Missing environment dependency: fix image or CPFS environment, then run
  `smoke_env` before submitting expensive jobs.

Do not let a failed worker silently mark a run as complete.

## Required Code Adjustments

1. Add `pta/scripts/dlc` submission and worker scripts.
2. Add unit tests for command construction, argument validation, and OOD
   optional ablation seed coverage.
3. Update `run_ood_eval_v2.py` so `m7_noprobe` and `m7_nobelief` evaluate seeds
   `42`, `0`, and `1` for the approved ablation cycle.
4. Add a probe repo DLC runbook documenting DSW setup, dry-run, smoke, ablation
   submission, OOD submission, and result collection.
5. Update the ARIS mirror logs to point to the DLC path as the preferred
   acceleration route for the ablation-first cycle.

## Non-Goals

- No DLC worker cron.
- No per-worker opencode, Claude, or ARIS agent loop.
- No automatic paper writing from DLC jobs.
- No new experiment variants beyond `m7_noprobe` and `m7_nobelief`.
- No large refactor of existing training scripts.

## Verification

Before any real DLC submission:

```bash
python -m pytest tests/test_dlc_submit_jobs.py tests/test_run_ood_eval_v2.py -q
bash pta/scripts/dlc/run_task.sh smoke_env
python pta/scripts/dlc/submit_jobs.py --suite ablation --variants no_probe --seeds 42 --dry-run
```

On DSW:

```bash
python pta/scripts/dlc/submit_jobs.py --suite smoke --name pta_smoke
python pta/scripts/dlc/submit_jobs.py --suite ablation --variants no_probe no_belief --seeds 42 0 1
```

After all six checkpoints exist:

```bash
python pta/scripts/dlc/submit_jobs.py --suite ood-ablation
```
