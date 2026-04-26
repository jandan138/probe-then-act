# DLC Execution Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a DLC submission and worker execution layer so the ablation-first diagnostic runs can execute on Alibaba PAI-DLC from a DSW machine.

**Architecture:** Keep execution logic inside `probe-then-act`: DSW runs `submit_jobs.py`, DLC workers run `run_task.sh`, and all outputs land under existing `checkpoints/`, `logs/`, `results/`, plus `results/dlc/` metadata. ARIS remains a record layer and does not run inside DLC workers.

**Tech Stack:** Python 3.11, Bash, pytest, Alibaba PAI-DLC CLI, Genesis runtime, Stable-Baselines3.

---

## File Map

- Create `pta/scripts/dlc/launch_job.sh`: `dlc submit pytorchjob` wrapper modeled after `usd-scene-physics-prep`.
- Create `pta/scripts/dlc/run_task.sh`: worker dispatcher for `smoke_env`, `train_ablation`, `eval_ood`, and `custom`.
- Create `pta/scripts/dlc/submit_jobs.py`: DSW-side suite submitter with dry-run and manifest support.
- Create `pta/scripts/dlc/submit_ablation_sweep.sh`: small convenience wrapper for the approved six ablation jobs.
- Create `tests/test_dlc_submit_jobs.py`: unit tests for command generation, dry-run safety, and manifest writing.
- Modify `pta/scripts/run_ood_eval_v2.py`: evaluate `m7_noprobe` and `m7_nobelief` seeds `42`, `0`, `1`.
- Modify `tests/test_run_ood_eval_v2.py`: regression for optional ablation seed coverage.
- Create `docs/30_records/DLC_EXECUTION_RUNBOOK.md`: DSW/DLC usage.
- Modify ARIS mirror docs in `/home/zhuzihou/dev/Auto-claude-code-research-in-sleep/projects/probe-then-act/`: record DLC route.

---

### Task 1: Make OOD Ablation Seed Coverage Match the Approved Plan

**Files:**
- Modify: `pta/scripts/run_ood_eval_v2.py`
- Modify: `tests/test_run_ood_eval_v2.py`

- [ ] **Step 1: Write the failing test**

Append this test to `tests/test_run_ood_eval_v2.py`:

```python
def test_optional_ablation_methods_cover_three_approved_seeds():
    from pta.scripts import run_ood_eval_v2

    assert run_ood_eval_v2.METHODS["m7_noprobe"]["seeds"] == [42, 0, 1]
    assert run_ood_eval_v2.METHODS["m7_nobelief"]["seeds"] == [42, 0, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
source /home/zhuzihou/dev/Genesis/.venv/bin/activate
pytest tests/test_run_ood_eval_v2.py::test_optional_ablation_methods_cover_three_approved_seeds -q
```

Expected: FAIL because both optional ablation method configs currently list only `[42, 0]`.

- [ ] **Step 3: Implement minimal code**

Change `pta/scripts/run_ood_eval_v2.py`:

```python
"m7_noprobe": {
    "seeds": [42, 0, 1],
    "ckpt_pattern": "checkpoints/m7_pta_noprobe_seed{seed}/best/best_model",
    "use_privileged": False,
    "use_m7_env": True,
    "ablation": "no_probe",
},
"m7_nobelief": {
    "seeds": [42, 0, 1],
    "ckpt_pattern": "checkpoints/m7_pta_nobelief_seed{seed}/best/best_model",
    "use_privileged": False,
    "use_m7_env": True,
    "ablation": "no_belief",
},
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
source /home/zhuzihou/dev/Genesis/.venv/bin/activate
pytest tests/test_run_ood_eval_v2.py::test_optional_ablation_methods_cover_three_approved_seeds -q
```

Expected: PASS.

---

### Task 2: Add DLC Submitter Pure Logic and Tests

**Files:**
- Create: `pta/scripts/dlc/submit_jobs.py`
- Create: `tests/test_dlc_submit_jobs.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_dlc_submit_jobs.py`:

```python
import json
from pathlib import Path

import pytest

from pta.scripts.dlc import submit_jobs


def test_ablation_suite_builds_one_job_per_variant_seed():
    jobs = submit_jobs.build_job_specs(
        suite="ablation",
        name="pta_ablation",
        variants=["no_probe", "no_belief"],
        seeds=[42, 0, 1],
        gpu_count=1,
        data_sources="d-a,d-b",
    )

    assert [job.job_name for job in jobs] == [
        "pta_ablation_no_probe_s42",
        "pta_ablation_no_probe_s0",
        "pta_ablation_no_probe_s1",
        "pta_ablation_no_belief_s42",
        "pta_ablation_no_belief_s0",
        "pta_ablation_no_belief_s1",
    ]
    assert jobs[0].command_args == "train_ablation no_probe 42"
    assert jobs[-1].command_args == "train_ablation no_belief 1"
    assert all(job.gpu_count == 1 for job in jobs)
    assert all(job.data_sources == "d-a,d-b" for job in jobs)


def test_ood_ablation_suite_builds_single_eval_job():
    jobs = submit_jobs.build_job_specs(
        suite="ood-ablation",
        name="pta_ood_ablation",
        variants=[],
        seeds=[],
        gpu_count=1,
        data_sources=None,
    )

    assert len(jobs) == 1
    assert jobs[0].job_name == "pta_ood_ablation"
    assert jobs[0].command_args == (
        "eval_ood --residual-scale 0.05 --methods m7_noprobe m7_nobelief"
    )


def test_smoke_suite_builds_single_smoke_job():
    jobs = submit_jobs.build_job_specs(
        suite="smoke",
        name="pta_smoke",
        variants=[],
        seeds=[],
        gpu_count=1,
        data_sources=None,
    )

    assert len(jobs) == 1
    assert jobs[0].command_args == "smoke_env"


def test_rejects_unknown_ablation_variant():
    with pytest.raises(ValueError, match="unsupported variant"):
        submit_jobs.build_job_specs(
            suite="ablation",
            name="pta_ablation",
            variants=["bad_variant"],
            seeds=[42],
            gpu_count=1,
            data_sources=None,
        )


def test_dry_run_does_not_call_subprocess_and_writes_manifest(tmp_path, monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("subprocess should not be called in dry-run")

    monkeypatch.setattr(submit_jobs.subprocess, "run", fake_run)
    jobs = submit_jobs.build_job_specs(
        suite="smoke",
        name="pta_smoke",
        variants=[],
        seeds=[],
        gpu_count=1,
        data_sources="d-a",
    )

    submit_jobs.submit_specs(
        jobs,
        repo_root=tmp_path,
        launch_script=Path("pta/scripts/dlc/launch_job.sh"),
        dry_run=True,
        manifest_path=tmp_path / "jobs.jsonl",
    )

    assert calls == []
    rows = [
        json.loads(line)
        for line in (tmp_path / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["job_name"] == "pta_smoke"
    assert rows[0]["suite"] == "smoke"
    assert rows[0]["command_args"] == "smoke_env"
    assert rows[0]["dry_run"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source /home/zhuzihou/dev/Genesis/.venv/bin/activate
pytest tests/test_dlc_submit_jobs.py -q
```

Expected: FAIL because `pta/scripts/dlc/submit_jobs.py` does not exist.

- [ ] **Step 3: Implement minimal submitter**

Create `pta/scripts/dlc/submit_jobs.py` with:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


SUPPORTED_VARIANTS = {"no_probe", "no_belief"}
DEFAULT_SEEDS = [42, 0, 1]
DEFAULT_VARIANTS = ["no_probe", "no_belief"]
DEFAULT_DATA_SOURCES = "d-mzps5b7joy2axmqpa8,d-d49o5g0h2818sw8j1g,d-8wz4emfs21s5ajs9oz"


@dataclass(frozen=True)
class JobSpec:
    suite: str
    job_name: str
    chunk_id: int
    chunk_total: int
    command_args: str
    gpu_count: int
    data_sources: str | None = None
    variant: str | None = None
    seed: int | None = None


def _validate_gpu_count(gpu_count: int) -> None:
    if gpu_count not in {1, 2, 4, 8}:
        raise ValueError("gpu_count must be one of 1, 2, 4, 8")


def build_job_specs(
    *,
    suite: str,
    name: str,
    variants: list[str],
    seeds: list[int],
    gpu_count: int,
    data_sources: str | None,
) -> list[JobSpec]:
    _validate_gpu_count(gpu_count)
    if suite == "ablation":
        chosen_variants = variants or DEFAULT_VARIANTS
        chosen_seeds = seeds or DEFAULT_SEEDS
        jobs: list[JobSpec] = []
        for variant in chosen_variants:
            if variant not in SUPPORTED_VARIANTS:
                raise ValueError(f"unsupported variant: {variant}")
            for seed in chosen_seeds:
                if not isinstance(seed, int):
                    raise ValueError(f"seed must be int: {seed!r}")
                jobs.append(
                    JobSpec(
                        suite=suite,
                        job_name=f"{name}_{variant}_s{seed}",
                        chunk_id=len(jobs),
                        chunk_total=len(chosen_variants) * len(chosen_seeds),
                        command_args=f"train_ablation {variant} {seed}",
                        gpu_count=gpu_count,
                        data_sources=data_sources,
                        variant=variant,
                        seed=seed,
                    )
                )
        return jobs
    if suite == "ood-ablation":
        return [
            JobSpec(
                suite=suite,
                job_name=name,
                chunk_id=0,
                chunk_total=1,
                command_args=(
                    "eval_ood --residual-scale 0.05 "
                    "--methods m7_noprobe m7_nobelief"
                ),
                gpu_count=gpu_count,
                data_sources=data_sources,
            )
        ]
    if suite == "smoke":
        return [
            JobSpec(
                suite=suite,
                job_name=name,
                chunk_id=0,
                chunk_total=1,
                command_args="smoke_env",
                gpu_count=gpu_count,
                data_sources=data_sources,
            )
        ]
    raise ValueError(f"unsupported suite: {suite}")


def append_manifest(path: Path, spec: JobSpec, *, dry_run: bool, returncode: int | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        **asdict(spec),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "returncode": returncode,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def submit_specs(
    specs: list[JobSpec],
    *,
    repo_root: Path,
    launch_script: Path,
    dry_run: bool,
    manifest_path: Path,
) -> None:
    for spec in specs:
        data_sources = spec.data_sources or DEFAULT_DATA_SOURCES
        cmd = [
            "bash",
            str(launch_script),
            spec.job_name,
            str(spec.chunk_id),
            str(spec.chunk_total),
            data_sources,
            spec.command_args,
        ]
        env = os.environ.copy()
        env["DLC_GPU_COUNT"] = str(spec.gpu_count)
        print(" ".join(cmd), flush=True)
        if dry_run:
            append_manifest(manifest_path, spec, dry_run=True, returncode=None)
            continue
        completed = subprocess.run(cmd, cwd=repo_root, env=env, check=True)
        append_manifest(
            manifest_path,
            spec,
            dry_run=False,
            returncode=completed.returncode,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit probe-then-act DLC jobs")
    parser.add_argument("--suite", choices=["ablation", "ood-ablation", "smoke"], required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--gpu-count", type=int, default=int(os.environ.get("DLC_GPU_COUNT", "1")))
    parser.add_argument("--data-sources", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    default_names = {
        "ablation": "pta_ablation",
        "ood-ablation": "pta_ood_ablation",
        "smoke": "pta_smoke",
    }
    specs = build_job_specs(
        suite=args.suite,
        name=args.name or default_names[args.suite],
        variants=args.variants or [],
        seeds=args.seeds or [],
        gpu_count=args.gpu_count,
        data_sources=args.data_sources,
    )
    submit_specs(
        specs,
        repo_root=repo_root,
        launch_script=repo_root / "pta" / "scripts" / "dlc" / "launch_job.sh",
        dry_run=args.dry_run,
        manifest_path=repo_root / "results" / "dlc" / "jobs.jsonl",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
source /home/zhuzihou/dev/Genesis/.venv/bin/activate
pytest tests/test_dlc_submit_jobs.py -q
```

Expected: PASS.

---

### Task 3: Add DLC Launch and Worker Scripts

**Files:**
- Create: `pta/scripts/dlc/launch_job.sh`
- Create: `pta/scripts/dlc/run_task.sh`
- Create: `pta/scripts/dlc/submit_ablation_sweep.sh`

- [ ] **Step 1: Add worker scripts**

Create `pta/scripts/dlc/launch_job.sh`:

```bash
#!/bin/bash
set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: bash launch_job.sh <TASK_NAME> <CHUNK_ID> <CHUNK_TOTAL> [DATA_SOURCES] [COMMAND_ARGS]" >&2
    exit 1
fi

TASK_NAME=$1
CHUNK_ID=$2
CHUNK_TOTAL=$3
DATA_SOURCES=${4:-"d-mzps5b7joy2axmqpa8,d-d49o5g0h2818sw8j1g,d-8wz4emfs21s5ajs9oz"}
COMMAND_ARGS=${5:-"smoke_env"}

WORKSPACE_ID=${DLC_WORKSPACE_ID:-"270969"}
IMAGE=${DLC_IMAGE:-"dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai-training-algorithm/pytorch:py311-cu126"}
CODE_ROOT=${PTA_CODE_ROOT:-${DLC_CODE_ROOT:-"/cpfs/shared/simulation/zhuzihou/dev/probe-then-act"}}
DLC_BIN=${DLC_BIN:-"$CODE_ROOT/dlc"}
GPU_COUNT=${DLC_GPU_COUNT:-1}

case "$GPU_COUNT" in
    1) WORKER_GPU=1; WORKER_CPU=${DLC_WORKER_CPU:-14}; WORKER_MEMORY=${DLC_WORKER_MEMORY:-100Gi}; WORKER_SHARED_MEMORY=${DLC_WORKER_SHARED_MEMORY:-100Gi}; RESOURCE_ID=${DLC_RESOURCE_ID:-quota1r947pmazvk};;
    2) WORKER_GPU=2; WORKER_CPU=${DLC_WORKER_CPU:-28}; WORKER_MEMORY=${DLC_WORKER_MEMORY:-200Gi}; WORKER_SHARED_MEMORY=${DLC_WORKER_SHARED_MEMORY:-200Gi}; RESOURCE_ID=${DLC_RESOURCE_ID:-quota1r947pmazvk};;
    4) WORKER_GPU=4; WORKER_CPU=${DLC_WORKER_CPU:-56}; WORKER_MEMORY=${DLC_WORKER_MEMORY:-400Gi}; WORKER_SHARED_MEMORY=${DLC_WORKER_SHARED_MEMORY:-400Gi}; RESOURCE_ID=${DLC_RESOURCE_ID:-quota1r947pmazvk};;
    8) WORKER_GPU=8; WORKER_CPU=${DLC_WORKER_CPU:-128}; WORKER_MEMORY=${DLC_WORKER_MEMORY:-960Gi}; WORKER_SHARED_MEMORY=${DLC_WORKER_SHARED_MEMORY:-960Gi}; RESOURCE_ID=${DLC_RESOURCE_ID:-quotaksvqq2oh2pg};;
    *) echo "ERROR: unsupported DLC_GPU_COUNT=$GPU_COUNT" >&2; exit 1;;
esac

if [ ! -x "$DLC_BIN" ]; then
    echo "ERROR: DLC binary not found or not executable at $DLC_BIN" >&2
    exit 1
fi

JOB_NAME="${TASK_NAME}_${CHUNK_ID}_${CHUNK_TOTAL}"
echo "Submitting Job: $JOB_NAME"
echo "Code Root: $CODE_ROOT"
echo "Resolved config -> GPU=$WORKER_GPU CPU=$WORKER_CPU Memory=$WORKER_MEMORY SharedMem=$WORKER_SHARED_MEMORY Resource=$RESOURCE_ID"

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
    --command="bash $CODE_ROOT/pta/scripts/dlc/run_task.sh ${COMMAND_ARGS}"
```

Create `pta/scripts/dlc/run_task.sh`:

```bash
#!/bin/bash
set -euo pipefail

CODE_ROOT=${PTA_CODE_ROOT:-${DLC_CODE_ROOT:-"/cpfs/shared/simulation/zhuzihou/dev/probe-then-act"}}
GENESIS_ROOT=${GENESIS_ROOT:-"/cpfs/shared/simulation/zhuzihou/dev/Genesis"}
GENESIS_VENV=${GENESIS_VENV:-"$GENESIS_ROOT/.venv"}

export PYTHONUNBUFFERED=1
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}
export PYTHONPATH="$CODE_ROOT:$GENESIS_ROOT:${PYTHONPATH:-}"
cd "$CODE_ROOT"

RUN_ID=${DLC_RUN_ID:-"$(date -u +%Y%m%dT%H%M%SZ)_${1:-no_mode}_${HOSTNAME:-worker}"}
RUN_DIR="$CODE_ROOT/results/dlc/runs"
mkdir -p "$RUN_DIR" "$CODE_ROOT/results/dlc/logs"
RUN_RECORD="$RUN_DIR/$RUN_ID.json"

write_record() {
    local exit_code=$1
    local checkpoint_hint=${2:-""}
    local result_hint=${3:-""}
    python - "$RUN_RECORD" "$exit_code" "$checkpoint_hint" "$result_hint" "$*" <<'PY'
import json, os, socket, sys
from datetime import datetime, timezone
path, exit_code, checkpoint_hint, result_hint, command = sys.argv[1:6]
payload = {
    "run_id": os.environ.get("RUN_ID", ""),
    "mode": os.environ.get("RUN_MODE", ""),
    "command": command,
    "cwd": os.getcwd(),
    "start_time": os.environ.get("RUN_START_TIME", ""),
    "end_time": datetime.now(timezone.utc).isoformat(),
    "exit_code": int(exit_code),
    "hostname": socket.gethostname(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "checkpoint_hint": checkpoint_hint,
    "result_hint": result_hint,
}
with open(path, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
PY
}

if [ -f "$GENESIS_VENV/bin/activate" ]; then
    # shellcheck disable=SC1090
    source "$GENESIS_VENV/bin/activate"
fi

python - <<'PY'
import importlib.util, sys
required = ["genesis", "torch", "stable_baselines3", "gymnasium", "numpy", "pandas"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
print("Python:", sys.executable, flush=True)
if missing:
    raise SystemExit("missing required modules: " + ", ".join(missing))
optional = ["sb3_contrib"]
for name in optional:
    print(f"optional {name}: {'ok' if importlib.util.find_spec(name) else 'missing'}", flush=True)
PY

if [ $# -eq 0 ]; then
    echo "Usage: run_task.sh <smoke_env|train_ablation|eval_ood|custom> [args...]" >&2
    exit 2
fi

export RUN_ID
export RUN_START_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
export RUN_MODE="$1"
MODE=$1
shift

case "$MODE" in
    smoke_env)
        python - <<'PY'
import torch
print("cuda_available", torch.cuda.is_available(), flush=True)
print("cuda_device_count", torch.cuda.device_count(), flush=True)
PY
        python pta/scripts/train_m7.py --help >/dev/null
        write_record 0 "" "smoke_env"
        ;;
    train_ablation)
        if [ $# -ne 2 ]; then
            echo "Usage: run_task.sh train_ablation <no_probe|no_belief> <seed>" >&2
            exit 2
        fi
        VARIANT=$1
        SEED=$2
        case "$VARIANT" in
            no_probe|no_belief) ;;
            *) echo "ERROR: unsupported ablation variant: $VARIANT" >&2; exit 2;;
        esac
        case "$SEED" in
            ''|*[!0-9]*) echo "ERROR: seed must be a non-negative integer: $SEED" >&2; exit 2;;
        esac
        python -u pta/scripts/train_m7.py --ablation "$VARIANT" --seed "$SEED" --total-timesteps 500000 --residual-scale 0.05
        if [ "$VARIANT" = "no_probe" ]; then
            CKPT="checkpoints/m7_pta_noprobe_seed${SEED}/m7_pta_final"
        else
            CKPT="checkpoints/m7_pta_nobelief_seed${SEED}/m7_pta_final"
        fi
        write_record 0 "$CKPT" ""
        ;;
    eval_ood)
        python -u pta/scripts/run_ood_eval_v2.py "$@"
        write_record 0 "" "results/main_results.csv"
        ;;
    custom)
        "$@"
        write_record 0 "" ""
        ;;
    *)
        echo "ERROR: unsupported mode: $MODE" >&2
        exit 2
        ;;
esac
```

Create `pta/scripts/dlc/submit_ablation_sweep.sh`:

```bash
#!/bin/bash
set -euo pipefail

python pta/scripts/dlc/submit_jobs.py \
    --suite ablation \
    --variants no_probe no_belief \
    --seeds 42 0 1 \
    "$@"
```

- [ ] **Step 2: Mark scripts executable**

Run:

```bash
chmod +x pta/scripts/dlc/launch_job.sh pta/scripts/dlc/run_task.sh pta/scripts/dlc/submit_ablation_sweep.sh
```

- [ ] **Step 3: Run dry-run verification**

Run:

```bash
source /home/zhuzihou/dev/Genesis/.venv/bin/activate
python pta/scripts/dlc/submit_jobs.py --suite ablation --variants no_probe --seeds 42 --dry-run
```

Expected: prints one `bash ... launch_job.sh ... "train_ablation no_probe 42"` command and appends a dry-run row to `results/dlc/jobs.jsonl`.

---

### Task 4: Add Documentation and ARIS Mirror Records

**Files:**
- Create: `docs/30_records/DLC_EXECUTION_RUNBOOK.md`
- Modify: `/home/zhuzihou/dev/Auto-claude-code-research-in-sleep/projects/probe-then-act/AUTOMATION_PLAN.md`
- Modify: `/home/zhuzihou/dev/Auto-claude-code-research-in-sleep/projects/probe-then-act/EXECUTION_LOG.md`

- [ ] **Step 1: Add probe repo runbook**

Create `docs/30_records/DLC_EXECUTION_RUNBOOK.md` with:

```markdown
# DLC Execution Runbook

## Purpose

Run the ablation-first diagnostic cycle on Alibaba PAI-DLC from a DSW machine.
DLC workers execute deterministic training or evaluation commands only. ARIS
and result-to-claim remain outside worker jobs.

## Required CPFS Layout

```text
/cpfs/shared/simulation/zhuzihou/dev/probe-then-act
/cpfs/shared/simulation/zhuzihou/dev/Auto-claude-code-research-in-sleep
/cpfs/shared/simulation/zhuzihou/dev/Genesis
```

## DSW Environment

```bash
export PTA_CODE_ROOT=/cpfs/shared/simulation/zhuzihou/dev/probe-then-act
export ARIS_CODE_ROOT=/cpfs/shared/simulation/zhuzihou/dev/Auto-claude-code-research-in-sleep
export GENESIS_ROOT=/cpfs/shared/simulation/zhuzihou/dev/Genesis
export GENESIS_VENV=/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv
export DLC_WORKSPACE_ID=270969
export DLC_GPU_COUNT=1
```

## Smoke

```bash
python pta/scripts/dlc/submit_jobs.py --suite smoke --name pta_smoke
```

## Dry Run

```bash
python pta/scripts/dlc/submit_jobs.py \
  --suite ablation \
  --variants no_probe no_belief \
  --seeds 42 0 1 \
  --dry-run
```

## Submit Ablations

```bash
bash pta/scripts/dlc/submit_ablation_sweep.sh
```

## Submit Ablation OOD

Run after all six ablation checkpoints exist.

```bash
python pta/scripts/dlc/submit_jobs.py --suite ood-ablation
```

## Outputs

- Training checkpoints: `checkpoints/m7_pta_noprobe_seed*/`, `checkpoints/m7_pta_nobelief_seed*/`
- OOD results: `results/ood_eval_per_seed.csv`, `results/main_results.csv`
- DLC submit manifest: `results/dlc/jobs.jsonl`
- Worker run records: `results/dlc/runs/*.json`
```

- [ ] **Step 2: Update ARIS mirror**

Append a short entry to the Auto repo execution log:

```markdown
### 2026-04-26 DLC Execution Route Selected

- Strategy B selected: add a project-level DLC execution layer in `probe-then-act`.
- DLC workers will run deterministic `train_ablation` and `eval_ood` jobs only.
- ARIS remains the research record and result-to-claim layer; it does not run inside each DLC worker.
- Immediate target: parallelize `m7_noprobe` and `m7_nobelief` seeds `42/0/1`, then rerun corrected resumable OOD.
```

Add one line to the plan near the ablation phase:

```markdown
Acceleration route: if local 4090 remains the bottleneck, submit R001-R006 through `pta/scripts/dlc/submit_jobs.py` from DSW; keep local cron disabled or ignored for those cloud runs.
```

---

### Task 5: Final Verification

**Files:**
- All files touched above.

- [ ] **Step 1: Run focused tests**

Run:

```bash
source /home/zhuzihou/dev/Genesis/.venv/bin/activate
pytest tests/test_dlc_submit_jobs.py tests/test_run_ood_eval_v2.py -q
```

Expected: all selected tests pass.

- [ ] **Step 2: Run dry-run**

Run:

```bash
source /home/zhuzihou/dev/Genesis/.venv/bin/activate
python pta/scripts/dlc/submit_jobs.py --suite ablation --variants no_probe --seeds 42 --dry-run
```

Expected: command prints and `results/dlc/jobs.jsonl` receives one dry-run row.

- [ ] **Step 3: Inspect changed files**

Run:

```bash
git status --short
git diff -- pta/scripts/run_ood_eval_v2.py tests/test_run_ood_eval_v2.py tests/test_dlc_submit_jobs.py pta/scripts/dlc docs/30_records/DLC_EXECUTION_RUNBOOK.md docs/superpowers/specs/2026-04-26-dlc-execution-layer-design.md docs/superpowers/plans/2026-04-26-dlc-execution-layer.md
```

Expected: changes are limited to DLC execution layer, OOD ablation seed coverage, tests, and docs.
