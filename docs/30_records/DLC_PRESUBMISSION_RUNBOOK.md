# DLC Pre-Submission Reliability Runbook

Date: 2026-05-01

This runbook prepares the NeurIPS pre-submission reliability package and contains the gated real DLC submit commands to run from DSW. Execute the commands in order: smoke first, audits second, and extra training only if the audit gates pass.

## Purpose

The current paper is submission-ready structurally, but the core evidence is fragile: M7 improves on OOD elastoplastic with 3 seeds and high variance. Before spending more GPU on extra seeds, run two audits:

- Probe integrity: the current 3-step zero-residual probe must measurably perturb particles.
- Encoder sensitivity: M7 evaluation must not swing strongly when the freshly constructed evaluation encoder is initialized with different seeds.

Only if both audits pass should the extra M1/M7 seed jobs be submitted.

## DSW Environment

Run this once per DSW shell:

```bash
export PTA_CODE_ROOT=/cpfs/shared/simulation/zhuzihou/dev/probe-then-act
export GENESIS_ROOT=/cpfs/shared/simulation/zhuzihou/dev/Genesis
export GENESIS_VENV=$GENESIS_ROOT/.venv
export PYTHON_BIN=$GENESIS_VENV/bin/python
export DLC_RESULTS_ROOT=$PTA_CODE_ROOT/results/dlc
export DLC_BIN=/cpfs/shared/simulation/zhuzihou/dev/usd-scene-physics-prep/dlc
export DLC_IMAGE=pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang
export DLC_DATA_SOURCES=d-mzps5b7joy2axmqpa8,d-d49o5g0h2818sw8j1g,d-8wz4emfs21s5ajs9oz
export DLC_WORKSPACE_ID=270969
export DLC_RESOURCE_ID=quota1r947pmazvk
export DLC_GPU_COUNT=1
export PYOPENGL_PLATFORM=egl
export EGL_DEVICE_ID=0
cd "$PTA_CODE_ROOT"
```

`PTA_CODE_ROOT` and `GENESIS_ROOT` must be worker-visible CPFS paths because `launch_job.sh` embeds `PTA_CODE_ROOT` into the command executed inside the DLC worker. Do not use a DSW-local `/shared/smartbot/...` path here unless you have verified it is mounted inside DLC workers.

## Local DSW Preflight

```bash
git status --short --branch
"$PYTHON_BIN" -m pytest \
  tests/test_pre_submission_audit.py \
  tests/test_dlc_submit_jobs.py \
  tests/test_dlc_shell_contract.py \
  -q
"$PYTHON_BIN" -m py_compile tools/pre_submission_audit.py
```

Expected:

- The branch is the DLC preparation branch or synced `main` containing `tools/pre_submission_audit.py`.
- The targeted tests pass.
- `py_compile` exits without output.

## Smoke Job

Dry-run:

```bash
DLC_DRY_RUN=1 "$PYTHON_BIN" -m pta.scripts.dlc.submit_jobs \
  --suite smoke \
  --name pta_presub_smoke \
  --gpu-count 1 \
  --dry-run
```

Submit:

```bash
"$PYTHON_BIN" -m pta.scripts.dlc.submit_jobs \
  --suite smoke \
  --name pta_presub_smoke \
  --gpu-count 1
```

Monitor:

```bash
"$DLC_BIN" get job <SMOKE_JOB_ID> \
  --endpoint=pai-dlc.cn-beijing.aliyuncs.com \
  --region=cn-beijing
```

Gate G0 passes only if the job succeeds and the worker record has `exit_code=0`.

## Checkpoint Prerequisite For G2

The encoder-sensitivity audit re-evaluates existing M7 seed 42. Before submitting `pta_encoder_sensitivity_m7_ep`, restore or provide the checkpoint bundle so this file exists on the CPFS checkout:

```text
checkpoints/m7_pta_seed42/best/best_model.zip
```

Verify it before G2:

```bash
test -f checkpoints/m7_pta_seed42/best/best_model.zip
"$PYTHON_BIN" - <<'PY'
from stable_baselines3 import PPO

path = "checkpoints/m7_pta_seed42/best/best_model.zip"
model = PPO.load(path, device="cpu")
print(path, model.num_timesteps)
PY
```

If the file is missing, restore the checkpoint bundle first. Do not submit `pta_encoder_sensitivity_m7_ep` until the checkpoint loads successfully.

## Audit Jobs

Submit probe integrity jobs after G0 passes. These jobs do not need policy checkpoints:

```bash
for split in id_sand ood_elastoplastic ood_snow ood_sand_hard ood_sand_soft; do
  bash pta/scripts/dlc/launch_job.sh "pta_probe_integrity_${split}" 0 1 "$DLC_DATA_SOURCES" \
    custom env PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 "$PYTHON_BIN" -u tools/pre_submission_audit.py \
      --mode probe-integrity \
      --split "$split" \
      --seed 123 \
      --n-probes 3 \
      --settle-steps 80 \
      --residual-scale 0.05
done
```

Submit encoder sensitivity job after G0 passes and after `checkpoints/m7_pta_seed42/best/best_model.zip` has been restored and verified:

```bash
bash pta/scripts/dlc/launch_job.sh pta_encoder_sensitivity_m7_ep 0 1 "$DLC_DATA_SOURCES" \
  custom env PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 "$PYTHON_BIN" -u tools/pre_submission_audit.py \
    --mode encoder-sensitivity \
    --method m7_pta \
    --seed 42 \
    --split ood_elastoplastic \
    --encoder-seeds 11 22 33 \
    --n-episodes 3 \
    --max-transfer-range-pp 5.0 \
    --residual-scale 0.05
```

Monitor each returned job id:

```bash
"$DLC_BIN" get job <JOB_ID> \
  --endpoint=pai-dlc.cn-beijing.aliyuncs.com \
  --region=cn-beijing
```

Audit gate check:

```bash
"$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("results/presub").glob("audit_probe_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    probe = data["probe_displacement"]
    print(path.name, "probe_rms_m", probe["rms_m"], "probe_max_m", probe["max_m"], "persistent_fraction", data["persistent_fraction_rms"])

for path in sorted(Path("results/presub").glob("audit_encoder_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    print(path.name, "passes", data["passes"], "transfer_range_pp", data["transfer_range_pp"], "total_failed_episodes", data["total_failed_episodes"], "reasons", data["reasons"])
PY
```

Pass criteria:

- G1 probe integrity: `audit_probe_ood_elastoplastic_seed123.json` has `probe_rms_m >= 0.0001` or `probe_max_m >= 0.001`.
- G2 encoder sensitivity: `audit_encoder_m7_pta_s42_ood_elastoplastic.json` has `passes == true`, `transfer_range_pp <= 5.0`, and `total_failed_episodes == 0`.

If G1 or G2 fails, do not submit extra seed training jobs.

## Extra Seed Training Jobs

Submit only after G1 and G2 pass.

M1 seed 2 and seed 3:

```bash
for seed in 2 3; do
  bash pta/scripts/dlc/launch_job.sh "pta_presub_m1_s${seed}" 0 1 "$DLC_DATA_SOURCES" \
    custom env PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 "$PYTHON_BIN" -u pta/scripts/train_baselines.py \
      --method m1 \
      --seed "$seed" \
      --total-timesteps 500000 \
      --residual-scale 0.05 \
      --horizon 500 \
      --eval-freq 50000
done
```

M7 seed 2 and seed 3:

```bash
for seed in 2 3; do
  bash pta/scripts/dlc/launch_job.sh "pta_presub_m7_s${seed}" 0 1 "$DLC_DATA_SOURCES" \
    custom env PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 "$PYTHON_BIN" -u pta/scripts/train_m7.py \
      --seed "$seed" \
      --total-timesteps 500000 \
      --residual-scale 0.05 \
      --horizon 500 \
      --latent-dim 16 \
      --n-probes 3 \
      --eval-freq 50000
done
```

Checkpoint verification after all four jobs succeed:

```bash
"$PYTHON_BIN" - <<'PY'
from stable_baselines3 import PPO

paths = [
    "checkpoints/m1_reactive_seed2/best/best_model.zip",
    "checkpoints/m1_reactive_seed3/best/best_model.zip",
    "checkpoints/m7_pta_seed2/best/best_model.zip",
    "checkpoints/m7_pta_seed3/best/best_model.zip",
]
for path in paths:
    model = PPO.load(path, device="cpu")
    print(path, model.num_timesteps)
PY
```

## Extra Seed Evaluation Job

Submit only after all four extra checkpoints load:

```bash
bash pta/scripts/dlc/launch_job.sh pta_presub_eval_extra_s2_s3 0 1 "$DLC_DATA_SOURCES" \
  custom env PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 "$PYTHON_BIN" -u tools/pre_submission_audit.py \
    --mode eval-extra-seeds \
    --methods m1_reactive m7_pta \
    --seeds 2 3 \
    --splits id_sand ood_elastoplastic ood_snow ood_sand_hard ood_sand_soft \
    --n-episodes 10 \
    --residual-scale 0.05 \
    --output results/presub/ood_eval_extra_seeds.csv
```

Verify row count:

```bash
"$PYTHON_BIN" - <<'PY'
import csv
from collections import Counter

path = "results/presub/ood_eval_extra_seeds.csv"
with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
print(path, len(rows), Counter((row["method"], row["split"]) for row in rows))
assert len(rows) == 20
PY
```

Summarize five-seed elastoplastic evidence:

```bash
"$PYTHON_BIN" -u tools/pre_submission_audit.py \
  --mode summarize-five-seed \
  --inputs results/ood_eval_per_seed.csv results/presub/ood_eval_extra_seeds.csv
```

Paper-strengthening gate G4 passes only if:

- `n_pairs=5`
- `mean_delta_pp > 0`
- `positive_pairs >= 3/5`

## Final Paper Step

Do not strengthen the paper until G1, G2, and G4 are read. Regardless of extra-seed outcome, add the NeurIPS checklist before final upload and rebuild:

```bash
cd "$PTA_CODE_ROOT/paper"
make nips2026
make all
```
