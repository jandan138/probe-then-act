# DLC Pre-Submission Reliability Runbook

Date: 2026-05-01

This runbook prepares the NeurIPS pre-submission reliability package and contains the gated real DLC submit commands to run from DSW. Execute the commands in order: smoke first, audits second, and extra training only if the audit gates pass.

## Purpose

The current paper is submission-ready structurally, but the core evidence is fragile: M7 improves on OOD elastoplastic with 3 seeds and high variance. Before spending more GPU on extra seeds, run two audits:

- Probe integrity: the current 3-step zero-residual probe must measurably perturb particles.
- G2 evaluation: the old `random_eval_encoder_stress` job is a non-claim-bearing diagnostic; corrected M7 G2 is a `matched-encoder-audit` using the encoder sidecar paired with the policy checkpoint.

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

`PTA_CODE_ROOT` and `GENESIS_ROOT` must be worker-visible CPFS paths. `launch_job.sh` uses the local `PTA_CODE_ROOT` to choose the `run_task.sh` script path, but it does not inject `PTA_CODE_ROOT` into the worker environment before `run_task.sh` starts. For jobs that must run from an alternate code root, submit with a direct DLC command whose `--command` begins with `PTA_CODE_ROOT=<alternate-root> ... bash <alternate-root>/pta/scripts/dlc/run_task.sh ...`.

```bash
WORKER_COMMAND="PTA_CODE_ROOT=/cpfs/shared/simulation/zhuzihou/dev/probe-then-act-matched-seed42 GENESIS_ROOT=$GENESIS_ROOT GENESIS_VENV=$GENESIS_VENV PYTHON_BIN=$PYTHON_BIN PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 bash /cpfs/shared/simulation/zhuzihou/dev/probe-then-act-matched-seed42/pta/scripts/dlc/run_task.sh custom $PYTHON_BIN -u pta/scripts/train_m7.py --seed 42 --total-timesteps 500000 --eval-freq 10000 --horizon 500 --residual-scale 0.05"
"$DLC_BIN" submit pytorchjob --name=pta_m7_matched_seed42_full_0_1 --workers=1 --worker_gpu=1 --worker_cpu=14 --worker_memory=100Gi --worker_shared_memory=100Gi --worker_image="$DLC_IMAGE" --workspace_id="$DLC_WORKSPACE_ID" --resource_id="$DLC_RESOURCE_ID" --data_sources="$DLC_DATA_SOURCES" --oversold_type=ForbiddenQuotaOverSold --priority 7 --command="$WORKER_COMMAND"
```

Do not use a DSW-local `/shared/smartbot/...` path here unless you have verified it is mounted inside DLC workers.

Image safety gate: the verified Genesis/PTA training image is `pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang`. `launch_job.sh` now defaults to this image, but keep `DLC_IMAGE` exported and verify every submitted training job with `dlc get job <JOB_ID>` before counting it as a valid submission. The expected field is:

```text
JobSpecs[0].Image = pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang
```

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

Submit the legacy random-evaluation-encoder stress diagnostic after G0 passes and after `checkpoints/m7_pta_seed42/best/best_model.zip` has been restored and verified, if you need to reproduce the old sensitivity finding. This is not claim-bearing matched M7 evidence:

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

Run the corrected matched-encoder G2 audit directly from DSW or a local shell after the matched policy-plus-encoder bundle exists and the registry verifies it. This direct command does not submit a DLC job and does not return a JobId:

```bash
/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python tools/pre_submission_audit.py \
  --mode matched-encoder-audit \
  --method m7_pta \
  --seed 42 \
  --split ood_elastoplastic \
  --n-episodes 10
```

Expected corrected G2 output file:

```text
results/presub/audit_matched_encoder_m7_pta_s42_ood_elastoplastic.json
```

Monitor only the DLC-submitted probe-integrity jobs and legacy random-stress diagnostic job ids:

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

for path in sorted(Path("results/presub").glob("audit_matched_encoder_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    print(path.name, "passes", data["passes"], "total_failed_episodes", data["total_failed_episodes"], "reasons", data["reasons"])
PY
```

Pass criteria:

- G1 probe integrity: `audit_probe_ood_elastoplastic_seed123.json` has `probe_rms_m >= 0.0001` or `probe_max_m >= 0.001`.
- Legacy random-stress diagnostic: `audit_encoder_m7_pta_s42_ood_elastoplastic.json` records `random_eval_encoder_stress` behavior only. It is explicitly non-claim-bearing and does not satisfy corrected G2.
- Corrected G2 matched-encoder audit: `audit_matched_encoder_m7_pta_s42_ood_elastoplastic.json` has `passes == true` and `total_failed_episodes == 0`.

If G1 fails or corrected G2 has not passed, do not submit extra seed training jobs for claim strengthening.

Corrected M7 G2 must also verify the matched artifact bundle before any claim-bearing evaluation, but registry verification does not substitute for the matched evaluation audit:

```bash
python tools/artifact_registry.py verify \
  --repo-root "$PTA_CODE_ROOT" \
  --requirement g2-matched-encoder \
  --manifest results/presub/g2_matched_encoder_manifest.json
```

For full `m7_pta` matched eval, the policy zip alone is insufficient. The claim artifact must include the paired policy metadata, `belief_encoder.pt`, and `belief_encoder_metadata.json` sidecars that identify the matched encoder protocol.

## Isolated Matched Seed42 Rollout

Do not overwrite the main runtime tree's legacy `checkpoints/m7_pta_seed42` directory while producing corrected matched G2 evidence. Use a separate CPFS runtime copy for the full seed42 matched run:

```bash
set -euo pipefail
SOURCE_ROOT=/cpfs/shared/simulation/zhuzihou/dev/probe-then-act
ISOLATED_ROOT=/cpfs/shared/simulation/zhuzihou/dev/probe-then-act-matched-seed42
test -d "$SOURCE_ROOT"
test "$ISOLATED_ROOT" = /cpfs/shared/simulation/zhuzihou/dev/probe-then-act-matched-seed42
test "$SOURCE_ROOT" != "$ISOLATED_ROOT"
mkdir -p "$ISOLATED_ROOT"
rsync -ani --delete \
  --exclude '.git/' \
  --exclude '.worktrees/' \
  "$SOURCE_ROOT/" \
  "$ISOLATED_ROOT/" > /tmp/pta_matched_seed42_rsync_dry_run.txt
test -f /tmp/pta_matched_seed42_rsync_dry_run.txt
```

Review `/tmp/pta_matched_seed42_rsync_dry_run.txt` before running the destructive sync. Then run:

```bash
set -euo pipefail
SOURCE_ROOT=/cpfs/shared/simulation/zhuzihou/dev/probe-then-act
ISOLATED_ROOT=/cpfs/shared/simulation/zhuzihou/dev/probe-then-act-matched-seed42
test "$ISOLATED_ROOT" = /cpfs/shared/simulation/zhuzihou/dev/probe-then-act-matched-seed42
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.worktrees/' \
  "$SOURCE_ROOT/" \
  "$ISOLATED_ROOT/"
```

Verify the isolated runtime before submitting the full run:

```bash
cd /cpfs/shared/simulation/zhuzihou/dev/probe-then-act-matched-seed42
/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m py_compile \
  pta/training/utils/checkpoint_io.py \
  pta/scripts/train_m7.py \
  pta/scripts/run_ood_eval_v2.py \
  tools/pre_submission_audit.py \
  tools/artifact_registry.py
/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest \
  tests/test_m7_encoder_artifacts.py \
  tests/test_train_m7_encoder_sidecars.py \
  tests/test_run_ood_eval_v2.py \
  tests/test_pre_submission_audit.py \
  tests/test_artifact_registry.py \
  tests/test_dlc_shell_contract.py \
  tests/test_dlc_submit_jobs.py
```

Submit the full matched seed42 training job from the isolated runtime:

```bash
set -euo pipefail
mkdir -p results/dlc
DLC_BIN=/cpfs/shared/simulation/zhuzihou/dev/usd-scene-physics-prep/dlc
ISOLATED_ROOT=/cpfs/shared/simulation/zhuzihou/dev/probe-then-act-matched-seed42
GENESIS_ROOT=/cpfs/shared/simulation/zhuzihou/dev/Genesis
PYTHON_BIN=$GENESIS_ROOT/.venv/bin/python
DLC_DATA_SOURCES=d-mzps5b7joy2axmqpa8,d-d49o5g0h2818sw8j1g,d-8wz4emfs21s5ajs9oz
DLC_IMAGE=pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang
WORKER_COMMAND="PTA_CODE_ROOT=$ISOLATED_ROOT GENESIS_ROOT=$GENESIS_ROOT GENESIS_VENV=$GENESIS_ROOT/.venv PYTHON_BIN=$PYTHON_BIN PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 bash $ISOLATED_ROOT/pta/scripts/dlc/run_task.sh custom $PYTHON_BIN -u pta/scripts/train_m7.py --seed 42 --total-timesteps 500000 --eval-freq 10000 --horizon 500 --residual-scale 0.05"
"$DLC_BIN" submit pytorchjob \
  --name=pta_m7_matched_seed42_full_0_1 \
  --workers=1 \
  --job_max_running_time_minutes=0 \
  --worker_gpu=1 \
  --worker_cpu=14 \
  --worker_memory=100Gi \
  --worker_shared_memory=100Gi \
  --worker_image="$DLC_IMAGE" \
  --workspace_id=270969 \
  --resource_id=quota1r947pmazvk \
  --data_sources="$DLC_DATA_SOURCES" \
  --oversold_type=ForbiddenQuotaOverSold \
  --priority 7 \
  --command="$WORKER_COMMAND" \
  2>&1 | tee results/dlc/matched_seed42_submit.log

export MATCHED_SEED42_JOB_ID=$(
  /cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python - <<'PY'
import re
from pathlib import Path

text = Path('results/dlc/matched_seed42_submit.log').read_text(encoding='utf-8')
matches = re.findall(r'\bdlc[0-9a-z]{10,}\b', text)
if not matches:
    raise SystemExit('DLC submit log did not contain a JobId')
print(matches[-1])
PY
)
test -n "$MATCHED_SEED42_JOB_ID"
printf 'MATCHED_SEED42_JOB_ID=%s\n' "$MATCHED_SEED42_JOB_ID"
```

After the job succeeds, verify the matched bundle in the isolated runtime before running corrected G2:

```bash
/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python - <<'PY'
from pathlib import Path
from stable_baselines3 import PPO
from pta.training.utils.checkpoint_io import load_m7_encoder_artifact

policy = Path('checkpoints/m7_pta_seed42/best/best_model.zip')
model = PPO.load(str(policy), device='cpu')
encoder, metadata = load_m7_encoder_artifact(
    policy,
    expected={'method': 'm7_pta', 'seed': 42, 'ablation': 'none'},
)
print('model_timesteps', model.num_timesteps)
print('encoder_trace_dim', encoder.trace_dim)
print('protocol', metadata['protocol'])
print('paired_policy_path', metadata['paired_policy_path'])
PY

/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python tools/artifact_registry.py verify \
  --repo-root /cpfs/shared/simulation/zhuzihou/dev/probe-then-act-matched-seed42 \
  --requirement g2-matched-encoder \
  --manifest results/presub/g2_matched_encoder_manifest.json
```

Only after these checks pass should the corrected matched G2 audit be run from the isolated runtime with `--mode matched-encoder-audit`.

## 2026-05-01 Gate Status

Completed checks and current DLC status:

- G0 smoke job `dlcy7gu8a1ghjg77`: succeeded.
- G1 probe-integrity jobs for all five splits: succeeded. The elastoplastic gate file `results/presub/audit_probe_ood_elastoplastic_seed123.json` passed the displacement threshold.
- Legacy `random_eval_encoder_stress` diagnostic job `dlc1skykqxol8zl0`: DLC job succeeded, worker record `results/dlc/runs/20260501T141214Z_custom_dlc1skykqxol8zl0-master-0.json` has `exit_code=0`, and the diagnostic failed its sensitivity threshold. This is not a corrected matched-encoder G2 result.
- Corrected matched-encoder G2: **passed.** `dlcqfs83uu5rmvp7` succeeded on 2026-05-03 with worker `exit_code=0`. Audit output: `results/presub/audit_matched_encoder_m7_pta_s42_ood_elastoplastic.json` (in isolated seed42 root). Result: `passes=true`, `mean_transfer=0.96797`, `total_failed_episodes=0`, `protocol=matched_encoder_v1`. Replaces the old `random_eval_encoder_stress` diagnostic.
- 2026-05-02 runtime smoke `dlc156toui8myk40`: succeeded on the verified image with worker `exit_code=0`.
- 2026-05-02 matched-sidecar smoke `dlc7m883ucvcce7n`: succeeded on the verified image with worker `exit_code=0`; seed `9042` produced a load-verified `matched_encoder_v1` final bundle.
- 2026-05-02 isolated matched seed42 full run `dlc1weyuiyngs6ow`: succeeded on 2026-05-03, worker `exit_code=0`. Best and final matched bundles verified.

Historical queue/running status snapshot:

```text
/cpfs/shared/simulation/zhuzihou/dev/probe-then-act-matched-seed42/results/dlc/status_snapshot_20260502T0919Z.json
```

At `2026-05-02T09:22:49Z`, the 12 verified-image G3 replacement jobs were still `Running` with no worker records yet. This snapshot is retained for audit history; the 2026-05-03 refresh supersedes it and shows those 12 training jobs succeeded, while `dlc1weyuiyngs6ow` is running.

Legacy diagnostic result file:

```text
results/presub/audit_encoder_m7_pta_s42_ood_elastoplastic.json
```

Observed legacy diagnostic result:

```text
passes=false
transfer_range_pp=65.99578301705962
total_failed_episodes=0
reasons=["encoder sensitivity transfer range exceeded threshold"]
```

Initial gate decision: do not treat the old random-stress diagnostic as matched M7 evidence. The DLC job itself was healthy, but the diagnostic only showed that pairing the policy with fresh random evaluation encoders is unstable. The later six-seed G3 training submission was an explicit operator override for evidence collection only; G4 extra-seed evaluation still waits for worker-record and checkpoint verification and corrected matched G2 completion.

2026-05-03 update: the verified-image G3 replacement training jobs reached `Status=Succeeded` and worker records report `exit_code=0`. Their legacy-policy-only eval jobs have now been submitted with non-default output paths. These jobs are still not matched-encoder M7 headline evidence, because the old full `m7_pta` G3 training code did not persist encoder sidecars.

2026-05-03 matched M7 update: all 6 matched-encoder `m7_pta` full training jobs (seeds `0-4, 42`) succeeded on DLC with worker `exit_code=0`. Each seed produced a complete `matched_encoder_v1` bundle (best and final checkpoints with belief encoder sidecars, PPO-load and `load_m7_encoder_artifact` verified). Corrected G2 `dlcqfs83uu5rmvp7` passed. Matched G4 `dlcrjqlmycuna0or` produced all 30 M7 rows, then stalled on the M1 eval transition; M1 comparison rows are available from verified legacy eval outputs. Seed4 elastoplastic uses the accepted final-checkpoint value `0.9712`, superseding the best-checkpoint outlier. DLC priority 8 blocked by workspace role check; use priority ≤7.

Recovery training job `dlc1hn82yye94ojd` succeeded on 2026-05-01. Worker record `results/dlc/runs/20260501T113021Z_custom_dlc1hn82yye94ojd-master-0.json` has `exit_code=0`.

Final recovered artifacts are registered under:

```text
/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/20260501/20260501_dlc1hn82yye94ojd_m7_pta_seed42_final_recovery/
```

This final recovery archive contains 12 M7 seed42 checkpoint zip files, all SB3-load verified. The final checkpoint `checkpoints/m7_pta_seed42/m7_pta_final.zip` has `num_timesteps=500224` and `sha256=55bf288ab6211f15b016a6210b51435c5650d71a5ff0a4fc65e04c5835085116`.

The legacy random-stress result remains a non-claim-bearing diagnostic, not an infrastructure failure and not a corrected matched method failure. The non-overwriting legacy G4 eval jobs succeeded with verified row counts and supply the six reactive-baseline comparison rows. The corrected matched M7 path is supported by matched artifacts, corrected G2, and the six-seed elastoplastic audit; keep that audit separate from legacy policy-only diagnostics.

## Historical Extra Seed Training Commands

These commands document the original gated intent. The verified-image G3 replacement jobs were already submitted as an explicit operator override for evidence collection only, before corrected matched G2 passed. Do not submit more extra-seed training for claim strengthening until corrected matched G2 passes and current G3 worker records/checkpoints verify.

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

## 2026-05-03 Non-Overwriting Extra-Seed Eval Submissions

The current submission splits legacy evaluation into four DLC jobs so each output file is unique and resumable. These jobs close the train-to-eval loop for the 12 verified-image G3 training jobs. They must not be interpreted as corrected matched-encoder M7 evidence.

| Scope | DLC JobId | Status | Worker record | Output | Verified rows |
|---|---|---|---|---|---:|
| `m1_reactive`, `m7_pta`; seeds `2, 3` | `dlcoucjyiozupi5h` | `Succeeded` | `results/dlc/runs/20260503T030353Z_custom_dlcoucjyiozupi5h-master-0.json` | `results/presub/ood_eval_extra_seeds_s2_s3_legacy_policy_only_20260503.csv` | 20 |
| `m1_reactive`, `m7_pta`; seed `4` | `dlcpyaxhmcnzk87e` | `Succeeded` | `results/dlc/runs/20260503T031326Z_custom_dlcpyaxhmcnzk87e-master-0.json` | `results/presub/ood_eval_extra_seed4_legacy_policy_only_20260503.csv` | 10 |
| `m7_noprobe`; seeds `2, 3, 4` | `dlc5pczyite2yvpp` | `Succeeded` | `results/dlc/runs/20260503T031326Z_custom_dlc5pczyite2yvpp-master-0.json` | `results/presub/ood_eval_ablation_no_probe_extra_seeds_s2_s3_s4_legacy_policy_only_20260503.csv` | 15 |
| `m7_nobelief`; seeds `2, 3, 4` | `dlc7na5myukdtyc1` | `Succeeded` | `results/dlc/runs/20260503T030410Z_custom_dlc7na5myukdtyc1-master-0.json` | `results/presub/ood_eval_ablation_no_belief_extra_seeds_s2_s3_s4_legacy_policy_only_20260503.csv` | 15 |

Completion verification at `2026-05-03T06:16:35Z` checked that all four jobs use the verified `mahaoxiang` image, have worker `exit_code=0`, produce the expected row counts, cover every expected method/seed/split combination exactly once, do not pass `--no-resume`, do not target `results/ood_eval_per_seed.csv`, do not target `results/main_results.csv`, and do not reuse `results/presub/ood_eval_extra_seeds.csv`.

The row-count verification command is:

```bash
"$PYTHON_BIN" - <<'PY'
import csv
from pathlib import Path

expected = {
    "results/presub/ood_eval_extra_seeds_s2_s3_legacy_policy_only_20260503.csv": 20,
    "results/presub/ood_eval_extra_seed4_legacy_policy_only_20260503.csv": 10,
    "results/presub/ood_eval_ablation_no_probe_extra_seeds_s2_s3_s4_legacy_policy_only_20260503.csv": 15,
    "results/presub/ood_eval_ablation_no_belief_extra_seeds_s2_s3_s4_legacy_policy_only_20260503.csv": 15,
}
for path, count in expected.items():
    with Path(path).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(path, len(rows))
    assert len(rows) == count
PY
```

Summarize the legacy expanded-seed main-method diagnostic only after the main-method CSV files exist:

```bash
"$PYTHON_BIN" -u tools/pre_submission_audit.py \
  --mode summarize-five-seed \
  --inputs \
    results/ood_eval_per_seed.csv \
    results/presub/ood_eval_extra_seeds_s2_s3_legacy_policy_only_20260503.csv \
    results/presub/ood_eval_extra_seed4_legacy_policy_only_20260503.csv
```

Do not include the ablation CSV files in `summarize-five-seed`; they answer the separate question of whether `no_probe` and `no_belief` degrade relative to full M7. The original G4 gate expected `n_pairs=5`; with seed `4` included, the current expanded diagnostic reports `n_pairs=6`.

Observed expanded main-method legacy summary:

```text
n_pairs=6
mean_delta_pp=19.192
std_delta_pp=78.548
positive_pairs=4/6
```

Original paper-strengthening gate G4 passed for the legacy policy-only diagnostic if:

- `n_pairs=5`
- `mean_delta_pp > 0`
- `positive_pairs >= 3/5`

Corrected matched-encoder elastoplastic audit status: the claim-bearing M7 matched comparison now uses 6 seeds (`0, 1, 2, 3, 4, 42`) with seed4 final checkpoint transfer `0.9712`. Summary: `n_pairs=6`, `mean_delta_pp=+54.2`, `positive_pairs=4/6`, with seeds 1 and 3 near ties. Keep this audit separate from the original 3-seed cross-split main table.

## Final Paper Step

G1 and corrected matched G2 have passed, and the matched-encoder six-seed elastoplastic audit is ready for paper wording. Regardless of extra-seed outcome, add the NeurIPS checklist before final upload and rebuild:

```bash
cd "$PTA_CODE_ROOT/paper"
make nips2026
make all
```
