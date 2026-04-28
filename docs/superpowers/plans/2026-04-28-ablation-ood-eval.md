# Ablation OOD Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a corrected OOD table for `m7_noprobe` and `m7_nobelief` using all required seeds `42`, `0`, and `1`.

**Architecture:** Treat the five completed DLC resume jobs as validated training artifacts, then reconcile the missing local `m7_noprobe seed=42` checkpoint into the DSW checkout. Before launching OOD, make the checkpoint policy explicit because the current evaluator uses `best/best_model` while the DLC resume jobs also produced `m7_pta_final`.

**Tech Stack:** Python 3.10, Stable-Baselines3 PPO, Genesis, PAI-DLC PyTorchJob, pytest, CSV result files.

---

### Task 1: Record Training Completion

**Files:**
- Modify: `docs/30_records/DLC_M7_ABLATION_RESUME_2026-04-27.md`
- Modify: `refine-logs/EXPERIMENT_TRACKER.md`

- [ ] **Step 1: Confirm DLC status**

Run:

```bash
DLC_BIN=/cpfs/shared/simulation/zhuzihou/dev/usd-scene-physics-prep/dlc
HTTPS_PROXY= HTTP_PROXY= https_proxy= http_proxy= \
NO_PROXY=127.0.0.1,localhost,::1,pai-dlc.cn-beijing.aliyuncs.com \
  "$DLC_BIN" get job \
  --job_ids "dlc14uard6mq7vsw,dlc15e9y4qc12v0j,dlc15o9jiielquzg,dlc15y94wa65t7c8,dlc16i8bnuxurvgb" \
  --workspace_id 270969 \
  --endpoint=pai-dlc.cn-beijing.aliyuncs.com \
  --region=cn-beijing
```

Expected: all five rows show `JobStatus` as `Succeeded`.

- [ ] **Step 2: Confirm worker records**

Run:

```bash
/shared/smartbot/zhuzihou/envs/pta-genesis-py310-cu128/bin/python - <<'PY'
import json
from pathlib import Path

for path in sorted(Path("results/dlc/runs").glob("pta_resume_*_400k_500k_*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    print(path.name, data["exit_code"], data["command"])
PY
```

Expected: five records and each `exit_code` is `0`.

- [ ] **Step 3: Mark R002-R006 done**

Update `refine-logs/EXPERIMENT_TRACKER.md`:

```markdown
| R002 | M1 | Train no-probe ablation | `m7_noprobe` seed 0 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc14uard6mq7vsw`; final checkpoint `checkpoints/m7_pta_noprobe_seed0/m7_pta_final.zip`; `num_timesteps=500352`. |
| R003 | M1 | Train no-probe ablation | `m7_noprobe` seed 1 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc15e9y4qc12v0j`; final checkpoint `checkpoints/m7_pta_noprobe_seed1/m7_pta_final.zip`; `num_timesteps=500352`. |
| R004 | M1 | Train no-belief ablation | `m7_nobelief` seed 42 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc15o9jiielquzg`; final checkpoint `checkpoints/m7_pta_nobelief_seed42/m7_pta_final.zip`; `num_timesteps=500352`. |
| R005 | M1 | Train no-belief ablation | `m7_nobelief` seed 0 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc15y94wa65t7c8`; final checkpoint `checkpoints/m7_pta_nobelief_seed0/m7_pta_final.zip`; `num_timesteps=500352`. |
| R006 | M1 | Train no-belief ablation | `m7_nobelief` seed 1 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc16i8bnuxurvgb`; final checkpoint `checkpoints/m7_pta_nobelief_seed1/m7_pta_final.zip`; `num_timesteps=500352`. |
```

### Task 2: Reconcile `m7_noprobe seed=42`

**Files:**
- Runtime artifact directory: `checkpoints/m7_pta_noprobe_seed42/`
- Modify: `docs/30_records/CHECKPOINT_MANIFEST.md`

- [ ] **Step 1: Check whether the seed-42 checkpoint is present**

Run:

```bash
find checkpoints/m7_pta_noprobe_seed42 -maxdepth 2 -type f | sort
```

Expected before reconciliation on this DSW: directory is missing.

- [ ] **Step 2: Copy the local R001 artifact into this checkout**

Copy the complete directory from the machine that ran R001 into:

```text
/shared/smartbot/zhuzihou/dev/probe-then-act/checkpoints/m7_pta_noprobe_seed42/
```

Required files:

```text
checkpoints/m7_pta_noprobe_seed42/best/best_model.zip
checkpoints/m7_pta_noprobe_seed42/m7_pta_final.zip
```

- [ ] **Step 3: Verify the copied checkpoint**

Run:

```bash
/shared/smartbot/zhuzihou/envs/pta-genesis-py310-cu128/bin/python - <<'PY'
from stable_baselines3 import PPO

for path in [
    "checkpoints/m7_pta_noprobe_seed42/best/best_model.zip",
    "checkpoints/m7_pta_noprobe_seed42/m7_pta_final.zip",
]:
    model = PPO.load(path, device="cpu")
    print(path, model.num_timesteps)
PY
```

Expected: both files load; `m7_pta_final.zip` reports about `500000` timesteps.

### Task 3: Make OOD Checkpoint Policy Explicit

**Files:**
- Modify: `pta/scripts/run_ood_eval_v2.py`
- Modify: `tests/test_run_ood_eval_v2.py`
- Modify: `docs/30_records/DLC_EXECUTION_RUNBOOK.md`

- [ ] **Step 1: Add a failing resolver test**

Add this test to `tests/test_run_ood_eval_v2.py`:

```python
def test_resolve_checkpoint_prefers_final_when_configured(tmp_path):
    from pta.scripts.run_ood_eval_v2 import resolve_checkpoint_path

    run_dir = tmp_path / "checkpoints" / "m7_pta_noprobe_seed0"
    (run_dir / "best").mkdir(parents=True)
    (run_dir / "best" / "best_model.zip").write_text("best", encoding="utf-8")
    (run_dir / "m7_pta_final.zip").write_text("final", encoding="utf-8")

    resolved = resolve_checkpoint_path(
        tmp_path,
        "checkpoints/m7_pta_noprobe_seed{seed}/m7_pta_final",
        seed=0,
    )

    assert resolved == run_dir / "m7_pta_final.zip"
```

- [ ] **Step 2: Run the failing test**

Run:

```bash
/shared/smartbot/zhuzihou/envs/pta-genesis-py310-cu128/bin/python -m pytest \
  tests/test_run_ood_eval_v2.py::test_resolve_checkpoint_prefers_final_when_configured -q
```

Expected before implementation: fails because `resolve_checkpoint_path` does not exist.

- [ ] **Step 3: Implement checkpoint resolver**

Add to `pta/scripts/run_ood_eval_v2.py`:

```python
def resolve_checkpoint_path(project_root: Path, ckpt_pattern: str, seed: int) -> Path | None:
    ckpt_path = project_root / ckpt_pattern.format(seed=seed)
    if ckpt_path.exists():
        return ckpt_path
    zip_path = ckpt_path.with_suffix(".zip")
    if zip_path.exists():
        return zip_path
    return None
```

Replace the inline checkpoint existence block in `main()` with:

```python
ckpt_path = resolve_checkpoint_path(_PROJECT_ROOT, method_cfg["ckpt_pattern"], seed)
if ckpt_path is None:
    print(
        f"  SKIP: {method_name} seed={seed} -- checkpoint not found: "
        f"{_PROJECT_ROOT / method_cfg['ckpt_pattern'].format(seed=seed)}"
    )
    continue

print(f"\n>>> {method_name} seed={seed}")
model = PPO.load(str(ckpt_path))
```

- [ ] **Step 4: Choose the policy for ablation OOD**

Recommended for this completed-DLC state: evaluate final 500k artifacts for ablations, because the resume jobs disabled eval callbacks and `best/best_model.zip` was not refreshed during the final 100k.

Set:

```python
"m7_noprobe": {
    "seeds": [42, 0, 1],
    "ckpt_pattern": "checkpoints/m7_pta_noprobe_seed{seed}/m7_pta_final",
    "use_privileged": False,
    "use_m7_env": True,
    "ablation": "no_probe",
},
"m7_nobelief": {
    "seeds": [42, 0, 1],
    "ckpt_pattern": "checkpoints/m7_pta_nobelief_seed{seed}/m7_pta_final",
    "use_privileged": False,
    "use_m7_env": True,
    "ablation": "no_belief",
},
```

Document this in `docs/30_records/DLC_EXECUTION_RUNBOOK.md` so the result table records that ablation OOD used final checkpoints while older baseline OOD rows used their existing configured best checkpoints.

- [ ] **Step 5: Run targeted tests**

Run:

```bash
/shared/smartbot/zhuzihou/envs/pta-genesis-py310-cu128/bin/python -m pytest \
  tests/test_run_ood_eval_v2.py tests/test_dlc_submit_jobs.py -q
```

Expected: all tests pass.

### Task 4: Submit Ablation OOD DLC

**Files:**
- Runtime outputs: `results/ood_eval_per_seed.csv`, `results/main_results.csv`, `results/dlc/runs/*.json`

- [ ] **Step 1: Confirm current OOD rows**

Run:

```bash
/shared/smartbot/zhuzihou/envs/pta-genesis-py310-cu128/bin/python - <<'PY'
import csv
from collections import Counter

for path in ["results/ood_eval_per_seed.csv", "results/main_results.csv"]:
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    print(path, len(rows), dict(Counter(row["method"] for row in rows)))
PY
```

Expected before ablation OOD: `m7_noprobe` and `m7_nobelief` are absent.

- [ ] **Step 2: Dry-run submit command**

Run:

```bash
PTA_CODE_ROOT=/shared/smartbot/zhuzihou/dev/probe-then-act \
GENESIS_ROOT=/shared/smartbot/zhuzihou/dev/Genesis \
GENESIS_VENV=/shared/smartbot/zhuzihou/envs/pta-genesis-py310-cu128 \
PYTHON_BIN=/shared/smartbot/zhuzihou/envs/pta-genesis-py310-cu128/bin/python \
DLC_RESULTS_ROOT=/shared/smartbot/zhuzihou/dev/probe-then-act/results/dlc \
DLC_BIN=/cpfs/shared/simulation/zhuzihou/dev/usd-scene-physics-prep/dlc \
DLC_IMAGE=pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang \
PYOPENGL_PLATFORM=egl \
EGL_DEVICE_ID=0 \
DLC_DRY_RUN=1 \
/shared/smartbot/zhuzihou/envs/pta-genesis-py310-cu128/bin/python -m pta.scripts.dlc.submit_jobs \
  --suite ood-ablation \
  --name pta_ood_ablation_final_ckpts \
  --gpu-count 1
```

Expected: printed DLC command uses the verified image and `eval_ood --residual-scale 0.05 --methods m7_noprobe m7_nobelief`.

- [ ] **Step 3: Submit the real OOD job**

Run the same command without `DLC_DRY_RUN=1`.

- [ ] **Step 4: Monitor OOD completion**

Run:

```bash
/cpfs/shared/simulation/zhuzihou/dev/usd-scene-physics-prep/dlc get job <OOD_JOB_ID> \
  --endpoint=pai-dlc.cn-beijing.aliyuncs.com \
  --region=cn-beijing
```

Expected: `Succeeded`. If interrupted, rerun the same OOD job; `run_ood_eval_v2.py` resumes existing rows.

### Task 5: Analyze And Record Result-To-Claim

**Files:**
- Modify: `findings.md`
- Modify: `refine-logs/EXPERIMENT_TRACKER.md`
- Modify: `docs/10_protocols/04_VALIDATION_GATES.md` if the decision gate changes

- [ ] **Step 1: Verify ablation OOD row counts**

Run:

```bash
/shared/smartbot/zhuzihou/envs/pta-genesis-py310-cu128/bin/python - <<'PY'
import csv
from collections import Counter

with open("results/ood_eval_per_seed.csv", newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
counts = Counter(row["method"] for row in rows)
print(counts)
assert counts["m7_noprobe"] == 15
assert counts["m7_nobelief"] == 15
PY
```

- [ ] **Step 2: Compare against M1 and full M7**

Use `results/main_results.csv` to answer:

```text
Does no-probe repair the full M7 regression?
Does no-belief repair the full M7 regression?
Which component appears to hurt ID/OOD transfer and spill?
Is any broad PTA robustness claim supported?
```

- [ ] **Step 3: Update the written verdict**

Update `findings.md` with the ablation result. Keep the claim language conservative until the table supports it.
