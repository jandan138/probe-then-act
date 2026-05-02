# Six-Seed DLC Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Submit 12 DLC training jobs that extend `m1_reactive`, `m7_pta`, `m7_noprobe`, and `m7_nobelief` from policy seeds `0, 1, 42` to `0, 1, 2, 3, 4, 42`.

**Architecture:** Use the existing CPFS runtime tree and `launch_job.sh` without changing training code. Submit one single-GPU DLC job per method/seed pair and update record docs with submitted job IDs only after the CLI returns them.

**Tech Stack:** Bash DLC launcher, PTA training scripts, CPFS runtime tree, Markdown records.

---

### Task 1: Verify Intended Matrix

**Files:**
- Read: `docs/30_records/SEED_SENSITIVITY_STATUS.md`
- Read: `pta/scripts/dlc/launch_job.sh`
- Read: `pta/scripts/train_baselines.py`
- Read: `pta/scripts/train_m7.py`

- [ ] **Step 1: Confirm extra seeds**

Use extra seeds `2`, `3`, and `4` only. Do not resubmit existing seeds `0`, `1`, or `42`.

- [ ] **Step 2: Confirm method commands**

Use these commands inside DLC workers:

```bash
/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -u pta/scripts/train_baselines.py --method m1 --seed <seed> --total-timesteps 500000 --residual-scale 0.05 --horizon 500 --eval-freq 50000
/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -u pta/scripts/train_m7.py --seed <seed> --total-timesteps 500000 --residual-scale 0.05 --horizon 500 --latent-dim 16 --n-probes 3 --eval-freq 50000
/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -u pta/scripts/train_m7.py --ablation no_probe --seed <seed> --total-timesteps 500000 --residual-scale 0.05 --horizon 500 --latent-dim 16 --n-probes 3 --eval-freq 50000
/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -u pta/scripts/train_m7.py --ablation no_belief --seed <seed> --total-timesteps 500000 --residual-scale 0.05 --horizon 500 --latent-dim 16 --n-probes 3 --eval-freq 50000
```

### Task 2: Submit DLC Jobs

**Files:**
- Runtime command only: `/cpfs/shared/simulation/zhuzihou/dev/probe-then-act/pta/scripts/dlc/launch_job.sh`

- [ ] **Step 1: Submit the 12 single-GPU jobs**

Run from `/cpfs/shared/simulation/zhuzihou/dev/probe-then-act` with these environment settings:

```bash
PTA_CODE_ROOT=/cpfs/shared/simulation/zhuzihou/dev/probe-then-act
GENESIS_ROOT=/cpfs/shared/simulation/zhuzihou/dev/Genesis
GENESIS_VENV=/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv
PYTHON_BIN=/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python
DLC_BIN=/cpfs/shared/simulation/zhuzihou/dev/usd-scene-physics-prep/dlc
DLC_DATA_SOURCES=d-mzps5b7joy2axmqpa8,d-d49o5g0h2818sw8j1g,d-8wz4emfs21s5ajs9oz
DLC_GPU_COUNT=1
DLC_IMAGE=pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang
```

`launch_job.sh` defaults to the same verified image, but keep this explicit export in operational submissions so the image choice is visible at the shell boundary.

Submit one job for each pair:

```text
pta_6seed_m1_s2, pta_6seed_m1_s3, pta_6seed_m1_s4
pta_6seed_m7_s2, pta_6seed_m7_s3, pta_6seed_m7_s4
pta_6seed_noprobe_s2, pta_6seed_noprobe_s3, pta_6seed_noprobe_s4
pta_6seed_nobelief_s2, pta_6seed_nobelief_s3, pta_6seed_nobelief_s4
```

- [ ] **Step 2: Capture DLC job IDs**

Expected: the DLC CLI prints one submitted job identifier per job. Record all 12 identifiers.

- [ ] **Step 3: Verify the image for every submitted job**

Run `dlc get job <JOB_ID> --endpoint=pai-dlc.cn-beijing.aliyuncs.com --region=cn-beijing` for each returned job ID. Expected: every `JobSpecs[0].Image` equals `pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang`.

### Task 3: Update Record Docs

**Files:**
- Modify: `docs/30_records/SEED_SENSITIVITY_STATUS.md`

- [ ] **Step 1: Add a submitted-jobs section**

Add a section titled `Six-Seed DLC Submissions - 2026-05-02` with tables for any superseded wrong-image jobs and the verified-image replacements. For replacements, include method, seed, job name, DLC job ID, and status from the image check.

- [ ] **Step 2: Verify docs formatting**

Run:

```bash
git diff --check
```

Expected: no output.
