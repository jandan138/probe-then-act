# Remote Handoff Bundle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a fresh DSW/DLC machine able to clone the research repos, install the pinned local Genesis runtime, retrieve minimal artifacts, and run smoke/DLC jobs without rediscovering local machine assumptions.

**Architecture:** Keep executable remote-handoff logic in `probe-then-act`, because this repo owns the experiment commands. Genesis is published as a separate GitHub fork/remote and referenced by lock documentation plus `.env` examples, while checkpoints remain release/storage artifacts instead of normal Git files.

**Tech Stack:** Bash for bootstrap and smoke scripts, Python stdlib for checkpoint manifest generation, pytest for contract tests, GitHub SSH remotes for pushes.

---

### Task 1: Remote Handoff Contracts

**Files:**
- Create: `tests/test_remote_handoff_assets.py`

- [x] **Step 1: Write failing tests for required handoff assets**

Add tests that assert the bootstrap/smoke/download scripts exist, are executable, pass shell syntax checks, and that the checkpoint manifest builder exposes the minimal corrected-OOD and ablation-resume artifact set.

- [x] **Step 2: Run the focused test and verify RED**

Run: `pytest tests/test_remote_handoff_assets.py -q`
Expected: FAIL because `scripts/build_checkpoint_manifest.py`, `.env.dsw.example`, `.env.local.example`, and the remote docs/scripts do not exist yet.

### Task 2: Checkpoint Manifest Builder

**Files:**
- Create: `scripts/build_checkpoint_manifest.py`
- Modify: `tests/test_remote_handoff_assets.py`

- [ ] **Step 1: Implement minimal manifest generation**

Create a Python script that scans known checkpoint candidates from the main repo and optional Stage-D worktree, emits deterministic JSON with `exists`, `size_bytes`, and `sha256` for present files, and can optionally create a tar archive from present candidates.

- [ ] **Step 2: Run focused tests**

Run: `pytest tests/test_remote_handoff_assets.py -q`
Expected: PASS for the Python manifest contract while shell/doc tests still fail until Task 3.

### Task 3: Remote Bootstrap Scripts And Environment Examples

**Files:**
- Create: `.env.dsw.example`
- Create: `.env.local.example`
- Create: `scripts/bootstrap_remote.sh`
- Create: `scripts/smoke_remote.sh`
- Create: `scripts/download_artifacts.sh`
- Create: `pta/scripts/dlc/preflight_remote.sh`

- [ ] **Step 1: Add scripts with strict shell settings**

Scripts should use `set -euo pipefail`, resolve repo roots without assuming the current working directory, avoid credentials, and support dry-run or missing-artifact cases with clear errors.

- [ ] **Step 2: Run shell syntax tests**

Run: `pytest tests/test_remote_handoff_assets.py -q`
Expected: PASS for executable and `bash -n` checks.

### Task 4: Remote Reproduction Documentation

**Files:**
- Create: `docs/30_records/REMOTE_REPRODUCTION_RUNBOOK.md`
- Create: `docs/30_records/GENESIS_RUNTIME_LOCK.md`
- Create: `docs/30_records/CHECKPOINT_MANIFEST.md`

- [ ] **Step 1: Document clone/install/run sequence**

Describe clone order, expected remotes, Genesis commit/branch status, environment variables, checkpoint retrieval, local smoke, DLC smoke, ablation training, and corrected OOD eval commands.

- [ ] **Step 2: Run docs/asset tests**

Run: `pytest tests/test_remote_handoff_assets.py -q`
Expected: PASS and docs include the GitHub Genesis remote plus artifact policy.

### Task 5: Commit, Push, And Genesis Publication

**Files:**
- Modify: Genesis working tree as already present locally

- [ ] **Step 1: Verify probe repo**

Run: `pytest tests/test_remote_handoff_assets.py tests/test_dlc_shell_contract.py tests/test_dlc_submit_jobs.py -q`
Expected: PASS.

- [ ] **Step 2: Commit and push probe handoff bundle**

Commit the new handoff assets to `probe-then-act/main` and push `origin/main`.

- [ ] **Step 3: Commit and push Genesis runtime**

Preserve the original Genesis upstream as a remote, add `git@github.com:jandan138/Genesis.git`, commit local WSLg/headless/bowl runtime changes, and push `main` to the personal Genesis remote.

- [ ] **Step 4: Final status check**

Run `git status --short --branch` in probe, Auto, usd-scene-physics-prep, and Genesis. Report remaining non-Git artifact requirements explicitly.
