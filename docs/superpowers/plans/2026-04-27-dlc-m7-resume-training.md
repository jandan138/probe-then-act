# DLC M7 Resume Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resume the five interrupted M7 ablation DLC trainings from their durable 400k SB3 checkpoints and finish the remaining 100k timesteps without TensorBoard writes.

**Architecture:** Add a small M7 resume entrypoint that reuses `train_m7.make_m7_env`, loads an SB3 PPO checkpoint, computes remaining timesteps from the checkpoint's `num_timesteps`, and saves final models under the existing checkpoint directories. Submit DLC jobs through the already verified image/spec with `PYOPENGL_PLATFORM=egl`.

**Tech Stack:** Python 3.10, Stable-Baselines3 PPO, Genesis, DLC PyTorchJob, pytest.

---

### Task 1: Document Failure And Resume State

**Files:**
- Create: `docs/30_records/DLC_M7_ABLATION_RESUME_2026-04-27.md`
- Modify: `docs/30_records/CHECKPOINT_MANIFEST.md`

- [x] **Step 1: Record failed job IDs and root cause**

Write a record with the failed DLC job IDs, `OSError: [Errno 28] No space left on device`, and the last logged timestep per run.

- [x] **Step 2: Record durable checkpoints**

Record that all five runs have a verified `m7_pta_400000_steps.zip` and that each loads with `PPO.load(...).num_timesteps == 400000`.

### Task 2: Add Resume Script

**Files:**
- Create: `tests/test_resume_m7.py`
- Create: `pta/scripts/resume_m7.py`

- [x] **Step 1: Write failing tests for checkpoint selection**

Add tests for selecting the highest `m7_pta_<step>_steps.zip`, computing remaining timesteps, and resolving M7 run names.

- [x] **Step 2: Run tests to confirm RED**

Run: `/shared/smartbot/zhuzihou/envs/pta-genesis-py310-cu128/bin/python -m pytest tests/test_resume_m7.py -q`

- [x] **Step 3: Implement resume script**

Implement `latest_step_checkpoint`, `remaining_timesteps`, CLI parsing, `PPO.load(..., tensorboard_log=None)`, `model.learn(total_timesteps=remaining, reset_num_timesteps=False)`, and final checkpoint save.

- [x] **Step 4: Run tests to confirm GREEN**

Run: `/shared/smartbot/zhuzihou/envs/pta-genesis-py310-cu128/bin/python -m pytest tests/test_resume_m7.py -q`

### Task 3: Submit DLC Resume Jobs

**Files:**
- No source file edits.

- [x] **Step 1: Verify current shared write capability**

Verified `/shared/smartbot` had 79T available and 2.6G free inodes before resubmission.

- [x] **Step 2: Submit five custom DLC jobs**

Use the verified image `pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang`, resource `quota1r947pmazvk`, and env `PYOPENGL_PLATFORM=egl,EGL_DEVICE_ID=0`.

- [ ] **Step 3: Verify submitted jobs**

Run `dlc get job` for all five new job IDs and inspect early logs for `Resuming M7 PPO` plus no immediate traceback.

Status query completed after submission: all five jobs were `Queuing` with `ReasonCode=JobEnqueued`. Early worker logs were not available yet while jobs were still queued.
