# 48-Hour Sprint Results: Controller Diagnostic & Joint-Space Residual

**Date:** 2026-04-07
**Sprint Lead:** sprint-lead (ARIS agent team)

## Executive Summary

The sprint **confirmed the root cause** of Gate 4 failure and **delivered a working joint-space learning interface**. The RL learner now reaches scripted baseline performance (vs. stuck at random with the old IK path). Gate 4 targets are **not yet met** — the bottleneck has shifted from "learner can't control the robot" to "base trajectory needs improvement."

**One-sentence update:** The control stack is fixed; the next bottleneck is the task strategy.

---

## Diagnostic Results (Day 1)

### Task 1: Controller Replay A/B Test

| Metric | Mode A (set_qpos) | Mode B (control_dofs_position) |
|---|---|---|
| Transfer efficiency | 10.65% | **0.00%** |
| Final EE y | 0.410 m | 0.228 m |
| EE z divergence | — | mean 0.348 m, max 0.679 m |

**Verdict:** `control_dofs_position()` completely fails. PD controller cannot hold the arm at low-z extended configurations. The learner's control path (Cartesian-delta → IK → PD control) is broken at the PD level, independent of IK.

### Task 2: Minimal IK Repro

**Verdict:** The y-axis inversion is **not a Genesis bug**. Iterative DLS (50 iters) and Genesis built-in `inverse_kinematics()` both produce correct results (8/8 sign matches). The single-step DLS in `_compute_ik()` has a coupling artifact: orientation-preservation constraint competes with position delta, yielding 3-35% gain and sign flips near zero.

### Go/No-Go: **GO** → Joint-space residual

Two independent failures compound in the old path:
1. Single-step DLS IK: near-zero gain, sign inversions
2. PD controller: can't track resulting joint targets (z floats 0.68m above table)

---

## Implementation Results (Day 2)

### Joint-Space Residual Wrapper

**File:** `pta/envs/wrappers/joint_residual_wrapper.py`

Design: `q_applied = q_base[t] + residual_scale * delta_q`
- Bypasses IK entirely — calls `robot.set_qpos()` directly
- Two pre-built trajectories: `"scoop"` (215 steps), `"edge_push"` (410 steps)
- 7D action space (joint residuals), 30D observation space
- Smoke-tested: zero residual reproduces scripted baseline

### Gate 4 Training

| Method | Mean Reward | Transfer % | vs. Random |
|--------|-----------|------------|------------|
| E1 Cartesian-delta PPO (old) | -39.6 ± 3 | ~0% | **No improvement** |
| v1 Joint-residual (scale=0.1) | -2.09 | ~12.5% | **20x better** |
| v2 Joint-residual (scale=0.2) | -1.20 (best) | ~12-15% | **33x better** |
| Gate 4 target | — | ≥30% | — |

**Gate 4 verdict: NOT PASSED** — but the learner clearly works now. It reaches scripted baseline performance and the value function learns well (explained variance 0.74).

---

## Root Cause Shift

| Before Sprint | After Sprint |
|---|---|
| "Learner can't learn" — stuck at random | Learner reaches scripted baseline in 20K steps |
| Blocked by: IK + PD controller | Blocked by: base trajectory quality |
| Fix: control stack | Fix: task strategy |

The bottleneck is no longer the learning infrastructure. The scripted edge-push trajectory only achieves ~12.5% transfer — still below the formal Gate 4 pass criteria. The residual policy reproduces this faithfully but can't push past it with ±0.1-0.2 rad corrections.

---

## Recommended Next Steps (Priority Order)

### 1. Better Base Trajectory
The 3-pass edge-push achieves 12.5%. Options:
- **Scoop trajectory** (`"scoop"`, already available in wrapper): lift-traverse-deposit may achieve higher transfer
- **Optimized edge-push**: adjust approach angle, push speed, number of passes
- **Wider residual scale** (0.3-0.5 rad) with curriculum annealing

### 2. Longer Training + Curriculum
- Start at scale=0.0 (pure replay) → anneal to scale=0.3 over 100K steps
- Run 1M+ steps total
- May allow residual to discover larger corrections that beat the base

### 3. PD Controller Gain Tuning (Optional)
If switching to `control_dofs_position()` for physical realism:
- Current gains too low for extended low-z configurations
- Need gravity compensation or higher kp for joints 2, 4, 6

### 4. Task Geometry
- Container placement and particle initialization affect achievable transfer
- Current layout may have a ceiling below 30% for any push strategy

---

## Deliverables Checklist

| # | Deliverable | Status |
|---|---|---|
| 1 | `pta/scripts/controller_replay_ab.py` | ✅ |
| 2 | `results/controller_replay_ab_test.csv` + plot | ✅ |
| 3 | `pta/scripts/ik_minimal_repro.py` | ✅ |
| 4 | `docs/40_investigations/IK_MINIMAL_REPRO.md` | ✅ |
| 5 | `pta/envs/wrappers/joint_residual_wrapper.py` | ✅ |
| 6 | `checkpoints/demos/scripted_joint_demos.npz` | ✅ (20 episodes, 1.45 MB) |
| 7 | Gate 4 learning curves | ✅ (v1 + v2) |
| 8 | This document | ✅ |

---

## Canonical Diagnosis Update

> **The joint-space residual control stack works — the learner reaches scripted baseline performance in 20K steps (vs. zero learning with Cartesian-delta IK). Gate 4 is blocked by base trajectory quality (~12.5% transfer, still below formal pass thresholds), not by the learning infrastructure. Next work should focus on improving the base strategy or widening the residual exploration range.**
