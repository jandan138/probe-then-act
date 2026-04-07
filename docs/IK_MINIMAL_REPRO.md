# IK Minimal Repro: Y-Axis Inversion Analysis

**Date:** 2026-04-07
**Script:** `pta/scripts/ik_minimal_repro.py`

## Background

During scripted baseline development, commanding `dy=-0.45` moved the Franka EE in the +y direction (opposite of commanded). This repro isolates whether the bug is:
1. A fundamental Genesis IK sign bug
2. A TCP / scoop-body frame mismatch
3. A single-step DLS coupling artifact

## Setup

- Robot: Franka Panda with scoop (`panda_scoop.xml`), 7 DOF
- Home qpos: `[0, -0.785, 0, -2.356, 0, 1.571, 0.785]`
- EE (scoop link) start: `(0.305, 0.000, 0.583)`
- Link7 start: `(0.306, 0.000, 0.690)`
- Sweep: dy in {-0.3, -0.2, -0.1, -0.05, +0.05, +0.1, +0.2, +0.3}

## Results

### Test 1: DLS Single-Step — Scoop Link (as used in task._compute_ik)

| dy_cmd | target_y | actual_y | dy_actual | sign_match | ratio |
|--------|----------|----------|-----------|------------|-------|
| -0.300 | -0.300   | -0.010   | -0.010    | YES        | 0.033 |
| -0.200 | -0.210   | -0.034   | -0.034    | YES        | 0.168 |
| -0.100 | -0.109   | -0.026   | -0.026    | YES        | 0.258 |
| -0.050 | -0.058   | -0.018   | -0.018    | YES        | 0.354 |
| +0.050 | +0.044   | -0.007   | -0.007    | **NO**     | -0.134|
| +0.100 | +0.106   | +0.021   | +0.021    | YES        | 0.212 |
| +0.200 | +0.208   | +0.032   | +0.032    | YES        | 0.161 |
| +0.300 | +0.309   | +0.034   | +0.034    | YES        | 0.115 |

**7/8 sign matches.** Sign inversion at dy=+0.05. Gain is extremely low (3-35% of commanded).

### Test 2: DLS Iterative (50 iters) — Scoop Link

| dy_cmd | target_y | actual_y | dy_actual | sign_match | error  |
|--------|----------|----------|-----------|------------|--------|
| -0.300 | -0.300   | -0.300   | -0.300    | YES        | 0.0000 |
| -0.200 | -0.200   | -0.200   | -0.200    | YES        | 0.0000 |
| -0.100 | -0.100   | -0.100   | -0.100    | YES        | 0.0000 |
| -0.050 | -0.050   | -0.050   | -0.050    | YES        | 0.0000 |
| +0.050 | +0.050   | +0.050   | +0.050    | YES        | 0.0000 |
| +0.100 | +0.100   | +0.100   | +0.100    | YES        | 0.0000 |
| +0.200 | +0.200   | +0.200   | +0.200    | YES        | 0.0000 |
| +0.300 | +0.300   | +0.300   | +0.300    | YES        | 0.0000 |

**8/8 sign matches.** Near-zero error. DLS IK is correct when iterated to convergence.

### Test 3: DLS Single-Step — Link7 (no scoop body)

| dy_cmd | target_y | actual_y | dy_actual | sign_match | ratio |
|--------|----------|----------|-----------|------------|-------|
| -0.300 | -0.290   | +0.018   | +0.009    | **NO**     | -0.029|
| -0.200 | -0.210   | -0.033   | -0.024    | YES        | 0.119 |
| -0.100 | -0.109   | -0.026   | -0.017    | YES        | 0.166 |
| -0.050 | -0.058   | -0.018   | -0.009    | YES        | 0.188 |
| +0.050 | +0.044   | -0.007   | -0.000    | **NO**     | -0.007|
| +0.100 | +0.106   | +0.021   | +0.015    | YES        | 0.148 |
| +0.200 | +0.208   | +0.033   | +0.024    | YES        | 0.120 |
| +0.300 | +0.309   | +0.035   | +0.025    | YES        | 0.085 |

**6/8 sign matches.** Inversions at dy=-0.3 and dy=+0.05. Same low-gain problem.

### Test 4: Genesis Built-in IK — Scoop Link

| dy_cmd | target_y | actual_y | dy_actual | sign_match | error  |
|--------|----------|----------|-----------|------------|--------|
| -0.300 | -0.300   | -0.287   | -0.286    | YES        | 0.0136 |
| -0.200 | -0.200   | -0.189   | -0.189    | YES        | 0.0108 |
| -0.100 | -0.100   | -0.091   | -0.091    | YES        | 0.0093 |
| -0.050 | -0.050   | -0.041   | -0.041    | YES        | 0.0094 |
| +0.050 | +0.050   | +0.060   | +0.060    | YES        | 0.0097 |
| +0.100 | +0.100   | +0.109   | +0.109    | YES        | 0.0089 |
| +0.200 | +0.200   | +0.205   | +0.205    | YES        | 0.0051 |
| +0.300 | +0.300   | +0.298   | +0.298    | YES        | 0.0019 |

**8/8 sign matches.** Small residual error (~1cm) but correct direction.

### Test 5: Jacobian Inspection at Home Pose

Jacobian y-row (scoop): `[0.308, 0.005, 0.400, 0.003, 0.107, 0.003, 0.0]`
Jacobian y-row (link7): `[0.308, 0.007, 0.475, 0.001, 0.000, 0.000, 0.0]`

J[y, joint1] = +0.308 for both links — correct sign (positive, meaning +dq1 → +dy).

## Root Cause Analysis

**The y-inversion is NOT a Genesis IK sign bug.** The Jacobian signs are correct, and both iterative DLS and Genesis built-in IK produce correct results.

The root cause is a **single-step DLS coupling artifact**:

1. **6-DOF delta with zero rotation**: The task's `_compute_ik()` builds a 6-DOF delta vector `[dx, dy, dz, 0, 0, 0]`. The DLS solver tries to satisfy all 6 constraints simultaneously. With zero rotation delta, the orientation-preservation constraint competes with the position delta, producing a near-zero compromise.

2. **Extremely low gain**: Single-step DLS achieves only 3-35% of the commanded dy. The damping term (lambda=0.01) combined with the 6-DOF coupling means most of the delta is absorbed by the orientation rows of the Jacobian.

3. **Sign inversions near zero**: At small deltas (dy=0.05), the numerical noise from the 6-DOF coupling can flip the sign of the tiny actual movement.

4. **Link-independent**: Both scoop and link7 show the same behavior, confirming this is NOT a scoop frame rotation issue.

**Why the scripted baseline saw a large y-inversion**: The scripted baseline sent `dy=-0.45` as a single-step action. The single-step DLS produced a near-zero or wrong-sign joint delta due to coupling. Over many steps, accumulated drift in the wrong direction looked like a persistent inversion.

## Conclusion

| Root Cause | Verdict |
|---|---|
| Genesis IK sign bug | **NO** — iterative DLS and built-in IK both correct |
| TCP / scoop frame mismatch | **NO** — link7 shows same behavior |
| Single-step DLS coupling | **YES** — 6-DOF delta with zero rotation produces near-zero, sometimes wrong-sign motions |

## Recommended Fix

The correct fix is to **bypass Cartesian IK entirely for the RL policy** and use **joint-space residual control**:
- Action space: 7-DOF joint position deltas
- No IK needed, no coupling artifacts
- Already planned as Task #3 (Joint-Space Residual Wrapper)

If Cartesian control is still needed for scripted baselines, use either:
- Genesis built-in IK (`robot.inverse_kinematics()`)
- Iterative DLS (50+ iterations per command)
- Position-only 3-DOF Jacobian (rows 0-2 only, ignoring orientation)
