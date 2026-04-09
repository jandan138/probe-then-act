# Current Blockers and Immediate Actions

> Internal repo document for `probe-then-act`.
>
> Purpose: compress the current diagnosis into one execution-facing note.
> This document answers four questions:
> 1. What is the current project state?
> 2. Which problems have been resolved?
> 3. What is the current blocker?
> 4. What should the team do next, in order?
>
> **Last updated: 2026-04-09 (bowl follow-up note only; core blocker snapshot still reflects the 2026-04-07 sprint)**

---

## 1. Executive Summary

The project is **no longer blocked by the control stack**. The 48-hour diagnostic sprint (2026-04-07) confirmed and resolved the IK/controller issues:

> The **joint-space residual control stack works** — the learner reaches scripted baseline performance in 20K steps. The new blocker is **base trajectory quality**.

More concretely:
- The Cartesian-delta → IK → PD controller path was confirmed broken (0% transfer, 0.68m z-divergence).
- The IK y-inversion is a single-step DLS coupling artifact, not a Genesis bug (iterative IK works correctly).
- `JointResidualWrapper` bypasses IK entirely and enables learning.
- But the scripted edge-push base trajectory only achieves ~12.5% transfer — below the 30% Gate 4 threshold.

Therefore, the right interpretation is:

> We have a working learning interface, but the base strategy is not good enough for the residual to improve past the success threshold.

---

## 2. Current State Snapshot

### What has been resolved (48-hour sprint)

| Issue | Resolution | Evidence |
|---|---|---|
| IK y-axis inversion | Not a Genesis bug — single-step DLS coupling artifact | `docs/IK_MINIMAL_REPRO.md`: iterative IK 8/8 sign matches |
| PD controller failure | `control_dofs_position()` can't track low-z configs | `results/controller_replay_ab_test.csv`: 0% transfer, z-div 0.68m |
| Learner stuck at random | Joint-space residual reaches baseline in 20K steps | `docs/GATE4_TRAINING_REPORT.md`: -2.09 vs. -39.6 (E1) |

### What is still blocked

| Issue | Status | Detail |
|---|---|---|
| Gate 4 (30% transfer) | **NOT PASSED** | Best: 12.5% transfer (v1) / 12-15% (v2). Threshold: 30%. |
| Base trajectory quality | **CURRENT BLOCKER** | Scripted edge-push achieves ~12.5% — residual can't improve beyond it |
| Gate 5 (full-scale) | BLOCKED | Requires Gate 4 |

### Gate status

| Gate | Status |
|---|---|
| 0 — Physical Feasibility | **PASSED** |
| 1 — Task/Theory Spec | PARTIAL |
| 2 — Implementation Correctness | **PASSED** (IK/controller bypassed) |
| 3 — System Smoke Test | **PASSED** |
| 4 — Tiny-Task Overfit | **PARTIAL** (12.5% vs. 30% target) |
| 5 — Full-Scale Experiment | BLOCKED |

---

## 3. Resolved Problems (Sprint Findings)

### 3.1 Resolved: IK y-axis inversion

**What we found:** Genesis's single-step DLS IK has a coupling artifact — the orientation-preservation constraint competes with the position delta, producing 3-35% y-gain and sign flips near zero. This is NOT a simulator-wide bug.

**Evidence:**
- Iterative DLS (50 iters): 8/8 sign matches, near-zero error
- Genesis built-in `inverse_kinematics()`: 8/8 correct
- Jacobian J[y, joint1] = +0.308 — correct positive sign

**Corrected claim:** “The single-step DLS IK in `_compute_ik()` has a coupling artifact for y-direction” (not “Genesis globally inverts y-axis”).

### 3.2 Resolved: PD controller can't track scripted trajectory

**What we found:** `control_dofs_position()` with current PD gains (kp=4500/3500/2000) cannot converge to the low-z, extended-arm configurations needed for edge-push. Mean z-divergence 0.348m, max 0.679m.

**Evidence:** `results/controller_replay_ab_test.csv` — Mode A (set_qpos): 10.65% transfer; Mode B (control_dofs_position): 0% transfer.

### 3.3 Resolved: Learner control path

**Solution:** `JointResidualWrapper` (`pta/envs/wrappers/joint_residual_wrapper.py`) bypasses IK and PD controller entirely. Uses `robot.set_qpos(q_base[t] + scale * delta_q)` directly.

**Evidence:** E3 joint-space residual v1 reaches -2.09 reward in 20K steps (E1 Cartesian-delta was stuck at -39.6).

---

## 4. Current Blocker: Base Trajectory Quality

### The problem

The scripted edge-push trajectory (3-pass flat push) achieves ~12.5% transfer. The residual policy reproduces this faithfully but cannot discover corrections that push past 30%.

### Why the residual can't improve enough

- `residual_scale=0.1` (v1): Policy converges to near-zero residuals. Exploration range too small to discover fundamentally different motions.
- `residual_scale=0.2` (v2): More volatile, best -1.20 vs v1's -2.09, but still ~12-15% transfer.
- The edge-push strategy has a physics ceiling: pushing particles off a platform edge with a flat sweeping motion is inherently limited.

### Practical reading

> The learning infrastructure works. The bottleneck is the base strategy, not the learning algorithm.

---

## 5. Decision: What Not To Do Right Now

Until Gate 4 is passed:
- Do NOT revert to Cartesian-delta actions (broken, confirmed).
- Do NOT launch large multi-material sweeps or OOD campaigns.
- Do NOT claim the method works based on reward improvements alone (12.5% ≠ 30%).
- Do NOT spend GPU on broad hyperparameter searches — focus on strategy.

---

## 6. Immediate Action Plan (Ordered)

### Priority A — Try a better base trajectory

**A1. Keep scoop / bowl out of the Gate 4 main line:**
- Do **not** reopen the scoop / bowl trajectory as the current base strategy for Gate 4.
- The flat-scene bowl diagnosis is now a documented side-track negative result.
- If bowl is continued at all, follow `docs/11_BOWL_TOOL_INVESTIGATION.md` and `docs/12_BOWL_TRANSPORT_DIAGNOSIS_RUNBOOK.md`: native contact-quality tuning first, sticky fallback second, no interference with edge-push work.

**A2. Optimize the edge-push trajectory**:
- Adjust approach angle, push speed, number of passes
- Particles near the platform edge may need a different entry angle
- Try deeper push (lower z) or wider sweep pattern

### Priority B — Widen residual exploration

**B1. Scale curriculum:**
- Start at `residual_scale=0.0` for 50K steps (pure replay warmup)
- Anneal to `residual_scale=0.3` over 100K steps
- Continue at 0.3 for remaining training

**B2. Longer training:**
- Current runs only reached 45K steps
- Run 500K-1M steps with scale=0.2-0.3

### Priority C — Task geometry investigation

**C1. Check if container placement caps transfer:**
- Source at (0.5, 0.0), target at (0.5, 0.35)
- Edge-push relies on particles falling off +y edge
- Are particles landing in target AABB or missing?

**C2. Adjust platform/container layout if needed:**
- Lower platform edge
- Move target closer to source
- Widen target AABB

### Priority D — BC warmstart (fallback)

If residual RL with better trajectory still plateaus:
- Use `checkpoints/demos/scripted_joint_demos.npz` (20 episodes already collected)
- BC pretrain → PPO fine-tune

---

## 7. Go / No-Go Decision Rule

### Go (proceed to Gate 5 work)
- At least one configuration achieves `success_rate >= 70%` and `transferred_mass_frac >= 0.25`
- Performance stable across 3 eval reruns
- Training curve clearly separates from scripted baseline (not just reproduces it)

### No-Go
- All trajectory variants plateau below 30% transfer
- In that case: revisit task design (is edge-push the right task for this paper?)

---

## 8. Canonical One-Sentence Diagnosis

> The joint-space residual control stack works — the learner reaches scripted baseline performance in 20K steps. Gate 4 is blocked by base trajectory quality (~12.5% transfer vs. 30% target), not by the learning infrastructure. Next work should focus on improving the base strategy or widening the residual exploration range.

---

## 9. Key Artifacts from Sprint

| Artifact | Path |
|---|---|
| Controller A/B test script | `pta/scripts/controller_replay_ab.py` |
| Controller A/B results | `results/controller_replay_ab_test.csv` |
| IK minimal repro script | `pta/scripts/ik_minimal_repro.py` |
| IK repro report | `docs/IK_MINIMAL_REPRO.md` |
| Joint-space residual wrapper | `pta/envs/wrappers/joint_residual_wrapper.py` |
| Scripted demos (20 episodes) | `checkpoints/demos/scripted_joint_demos.npz` |
| Gate 4 training report | `docs/GATE4_TRAINING_REPORT.md` |
| Full sprint results | `docs/08_48HR_SPRINT_RESULTS.md` |
