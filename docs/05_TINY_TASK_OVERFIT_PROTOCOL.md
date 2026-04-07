# Tiny-Task Overfit Protocol for `probe-then-act`

> Internal protocol.
> 
> Goal: create the smallest believable version of the scooping problem and force at least one learner to solve it reliably.

---

## 0. Why this protocol exists

The current project has a working simulator and training pipeline, but no method has achieved real scoop-transfer success.

Therefore, the next milestone is **not** "better benchmark performance".
The next milestone is:

> **Show that the learning stack can overfit a tiny, fixed, easy scooping task.**

If this protocol fails, the project should not move to OOD evaluation or full benchmark claims.

---

## 1. Success definition of this protocol

The protocol is successful only if a chosen learner achieves all of the following on the tiny-task evaluation set:
- `success_rate >= 70%`
- median `transferred_mass_frac >= 0.25`
- median `spill_frac <= 0.25`
- stable performance across **3 repeated evaluations**
- clear separation from random / reactive baseline

Anything below this is considered **not ready for scale-up**.

---

## 2. Tiny-task design

## 2.1 Environment scope
The tiny task must be dramatically simpler than the final benchmark.

### Fixed choices
- **Material family:** `sand` only
- **Material parameters:** fixed
- **Fill level:** fixed
- **Source bin geometry:** fixed
- **Target container geometry:** fixed
- **Robot start pose:** fixed or near-fixed
- **Tool geometry:** fixed custom scoop attachment
- **Lighting / observation noise:** off or minimal
- **Reset distribution:** tiny and controlled

### Explicit exclusions
Do **not** include any of the following in the tiny-task phase:
- liquid
- multiple material families
- large domain randomization
- OOD splits
- probing-policy benchmarking
- student generalization claims

---

## 2.2 Task phases
The task is decomposed into five phases:
1. **Approach** — move the scoop toward the source region.
2. **Insert** — penetrate the material with a stable scoop orientation.
3. **Drag** — move through the material to gather particles.
4. **Lift** — raise the scoop without excessive spill.
5. **Dump / Transfer** — move above the target and release contents.

The protocol must log **which phase was reached** in every rollout.

---

## 2.3 Action space
Use the simplest action space that still supports success.

### Current preferred option (updated 2026-04-07)
**7-D joint-space residual** via `JointResidualWrapper`:
- `q_applied = q_base[t] + residual_scale * delta_q`
- `q_base[t]` is a precomputed scripted trajectory (edge-push or scoop)
- Policy outputs 7D residual in [-1, 1]
- Bypasses IK entirely — uses `robot.set_qpos()` directly
- `residual_scale` default 0.1 rad (try 0.2-0.5 for broader exploration)
- Observation: 30D (22D base + 7D q_base[t] + 1D step_fraction)

### Deprecated options
**3-D Cartesian delta** (Option A):
- DEPRECATED — Genesis single-step DLS IK has coupling artifact (3-35% y-gain, sign flips)
- `control_dofs_position()` PD controller cannot track low-z extended configurations (0.68m z-divergence)
- Do NOT use for learning

**Waypoint parameterization** (Option B):
- Still valid as an alternative but not yet tested

### Forbidden in the first tiny-task iteration
- Cartesian-delta actions through IK (broken)
- unconstrained roll/pitch/yaw exploration
- joint-torque exploration unless a scripted controller already works

---

## 2.4 Timing
The policy must make **meaningful** decisions, not ultra-high-frequency noise.

### Recommended timing
- physics rate: keep simulator stable
- policy rate: **20–25 Hz**
- use `action_repeat` so the policy does not act every physics step
- effective episode duration: **3–6 seconds**

### Example
If physics is 500 Hz and `action_repeat=25`, then policy runs at 20 Hz.
At 4 seconds total duration, one episode has 80 policy decisions.

This is preferred over a 0.4-second task with hundreds of tiny actions.

---

## 3. Required precondition: scripted feasibility

No learning experiment may start before a scripted controller is implemented.

## 3.1 Scripted controller goals
The scripted controller should execute:
- pre-approach
- insertion
- drag through material
- lift
- move to target
- dump

## 3.2 Scripted pass condition
The scripted controller must achieve:
- `success_rate >= 80%`
- median `transferred_mass_frac >= 0.30`
- median `spill_frac <= 0.20`

If scripted control fails, the problem is still in the **feasibility / geometry** stage, not the learning stage.

---

## 4. Learner ladder

The learner ladder enforces an easy-to-hard progression.

## Level 0 — Scripted only
This is a non-learning baseline.
Its purpose is to prove the task is executable.

## Level 1 — Teacher overfit
Use the easiest possible learner:
- privileged observations allowed
- fixed tiny-task environment
- simplified action space
- strong reward shaping allowed

### Preferred choices
- Teacher PPO with corrected exploration settings
- residual RL on top of scripted controller

### Goal
Overfit the tiny task as quickly as possible.

## Level 2 — Demo-guided learner (optional)
If Level 1 remains unstable:
- BC warm start
- DAPG-style auxiliary BC loss
- short demonstration set from scripted rollouts

## Level 3 — Student on the same tiny task
Only after Level 1 passes:
- remove privileged observations
- keep the same tiny task
- verify that the observable-only student can also solve it

Only after Levels 0–3 are stable should the project widen the task distribution.

---

## 5. Training recipe for the first overfit attempt

## 5.1 Environment freeze
Create a dedicated config, for example:
- `configs/overfit/sand_tiny_task.yaml`

This config must freeze:
- material parameters
- fill level
- source / target positions
- robot reset pose
- scoop geometry
- action space
- timing
- reward scales

No hidden changes between runs are allowed.

## 5.2 Suggested PPO cleanup
For PPO-based runs, start conservative:
- `ent_coef = 0.0`
- `use_sde = True` if supported
- lower initial action std
- deterministic evaluation
- action repeat enabled

The goal is **controlled optimization**, not aggressive exploration.

## 5.3 Preferred first learner
Use one of the following as the first real learning attempt:

### Option A — Residual RL
`a_total = a_scripted + a_residual`

Why:
- the base motion already knows how to scoop;
- RL only learns corrections;
- exploration becomes local instead of blind.

### Option B — Teacher PPO with privileged obs
Why:
- easiest version of the learning problem;
- fastest way to test whether the task is learnable at all.

If neither works on the tiny task, fix the task before adding more methods.

---

## 6. Metrics to log

Do **not** rely on mean return alone.

Every run must log at least:
- `success_rate`
- `phase_reached`
- `lifted_mass_frac`
- `transferred_mass_frac`
- `spill_frac`
- `max_penetration_depth`
- `contact_duration`
- `distance_to_source`
- `distance_to_target`
- episode video for selected seeds

Optional but recommended:
- reward decomposition by phase
- action norm statistics
- policy std statistics
- seed-wise tables

---

## 7. Evaluation protocol

## 7.1 Fixed evaluation set
Create a small fixed evaluation set, e.g.:
- 16 episodes
- deterministic resets from saved seeds or saved initial states

Use this exact set for all tiny-task comparisons.

## 7.2 Repetition rule
Evaluate the same checkpoint **3 times** on the same fixed set.

Purpose:
- catch simulator nondeterminism;
- catch logging bugs;
- avoid one-off lucky success.

## 7.3 Promotion rule
A learner is promoted only if all 3 repeated evaluations satisfy the pass thresholds.

---

## 8. Experiment matrix

## E0 — Scripted feasibility
- objective: pass Gate 0
- output: `scripted_eval.csv`, videos, summary table
- **Status: PASSED** (42.2% transfer, 9.0% spill, 5/5 repeatable)

## E1 — Teacher overfit (Cartesian-delta)
- privileged observations allowed
- fixed tiny task
- objective: prove learnability
- **Status: FAILED** — Cartesian-delta → IK → PD controller path broken (IK coupling artifact + PD z-divergence). Reward oscillated at random baseline (-39.6).

## E2 — Residual overfit (Cartesian-delta)
- scripted base + learned correction (Cartesian space)
- fixed tiny task
- objective: improve stability and reduce exploration burden
- **Status: FAILED** — Same IK/PD issues as E1. Deprecated.

## E3 — Joint-space residual overfit (NEW, 2026-04-07)
- `JointResidualWrapper`: `q_base[t] + scale * delta_q`, bypasses IK
- fixed tiny task, edge-push trajectory
- **Status: PARTIAL**
  - v1 (scale=0.1): Converged to scripted baseline (-2.09 reward, ~12.5% transfer). Stable but no improvement beyond base.
  - v2 (scale=0.2): Best reward -1.20, more exploration but oscillating. 12-15% transfer.
  - Both dramatically better than E1/E2 (random-level → baseline-level in 20K steps)
  - Gate 4 targets NOT MET: 12.5% transfer vs. 30% required
- **Next**: try scoop trajectory, scale=0.3-0.5, curriculum, 1M+ steps

## E4 — Student tiny-task
- no privileged observations
- same tiny task
- objective: verify observability is sufficient
- **Status: NOT STARTED** — blocked by E3 not passing Gate 4

### Important
Do **not** start E3 before E1 or E2 has passed.

---

## 9. What counts as failure, and what it means

### Failure pattern A
**Scripted controller fails.**

Interpretation:
- geometry, timing, or task layout is still wrong.
- do not train RL.

### Failure pattern B
**Scripted succeeds, Teacher fails.**

Interpretation:
- training setup, action space, reward, or optimization is still wrong.

### Failure pattern C
**Teacher succeeds, Student fails.**

Interpretation:
- observability is insufficient or the student architecture is weak.

### Failure pattern D
**Return improves, but transfer stays near zero.**

Interpretation:
- reward proxy is misleading.
- reward must be revised to match task completion.

---

## 10. Required artifacts

Each overfit cycle must produce:
- config file
- training command
- commit hash
- best checkpoint
- training curve
- evaluation CSV
- at least 3 rollout videos
- short error analysis note

All artifacts must be stored under a dedicated overfit directory, e.g.:

```text
results/overfit/
  E0_scripted/
  E1_teacher/
  E2_residual/
  E3_student/
```

---

## 11. Exit criteria for scaling up

Only after the tiny-task protocol passes may the project expand in this order:
1. slightly broader reset distribution;
2. small variation in fill level;
3. small variation in source / target pose;
4. mild material parameter randomization;
5. additional material families;
6. OOD evaluation.

Scale **one axis at a time**.
Do not reopen all difficulty factors at once.

---

## 12. One-sentence doctrine

> **If we cannot make one learner overfit one tiny scooping task, we are not ready to claim a robot-learning method yet.**
