# Validation Gates for `probe-then-act`

> Internal execution doctrine for the current project.
> 
> Principle: **No full-scale experiment before passing the required gates.**
> 
> This document adapts the "formal proof → gradient check → system smoke test → tiny-set overfit" methodology into a robot-learning version suitable for multi-physics manipulation.

---

## 0. Why this document exists

The project is currently in a regime where multiple layers are entangled:
- physical feasibility of the task,
- task specification correctness,
- implementation correctness,
- training system correctness,
- learning correctness.

When all of these are mixed together, a failed 500K–2M step training run does **not** tell us which layer is broken.

This document defines a **gated workflow** so that failures are isolated early and cheaply.

---

## 1. Core rule

**Do not launch new full-scale benchmark runs, OOD campaigns, or method-comparison sweeps unless the project has passed the gates below.**

In the current project state, the highest priority is:
1. pass **Gate 0: Physical Feasibility**;
2. pass **Gate 4: Tiny-Task Overfit**.

Until then, large-scale PPO tuning is considered **premature**.

---

## 2. The gate stack

## Gate 0 — Physical Feasibility

### Objective
Prove that the task is physically executable in the current simulator and geometry configuration.

### Why it exists
If the end-effector geometry, task timing, or material setup make scooping physically implausible, RL cannot fix it.

### Required evidence
- A **custom scoop attachment** or other scoop-like end-effector integrated into the robot model.
- A **scripted controller** that performs `approach → insert → drag → lift → move → dump`.
- Quantitative metrics from scripted trials:
  - `success_rate`
  - `lifted_mass_frac`
  - `transferred_mass_frac`
  - `spill_frac`
- At least one saved rollout video or GIF.
- No NaNs, no reset corruption, no geometry penetration instability.

### Pass criteria
Gate 0 is considered passed only if all are true:
- scripted controller succeeds on the tiny task in at least **80%** of fixed evaluation episodes;
- median `transferred_mass_frac >= 0.30`;
- median `spill_frac <= 0.20`;
- results are repeatable across at least **2 reruns**.

### Typical failure meaning
- 0% scripted success → geometry / timing / task design is wrong.
- lifted mass > 0 but transfer mass ≈ 0 → dumping phase or container placement is wrong.
- frequent NaNs / explosions → simulator settings or contacts are unstable.

### Current status
**NOT PASSED**.

---

## Gate 1 — Task / Theory Specification

### Objective
Write the task as a precise contract instead of an informal intuition.

### Why it exists
Many RL failures come from ambiguous definitions of success, reward, phase transitions, or privileged information.

### Required artifacts
Create a task contract document that explicitly defines:
- **Task objective**
- **Action space**
- **Observation space**
  - observable signals
  - privileged signals
- **Episode timing**
  - physics rate
  - policy rate
  - effective episode duration in seconds
- **Success condition**
- **Failure condition**
- **Reward terms** and intended phase behavior
- **Allowed simplifications** for the tiny-task regime
- **Metrics** that matter more than mean return

### Pass criteria
Gate 1 is passed only if:
- every training script uses the same task contract;
- every reward term has a short explanation and expected range;
- there is no ambiguity around what counts as success;
- privileged observation usage is explicitly documented.

### Typical failure meaning
- high return but 0% success → reward contract does not reflect task completion.
- privileged Teacher underperforms unexpectedly → privileged observation contract may be wrong.

### Current status
**PARTIALLY PASSED**.

---

## Gate 2 — Implementation Correctness

### Objective
Verify that the code implements the intended task contract.

### Required checks
- Reward decomposition logging is correct and numerically stable.
- `success_rate`, `lifted_mass_frac`, `transferred_mass_frac`, and `spill_frac` are computed correctly.
- Action timing is correct:
  - `ctrl_dt`
  - `substeps`
  - `action_repeat`
  - actual policy frequency
  - actual episode duration in seconds
- Scripted controller executes the intended waypoints.
- Privileged observation wrapper matches its spec.
- Reset function truly resets particles, robot state, and hidden parameters.
- Checkpoint save / load / resume works.

### Pass criteria
Gate 2 is passed only if:
- all metrics are unit-tested or sanity-checked against scripted rollouts;
- one short run can be resumed from checkpoint and continue cleanly;
- reward logs match expected phase progression during scripted trajectories.

### Typical failure meaning
- metrics inconsistent with video → logging bug.
- reward improves but no phase progression → reward computation bug or poor proxy.
- resume diverges immediately → checkpoint state incomplete.

### Current status
**PARTIALLY PASSED**.

---

## Gate 3 — System Smoke Test

### Objective
Confirm that the full training system works as an engineering system.

### Required checks
- train loop runs without crash;
- eval loop runs without crash;
- metrics are written;
- checkpoints are saved;
- resume works;
- deterministic eval runs are reproducible enough for diagnosis.

### Pass criteria
Gate 3 is passed if a short smoke run completes with:
- valid logs,
- valid checkpoints,
- valid evaluation output,
- no infrastructure-level crash.

### Current status
**PASSED**.

---

## Gate 4 — Tiny-Task Overfit

### Objective
Demonstrate that the learning pipeline can solve a deliberately simplified version of the task.

### Why it exists
If the model cannot learn a tiny, fixed, easy task, it is not ready for OOD generalization or benchmark claims.

### Tiny-task assumptions
The tiny task should be drastically simplified:
- **sand-only**;
- fixed material parameters;
- fixed fill level;
- fixed source bin and target container geometry;
- fixed or near-fixed initial robot pose;
- custom scoop tool;
- simplified action space;
- longer effective episode duration (seconds, not just steps).

### Minimum acceptable learning setup
At least one of the following must succeed:
- Teacher with privileged observations;
- scripted controller + residual RL;
- BC / DAPG-style demo-guided learner.

### Pass criteria
Gate 4 is passed only if, on a fixed tiny-task evaluation set:
- `success_rate >= 70%` for the chosen learner;
- median `transferred_mass_frac >= 0.25`;
- performance is stable across at least **3 evaluation reruns**;
- training curve clearly separates from random / reactive baseline.

### Typical failure meaning
- scripted succeeds but learner fails → training setup is broken.
- learner reaches approach only → action space / reward / exploration remains wrong.
- Teacher cannot overfit → task is still too hard or implementation still wrong.

### Current status
**NOT PASSED**.

---

## Gate 5 — Full-Scale Experiment

### Objective
Only after the lower layers are validated, run the actual research program:
- larger train distributions,
- OOD materials,
- probing variants,
- ablations,
- student distillation,
- final benchmark tables.

### Entry condition
Gate 5 is allowed **only if Gates 0–4 are passed**.

### Forbidden before Gate 5
Do **not** spend substantial GPU budget on:
- large OOD sweeps,
- multi-method comparison campaigns,
- broad hyperparameter searches,
- paper-level claim generation.

### Current status
**NOT ALLOWED YET**.

---

## 3. Current project assessment (as of current diagnosis)

| Gate | Name | Status | Interpretation |
|---|---|---|---|
| 0 | Physical Feasibility | NOT PASSED | Current gripper-only setup has not been validated by scripted success. |
| 1 | Task / Theory Specification | PARTIAL | Reward/phase logic exists, but the task contract is not yet rigid enough. |
| 2 | Implementation Correctness | PARTIAL | Infra is good, but metric and wrapper correctness still need tighter verification. |
| 3 | System Smoke Test | PASSED | Environment, training infra, eval infra, and checkpoints are operational. |
| 4 | Tiny-Task Overfit | NOT PASSED | No method has demonstrated true scoop-transfer learning yet. |
| 5 | Full-Scale Experiment | BLOCKED | Not allowed until the lower gates pass. |

---

## 4. What is allowed right now

### Allowed
- build and integrate scoop attachment;
- write scripted controller;
- run scripted feasibility experiments;
- simplify the environment into a tiny-task regime;
- reduce action space and add action repeat;
- fix PPO exploration settings;
- run tiny-task Teacher / residual / demo-guided experiments.

### Not allowed
- new 500K–2M multi-material benchmark sweeps;
- new OOD claim generation;
- broad baseline expansion;
- narrative writing that assumes the method works.

---

## 5. Required evidence before promotion

## To promote from Gate 0 to Gate 1–4 work
Save all of the following:
- `scripted_eval.csv`
- rollout videos/GIFs
- tool geometry files
- the exact environment config used
- a short note explaining why the scripted policy works

## To promote from Gate 4 to Gate 5
Save all of the following:
- training curves
- tiny-task evaluation table
- seed-wise results
- best checkpoint
- failure case gallery
- one short note: **what changed to make learning finally work**

---

## 6. Immediate next actions

### Priority 1
Pass **Gate 0**.
- Add scoop attachment.
- Run scripted feasibility.
- Measure lifted/transferred mass.

### Priority 2
Pass **Gate 4**.
- Freeze a tiny task.
- Reduce action space.
- Use longer effective episode duration.
- Train Teacher or residual learner until it clearly overfits.

### Priority 3
Only after the above:
- reopen student learning,
- reopen OOD evaluation,
- reopen method comparison.

---

## 7. One-sentence doctrine

> **First prove the task can be done, then prove the learner can overfit the easy version, and only then scale to the real benchmark.**
