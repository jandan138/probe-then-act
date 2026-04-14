# 02_EXECUTION_PLAYBOOK

## 1. Goal of this document
This is the **team operating manual** for executing the `probe-then-act` project.

This document answers:
- who does what,
- in what order,
- with which milestones,
- under what decision rules,
- and what counts as success or failure.

---

## 2. Project framing

### 2.1 Core thesis
The paper should argue:

> In robot tool use under hidden physical properties, **active probing + latent physical belief inference + uncertainty-aware control** improves robustness and OOD generalization over reactive or passive-history baselines.

### 2.2 What this project is not
This project is **not**:
- a “Genesis showcase”,
- a pure simulator engineering demo,
- a generic PPO benchmark,
- a sim-to-real paper,
- or a deformable manipulation application paper without methodological depth.

### 2.3 What this project is
This project **is**:
- a **method paper**,
- with a **reproducible benchmark**,
- for **robot learning under hidden physics**,
- evaluated on **cross-material tool use** in multi-physics simulation.

---

## 3. Team roles

If the team is small, one person can own multiple roles. But the responsibilities should still remain distinct.

### Role A — Environment Lead
Owns:
- Genesis scene construction
- material families
- tool geometry library
- sensor pipeline
- task reset logic
- metrics
- debug videos

Deliverables:
- stable `scoop-and-transfer`
- stable `level-and-fill`
- trustworthy metrics and overlays

### Role B — Learning Lead
Owns:
- reactive baseline
- recurrent baseline
- teacher policy
- student policy
- belief encoder
- uncertainty head
- main Probe-Then-Act method

Deliverables:
- all baselines trainable from CLI
- main method trainable from CLI
- config-driven ablations

### Role C — Evaluation Lead
Owns:
- train/test split definitions
- ID / OOD evaluation runners
- seed aggregation
- table generation
- figure generation
- failure gallery and videos

Deliverables:
- paper-ready tables and figures
- reproducible eval scripts
- error analysis summaries

### Role D — Literature / Writing Lead
Owns:
- literature matrix
- claim tracking
- related work notes
- paper outline
- figure captions
- rebuttal-style reviewer question list

Deliverables:
- weekly literature updates
- claim-to-evidence mapping
- draft intro / related work / method text

---

## 4. Reading order for new team members

On day 1, every team member must read in this order:
1. `docs/00_foundation/00_PROJECT_BRIEF.md`
2. this file
3. `docs/00_foundation/01_REPO_BLUEPRINT.md`
4. `docs/10_protocols/03_EXPERIMENT_PROTOCOL.md`

Then they should answer three questions in writing:
1. What is the paper claim?
2. What are the two main tasks?
3. What is the minimum publishable experiment package?

If they cannot answer those three, they are not ready to code.

---

## 5. Execution rules

### Rule 1 — Environment before learning
No large-scale RL until:
- reset is stable,
- metrics are correct,
- scripts can visualize behavior,
- one scripted / oracle policy exists.

### Rule 2 — Mainline before fancy ideas
The project should first succeed on:
- `MPM + Rigid`
- two tasks only
- one robot
- one observation stack

Only then consider:
- richer tactile models,
- more complex coupling,
- third task,
- differentiable losses,
- more exotic tools.

### Rule 3 — Every claim must have a future table
If someone proposes a method change, they must answer:
- which future table will show it,
- against which baseline,
- on which split,
- using which metric.

### Rule 4 — No untracked split changes
Train/test IDs are sacred.
Any split change must be versioned and justified.

### Rule 5 — Debug videos are mandatory
Every major milestone must include:
- at least one success video,
- at least one failure video,
- a short note on why the behavior looks correct or wrong.

---

## 6. Week-by-week execution plan (8 weeks)

## Week 1 — Environment bootstrap
### Objective
Build minimal tasks and verify simulation stability.

### Required outputs
- one Franka scene loads
- one scoop tool loads
- one source container + one target container load
- one MPM material runs stably
- one debug rollout video
- initial metrics defined

### Deliverable
`Week1_environment_bootstrap.md`

### Exit criteria
- environment resets without NaNs
- at least 100 consecutive steps run stably
- task metrics change in sensible directions

---

## Week 2 — Instrumentation and scripted baselines
### Objective
Make the environment measurable before RL.

### Required outputs
- scripted probe action sequence
- scripted scoop sequence
- scripted deposit sequence
- metric overlays on video
- spill metric and transfer metric validated manually

### Deliverable
`Week2_instrumentation_and_oracle.md`

### Exit criteria
- at least one scripted trajectory produces non-trivial transfer
- metrics agree with visual inspection
- failure cases can be categorized

---

## Week 3 — Reactive and recurrent baselines
### Objective
Establish lower baselines.

### Required outputs
- `Reactive PPO`
- `RNN-PPO`
- first train/eval scripts
- ID training curves
- first ID videos

### Deliverable
`Week3_baseline_report.md`

### Exit criteria
- both baselines run from CLI
- both can solve something above random
- neither uses privileged material state

---

## Week 4 — Teacher policy
### Objective
Train a privileged upper-bound teacher.

### Required outputs
- teacher observation with hidden material parameters / privileged state
- stable teacher RL run
- exported demonstrations

### Deliverable
`Week4_teacher_report.md`

### Exit criteria
- teacher clearly outperforms reactive baseline
- teacher demonstrations can be stored and replayed
- teacher policy is good enough to supervise a student

---

## Week 5 — Student and main Probe-Then-Act pipeline
### Objective
Build the full learning pipeline.

### Required outputs
- probe policy
- latent belief encoder
- uncertainty head
- student policy
- distillation or two-stage pipeline

### Deliverable
`Week5_main_method_report.md`

### Exit criteria
- end-to-end training runs
- student sees observation-only inputs
- at least one setting beats `RNN-PPO` on OOD-material or spill rate

---

## Week 6 — OOD evaluation and ablations
### Objective
Turn a working method into a publishable result.

### Required outputs
- OOD-material split
- OOD-tool split
- OOD-container split
- OOD-sensor split
- ablation runs

### Deliverable
`Week6_ood_and_ablation_report.md`

### Exit criteria
- main result table can be filled
- ablations isolate probe / belief / uncertainty contributions
- failure gallery exists

---

## Week 7 — Consolidation
### Objective
Reduce noise and remove weak claims.

### Required outputs
- repeated seeds
- confidence intervals
- cleaned tables
- best qualitative videos
- claim-to-evidence matrix

### Deliverable
`Week7_consolidation_report.md`

### Exit criteria
- no claim depends on one lucky seed
- main method advantage is consistent
- weak or unsupported claims are removed

---

## Week 8 — Paper package assembly
### Objective
Prepare a real submission package.

### Required outputs
- final tables
- final figures
- appendix material
- multimedia shortlist
- paper outline mapped to evidence

### Deliverable
`Week8_submission_package.md`

### Exit criteria
- all main paper numbers are frozen
- every figure is reproducible
- every claim can be defended with direct evidence

---

## 7. Immediate task allocation (first 72 hours)

### Environment Lead
- build the minimal scene
- pick the first tool geometry
- pick the first two material families
- define metrics for transfer and spill
- export 3 debug videos

### Learning Lead
- scaffold model directories
- wire `Reactive PPO`
- define teacher vs student observation interfaces
- create config templates

### Evaluation Lead
- define run naming
- create result manifest
- create CSV schema for aggregation
- draft main result table and ablation table

### Literature / Writing Lead
- build a literature spreadsheet
- group papers into:
  - active probing / tactile exploration
  - belief-conditioned RL / POMDP
  - deformable / granular manipulation
  - uncertainty-aware robot learning
  - simulator-based benchmarks
- write 1-page “What is new here?” memo

---

## 8. Decision gates

### Gate A — Kill unstable environments early
If the main task cannot run stably with scripted actions, stop method work and fix the environment.

### Gate B — Kill weak task definitions early
If success is visually ambiguous or metrics are noisy, redefine the task before running more training.

### Gate C — Do not over-invest in differentiable tricks
If the mainline result is not already promising, do not spend time on differentiable regularizers or exotic losses.

### Gate D — Prefer stronger evaluation over fancier architecture
If time is limited, invest in:
- better OOD splits,
- better metrics,
- better ablations,
not in deeper encoders.

---

## 9. Risk register and fallback plan

### Risk 1 — Multi-physics instability
**Symptoms:** NaNs, exploding particles, invalid contact.
**Fallback:** reduce to the most stable `MPM + Rigid` setup and fewer randomized parameters.

### Risk 2 — Task too hard for RL
**Symptoms:** flat reward, no transfer, no learning progress.
**Fallback:** first train shorter-horizon subskills or stronger teacher; simplify action space.

### Risk 3 — Tactile signal too noisy or not informative
**Symptoms:** no benefit from tactile branch.
**Fallback:** retain probe actions but rely more on contact force + proprioception + RGB-D.

### Risk 4 — OOD gain is weak
**Symptoms:** method only wins on ID.
**Fallback:** strengthen hidden-physics gap, improve split difficulty, or narrow paper claim.

### Risk 5 — Too many moving parts
**Symptoms:** nobody can tell what caused improvement.
**Fallback:** freeze mainline and run disciplined ablations.

---

## 10. Literature review protocol

### Output format
Maintain one spreadsheet with columns:
- `paper_title`
- `year`
- `venue`
- `task_domain`
- `active_probe`
- `latent_belief`
- `uncertainty`
- `deformable/material interaction`
- `sim_only_or_real`
- `main_claim`
- `possible_relation_to_our_work`

### Questions the literature lead must answer weekly
1. Who else does active information gathering before manipulation?
2. Who else does belief-conditioned control under hidden dynamics?
3. Who reports cross-material generalization?
4. Which papers only do application demos without method novelty?
5. Which missing comparison will a reviewer most likely ask for?

### Weekly literature output
Every week produce:
- 1 short summary note,
- 3 strongest related papers,
- 3 papers we are clearly different from,
- 3 reviewer concerns suggested by literature.

---

## 11. Meeting cadence

### Twice-weekly technical sync
Each owner reports:
- what changed,
- what broke,
- one visual artifact,
- one key next decision.

### Weekly research sync
Must review:
- result table status,
- unsupported claims,
- risk register,
- week exit criteria.

Do not hold abstract brainstorming meetings after week 2.

---

## 12. Status template for each owner

Use this format:

```text
Owner:
This week goal:
What was completed:
Evidence:
What is blocked:
What decision is needed:
Next 3 concrete actions:
```

“Evidence” must include a file path, figure, CSV, or video.

---

## 13. Claim discipline

### Allowed early claims
- the environment is stable
- metrics are meaningful
- baseline X underperforms baseline Y on split Z
- probe behavior appears to gather useful contact information

### Not allowed early claims
- “state-of-the-art”
- “strong generalization”
- “safe manipulation”
- “deployable to real robots”
- “foundation model for materials”

Large claims require a table.

---

## 14. What success looks like
The project is on track if by week 6 we have:
- two stable tasks,
- one strong teacher,
- at least two lower baselines,
- one main method,
- one filled main result table,
- one filled ablation table,
- one convincing OOD advantage,
- several good qualitative videos.

---

## 15. Final instruction to the team
Move from **idea excitement** to **evidence production** as early as possible.
A paper is not “promising” because the idea sounds good. It is promising when the tables start to stabilize.
