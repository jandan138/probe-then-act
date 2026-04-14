# 03_EXPERIMENT_PROTOCOL

## 1. Goal
This document specifies the **minimum publishable experimental package** for the `probe-then-act` project.

Use this as the contract between:
- environment development,
- method development,
- evaluation,
- and paper writing.

If a result is not part of this protocol, it is secondary.

---

## 2. Main scientific question

Can **active probing + latent physical belief inference + uncertainty-aware control** improve robustness in robot tool use under **hidden material properties**, compared with:
- reactive policies,
- recurrent history-based policies,
- domain randomization,
- and discrete material routing baselines?

---

## 3. Hypotheses

### H1 — Active probing helps under hidden physics
Policies that explicitly gather information before action should outperform purely reactive policies under OOD material variation.

### H2 — Explicit belief beats passive memory
A method with explicit latent physical belief should outperform a recurrent policy that only gets observation history.

### H3 — Uncertainty helps robustness
Adding uncertainty-aware control / risk prediction should reduce spillage and unstable-contact failures, especially on OOD splits.

### H4 — Method value is larger under OOD than ID
The gap between the main method and simple baselines should be more visible on OOD-material, OOD-tool, and OOD-sensor splits than on the ID split.

---

## 4. Tasks

## Task A — Scoop-and-Transfer
### Description
The robot uses a scoop-like tool to collect material from a source container and transfer it to a target container.

### Main success variables
- amount transferred into target
- amount spilled outside valid regions
- whether a minimum target-fill threshold is reached

### Why this task matters
This is the most direct test of:
- cross-material adaptation,
- contact-rich behavior,
- and failure modes like spill / collapse / tool jam.

---

## Task B — Level-and-Fill
### Description
The robot spreads material into a target region while minimizing uneven fill and material waste.

### Main success variables
- target coverage
- height variance / levelness
- overspill / waste

### Why this task matters
This tests whether the learned belief affects not only pickup, but also downstream contact strategy.

---

## 5. Simulation design choices

### Mainline setup
Use the most stable setup first:
- one Franka arm
- rigid robot links
- rigid containers
- rigid attached tool geometry
- MPM material families

### Recommended first material set
- `sand-like`
- `snow-like`
- `elasto-plastic`
- `liquid-like` only if stable enough

### Recommended first sensing stack
- RGB-D or depth + RGB
- proprioception
- contact force
- tactile-like contact trace if stable and informative

### Important caution
Do not make the first paper depend on the most fragile simulator feature.
The mainline should survive without experimental extras.

---

## 6. Methods to include

### M0 — Random / weak heuristic
Purpose:
- sanity check lower bound
- verify task metrics

### M1 — Reactive PPO
Inputs:
- current observation only

Purpose:
- simplest learnable baseline

### M2 — RNN-PPO / GRU-PPO
Inputs:
- observation history

Purpose:
- tests whether passive memory is enough

### M3 — Domain Randomization PPO
Inputs:
- same as M1 or M2

Purpose:
- tests whether large randomization alone solves hidden-physics generalization

### M4 — Fixed-Probe + PPO
Behavior:
- use a scripted probe sequence before manipulation

Purpose:
- isolates value of probing itself vs learned probing

### M5 — Material Classifier + Routed Experts
Behavior:
- probe, classify material into a discrete class, route to an expert policy

Purpose:
- compares explicit continuous belief with coarse discrete routing

### M6 — Ours w/o uncertainty
Behavior:
- full probe + belief pipeline, but no uncertainty/risk head

Purpose:
- isolates the effect of uncertainty-aware control

### M7 — Ours: Probe-Then-Act
Behavior:
- active probing
- latent physical belief
- uncertainty / risk modeling
- belief-conditioned task policy

### M8 — Privileged Teacher Upper Bound
Behavior:
- full state / hidden material information allowed

Purpose:
- shows approximate ceiling and problem solvability

---

## 7. Minimum experiment matrix

You do **not** need all possible combinations at first.
You do need the following minimum matrix.

| Task | Split | Methods |
|---|---|---|
| Scoop-and-Transfer | ID | M1, M2, M3, M4, M6, M7, M8 |
| Scoop-and-Transfer | OOD-Material | M1, M2, M3, M4, M6, M7 |
| Scoop-and-Transfer | OOD-Tool | M1, M2, M3, M4, M6, M7 |
| Scoop-and-Transfer | OOD-Sensor | M1, M2, M3, M4, M6, M7 |
| Level-and-Fill | ID | M1, M2, M6, M7, M8 |
| Level-and-Fill | OOD-Material | M1, M2, M6, M7 |
| Level-and-Fill | OOD-Tool | M1, M2, M6, M7 |

If time is tight, prioritize:
1. Scoop-and-Transfer
2. OOD-Material
3. OOD-Tool
4. Level-and-Fill
5. OOD-Sensor
6. OOD-Container

---

## 8. Split design

### 8.1 ID split
Training and test share the same families, but test uses held-out random seeds / parameter draws.

### 8.2 OOD-Material split
Hold out at least one material family or a physically distinct parameter regime.

Examples:
- train on `sand-like + elasto-plastic`
- test on `snow-like`
or
- train on lower-viscosity band
- test on higher-viscosity band

### 8.3 OOD-Tool split
Train on one or two tool geometries and test on held-out tool geometry.

### 8.4 OOD-Container split
Train on source/target container shapes A/B and test on C.

### 8.5 OOD-Sensor split
Train on nominal sensor settings and test on perturbations:
- noise
- blur
- latency
- missing tactile samples
- calibration drift

### Split rule
The split definition file must be versioned and must list explicit IDs.
No implicit split logic allowed.

---

## 9. Metrics

## Primary metrics
### 9.1 Success Rate
Binary or thresholded task completion.
Useful but insufficient on its own.

### 9.2 Transfer Efficiency
Fraction of source material that ends in the valid target region.

### 9.3 Spill Ratio
Fraction of material ending outside valid target / workspace bounds.

### 9.4 Contact Failure Rate
Frequency of:
- unstable contact
- jam
- collapse-like failure
- large-impact failure

### 9.5 Task Time / Step Count
Measures efficiency.

---

## Secondary metrics
### 9.6 Coverage / Fill Quality
Especially for `level-and-fill`.

### 9.7 Height Variance / Levelness
Measures smoothness of final surface.

### 9.8 Uncertainty Calibration
Expected Calibration Error (ECE) or reliability-style metric for predicted failure or success probabilities.

### 9.9 Probe Utility
Optional but valuable:
- mutual-information proxy
- prediction gain
- downstream performance improvement conditioned on probing

---

## 10. Failure taxonomy

Every failed rollout should be categorized into one of:
1. **No acquisition** — failed to collect meaningful material
2. **Premature loss** — material lost before target region
3. **Deposit failure** — reaches target but fails to deposit
4. **Over-aggressive contact** — contact destabilizes scene
5. **Wrong strategy under hidden physics** — behavior clearly mismatched to material
6. **Sensor-induced failure** — policy breaks under perturbation
7. **Recovery failure** — initial mistake could have been corrected but was not

This taxonomy is required for qualitative analysis.

---

## 11. Main result table template

Use this exact table structure in the repo.

| Method | Active Probe | Explicit Belief | Uncertainty/Risk | ID Success ↑ | OOD-Material Success ↑ | OOD-Tool Success ↑ | OOD-Sensor Success ↑ | Spill Ratio ↓ | Contact Failure ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reactive PPO | No | No | No |  |  |  |  |  |  |
| RNN-PPO | No | No | No |  |  |  |  |  |  |
| Domain Randomization PPO | No | No | No |  |  |  |  |  |  |
| Fixed-Probe + PPO | Yes | No | No |  |  |  |  |  |  |
| Material Router | Yes | Weak/Discrete | No |  |  |  |  |  |  |
| Ours w/o Uncertainty | Yes | Yes | No |  |  |  |  |  |  |
| **Probe-Then-Act** | **Yes** | **Yes** | **Yes** |  |  |  |  |  |  |
| Privileged Teacher | Optional | Oracle | Optional |  |  |  |  |  |  |

---

## 12. Ablation table template

| Variant | Learned Probe | Tactile Signal | Belief Latent z | Uncertainty σ | Teacher-Student | OOD-Material Success ↑ | Spill Ratio ↓ | Calibration Error ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full model | Yes | Yes | Yes | Yes | Yes |  |  |  |
| No probe | No | Yes | No | No | Yes |  |  |  |
| Random probe | Random | Yes | Yes | Yes | Yes |  |  |  |
| No tactile | Yes | No | Yes | Yes | Yes |  |  |  |
| No uncertainty | Yes | Yes | Yes | No | Yes |  |  |  |
| No teacher-student | Yes | Yes | Yes | Yes | No |  |  |  |

---

## 13. Seed and statistics policy

### Minimum seed count
- exploratory stage: 3 seeds
- paper stage: 5 seeds minimum for main results if compute allows

### Reporting
Report:
- mean
- standard deviation or confidence interval
- seed count

Do not report best seed only.

### Statistical discipline
If the gain is tiny and unstable, narrow the claim rather than overstate it.

---

## 14. Required qualitative artifacts

At minimum, produce:
1. one success montage
2. one failure montage
3. one side-by-side baseline comparison
4. one probe visualization
5. one OOD-material comparison

Qualitative artifacts should support—not replace—the tables.

---

## 15. Sanity checks before large sweeps

Before launching multi-seed sweeps, verify:
- the student cannot access privileged material parameters
- probe actions actually change contact observations
- reward and metric are correlated but not identical
- OOD split examples look visually different from ID
- the teacher is meaningfully stronger than the student at initialization

---

## 16. Oracle / scripted baselines

Before any RL comparison is trusted, implement:
- scripted probe sequence
- heuristic scoop motion
- heuristic deposit motion

Why:
- proves task is mechanically solvable
- validates metrics
- helps debug reward scale
- provides rollout references for failure analysis

---

## 17. Decision rules for paper claims

### Claim A
“Active probing improves cross-material robustness.”

Required evidence:
- M7 > M1 and M2 on OOD-Material
- improvement visible in both success and spill/contact metrics

### Claim B
“Explicit belief outperforms passive memory.”

Required evidence:
- M7 > M2 on OOD-Material or OOD-Tool
- not just one lucky seed

### Claim C
“Uncertainty improves failure avoidance.”

Required evidence:
- M7 > M6 in spill/contact failure
- uncertainty quality not obviously degenerate

### Claim D
“Method generalizes beyond the training setup.”

Required evidence:
- improvement on at least two OOD splits, not only ID

If these are not met, remove or soften the corresponding claim.

---

## 18. Submission-oriented checklist

For a credible T-RL submission package, prepare:
- main paper tables
- appendix tables
- split definitions
- config files
- qualitative videos
- training details
- failure taxonomy
- ablation details
- limitations section

Also remember:
- T-RL regular/survey submissions are capped at **12 pages in Transactions format**
- abstract length is **up to 200 words**
- double-anonymous review applies
- multimedia can be submitted as a single zipped archive up to **60 MB**
- T-RL emphasizes robot-learning advances under physical-system constraints, generalizability, robustness, and safety/reliability, while also noting that application papers are generally expected to include real hardware in addition to simulation

Because this project is sim-only, the paper must lean hard into:
- method novelty,
- benchmark value,
- evaluation rigor,
- and reproducibility.

---

## 19. Genesis-specific implementation guidance

### Mainline recommendation
Start with:
- `MPM ↔ Rigid`
- rigid containers
- rigid tool geometry
- camera + proprio + contact signals

### Why
Genesis documentation currently exposes:
- multiple solver families including `SPH`, `MPM`, and `PBD`
- tactile sensor interfaces such as `KinematicContactProbe` and `ElastomerDisplacement`
- a documented two-stage teacher-student manipulation example
- supported solver pairs including `MPM ↔ Rigid`, `MPM ↔ PBD`, `SPH ↔ Rigid`, `PBD ↔ Rigid`, and `Tool ↔ MPM`

### Caution
`ToolSolver` is documented as a temporary workaround for differentiable rigid-soft interaction and supports one-way `tool -> other` coupling without full internal rigid dynamics. Therefore:
- do **not** make the first paper depend on `ToolSolver`
- use it only if the core pipeline is already working and there is a clear need

---

## 20. Final criterion for “minimum publishable package”
The project is minimally publishable when all of the following are true:
1. two tasks are implemented and stable
2. at least two non-trivial baselines are trained
3. the teacher-student path works
4. the main method works
5. the main result table is mostly filled
6. at least one OOD split shows a convincing advantage
7. the ablation table isolates probe/belief/uncertainty effects
8. videos and failure analysis support the numerical story

Until then, treat everything as pre-paper engineering.


---

## 21. Official references and submission snapshot

### T-RL author-side notes (snapshot: 2026-04-03)
- T-RL regular and survey submissions are capped at **12 pages** in Transactions format.
- Abstract length is **up to 200 words**.
- Review is **double-anonymous**.
- Multimedia can be uploaded as **one zipped archive up to 60 MB**.
- T-RL regular papers can be transferred for conference presentation; the author page currently lists:
  - `IROS 2026`: 2025-08-01 to 2026-04-30
  - `CASE 2026`: 2025-08-01 to 2026-04-30

### Genesis docs relevant to this project
- User Guide: `https://genesis-world.readthedocs.io/en/latest/user_guide/index.html`
- Beyond Rigid Bodies: `https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/beyond_rigid_bodies.html`
- Sensors: `https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/sensors.html`
- Manipulation with Two-Stage Training: `https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/manipulation.html`
- Non-rigid Coupling: `https://genesis-world.readthedocs.io/en/latest/user_guide/advanced_topics/solvers_and_coupling.html`
- ToolOptions note (temporary `ToolSolver`): `https://genesis-world.readthedocs.io/zh-cn/latest/api_reference/options/simulator_coupler_and_solver_options/tool_options.html`

### Instruction to the team
Before freezing the final paper package, re-check the official T-RL author page and the relevant Genesis docs again.
Do not rely on a months-old screenshot or memory.
