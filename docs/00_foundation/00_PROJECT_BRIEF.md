# T-RL Project Brief

## Project Goal
Build a **T-RL-targeted paper package** around a **method + benchmark** rather than a pure application demo. Because we currently have **simulation only** and **no real robot**, the paper must be framed as a contribution to **robot learning under hidden physical properties**, with strong emphasis on **generalization**, **robustness**, **safety/reliability**, and **reproducibility**.

---

## Why this direction fits T-RL
According to the official T-RL scope, the journal targets AI methods that address challenges specific to robotic and automation systems, including **limited pre-existing data**, **poor generalizability**, **slow learning timescales**, **low robustness in physical deployment**, and methods offering **safety and reliability guarantees**. T-RL also explicitly includes **benchmarks** that enable reproducibility and comparison, while noting that pure application papers are generally expected to include **real hardware** in addition to simulation. Therefore, our safest positioning is:

> **A robot learning method for manipulation under hidden physical properties + a reproducible cross-material benchmark in multi-physics simulation.**

**Positioning rule for the team:**
- Do **not** pitch this as “we used Genesis to do a cool task”.
- Do **not** pitch this as “sim-to-real”.
- Do pitch this as **robot learning under hidden physics**.
- Do pitch this as **active information gathering + uncertainty-aware control**.
- Do pitch this as **method + benchmark + evaluation protocol**.

---

## Preferred Paper Title
**Probe-Then-Act: Active Tactile System Identification for Robust Cross-Material Robot Tool Use in Multi-Physics Simulation**

### Alternate Title A
**Belief-Conditioned Tool Use under Hidden Material Properties in Multi-Physics Robot Learning**

### Alternate Title B
**Active Probing for Robust Cross-Material Manipulation in Multi-Physics Simulation**

---

## Draft Abstract (English)
**Draft only — revise after final experiments.**

Robotic tool use on deformable and flowable materials is challenging because key physical properties of the manipulated medium are often hidden and cannot be inferred reliably from a single observation. Policies trained reactively on instantaneous observations therefore tend to fail under cross-material variation, leading to spillage, unstable contact, and poor out-of-distribution generalization. We propose **Probe-Then-Act**, a robot learning framework that performs short-horizon **active probing** before task execution, collects visual, tactile, and proprioceptive interaction traces, and infers a latent belief over hidden material properties together with an uncertainty estimate. The downstream manipulation policy is conditioned on this belief and optimized to complete the task while explicitly reducing spillage and contact instability. To evaluate the method, we introduce a **cross-material multi-physics benchmark** in Genesis that combines rigid-body robot control with non-rigid material simulation across granular, liquid-like, and elasto-plastic media. We instantiate the benchmark on **scoop-and-transfer** and **level-and-fill** tasks, and evaluate generalization across unseen material parameters, held-out material families, tool geometries, container shapes, and sensor perturbations. Experiments are designed to compare active probing, passive history-based policies, domain randomization, and routing-by-material-class baselines. The target outcome of this work is to show that explicit physical belief inference and uncertainty-aware control substantially improve robustness in robot learning under hidden physics.

---

## Target Contributions (English)
**Draft only — these are the target claims that the experiments must support.**

1. **A cross-material robot learning benchmark under hidden physical properties.**  
   We will build a reproducible benchmark for tool-use manipulation where the manipulated medium varies across material families and continuous physical parameters, and where the robot must act without direct access to ground-truth material properties.

2. **An active probing and belief-conditioned manipulation framework.**  
   We will develop a two-stage policy that first performs short-horizon active probing and then executes the task conditioned on a latent physical belief and its uncertainty, rather than relying on purely reactive control.

3. **A robustness-oriented evaluation protocol for hidden-physics manipulation.**  
   We will evaluate not only task success, but also spillage, unstable contacts, contact-induced failure, efficiency, and uncertainty calibration under multiple out-of-distribution settings.

---

## One-Sentence Paper Pitch
**We study robot learning under hidden physical properties, where a robot must first gather information about an unknown material and then adapt its manipulation strategy accordingly.**

---

## Core Research Question
Can **active probing + latent physical belief inference + uncertainty-aware control** improve **robust cross-material robot tool use** more than:
- reactive RL,
- recurrent history-based RL,
- domain randomization alone,
- or discrete material classification followed by policy routing?

---

## Method Overview
### Working Method Name
**Probe-Then-Act**

### High-Level Pipeline
1. **Probe Policy**
   - Execute 1–3 short probing actions before the main task.
   - Candidate probe primitives: `tap`, `press`, `drag`, `skim`, `poke`, `partial insert`.
   - Goal: maximize information gain about hidden material properties.

2. **Physical Belief Encoder**
   - Inputs: `RGB/depth observation`, `tactile trace`, `contact force`, `proprioception`, `probe action history`.
   - Outputs: latent belief `z` and uncertainty `sigma`.
   - `z` should represent actionable physical properties rather than semantic material labels only.

3. **Task Policy**
   - Policy input: `(current observation, z, sigma)`.
   - Policy output: end-effector motion commands + tool interaction commands.
   - Objective: maximize task completion while minimizing spillage and unstable contact.

4. **Risk Head / Safety Head**
   - Predict short-horizon failure events:
     - `spill risk`
     - `jam risk`
     - `unstable contact risk`
   - Use this head for action filtering, loss shaping, or auxiliary training.

---

## Simulation Setup in Genesis
### Robot
- **Franka Panda** with a simple tool-attached end-effector.
- Start with Cartesian delta control.

### Tool Set
- `spoon-like scoop`
- `flat spatula`
- `angled scoop`
- Optional held-out tool shapes for OOD evaluation.

### Material Families
Use **multi-physics** support to create hidden-physics manipulation settings.

**Recommended primary set:**
- `MPM.Sand()`
- `MPM.Snow()`
- `MPM.ElastoPlastic()`
- `MPM.Liquid()` or `SPH.Liquid()` depending stability and coupling needs

### Containers / Scenes
- source bin
- target cup / bowl / tray
- flat target region for leveling
- held-out container geometries for generalization

### Sensors
- RGB camera / depth camera
- proprioception
- contact force
- tactile sensing via `KinematicContactProbe`
- optional `ElastomerDisplacement` for richer tactile observations

---

## Recommended Task Suite
### Task A: Scoop-and-Transfer
The robot scoops material from a source bin and deposits it into a target container.

**Key metrics:**
- success rate
- transferred mass / particle count
- spillage ratio
- execution time
- unstable contact rate

### Task B: Level-and-Fill
The robot spreads or levels material into a target region with minimal overfill and minimal empty area.

**Key metrics:**
- fill coverage
- levelness / height variance
- overshoot / waste
- completion time

### Why only two tasks initially
Two tasks are enough to show the method is not single-task overfitting, while still keeping the project executable. Do not add a third task until both of the above are stable.

---

## Observation Design
### Student Observation
- RGB or RGB-D
- tactile traces
- proprioception
- end-effector state
- recent action history

### Privileged Teacher Observation
- full simulator state
- hidden material parameters
- object / particle statistics if needed

### Important Rule
The **teacher** can use privileged information. The **student** cannot.

---

## Training Strategy
### Stage 1: Privileged Teacher Training
- RL with privileged state
- objective: learn a strong upper-bound policy
- output: teacher policy and optionally teacher value / action targets

### Stage 2: Student Distillation + RL Fine-Tuning
- distill teacher knowledge into a student policy that uses realistic observations only
- then fine-tune with RL under observation noise and domain randomization

### Optional Auxiliary Objectives
- latent prediction consistency
- future material motion prediction
- next-step contact prediction
- risk prediction for spill / jam / instability

---

## What We Need to Show Empirically
The paper should show **all three** of the following:

1. **Probe-Then-Act beats reactive policies under hidden physics.**
2. **Explicit belief inference beats passive memory alone.**
3. **Uncertainty-aware control improves robustness and failure avoidance.**

If one of these is not supported by experiments, revise the paper scope before writing.

---

## Baseline List
### Core Baselines
1. **Reactive PPO**
   - current observation only
   - no probing
   - no belief model

2. **RNN-PPO / GRU-PPO**
   - history-based policy
   - no explicit probing objective
   - no explicit physical belief

3. **Domain Randomization PPO**
   - aggressive randomization of material parameters and sensor noise
   - test whether randomization alone is enough

4. **Fixed-Probe + PPO**
   - hand-designed probe sequence
   - downstream policy sees the resulting traces
   - tests whether learned probing matters

5. **Material Classification + Expert Routing**
   - predict a discrete material class from probe data
   - route to one of several expert policies
   - tests whether continuous belief is better than discrete routing

6. **Ours w/o Uncertainty**
   - keep probing + latent belief
   - remove uncertainty estimation and risk head

7. **Privileged Upper Bound**
   - direct access to hidden physical parameters / privileged state
   - used only as an upper bound, not as the main competitor

### Optional Stronger Baselines
- **Bayesian belief filter + controller**
- **Contrastive latent dynamics policy**
- **Decision Transformer with history** if data collection becomes large enough

---

## Main Results Table Template
| Method | Active Probing | Explicit Belief | Uncertainty / Risk Head | ID Success ↑ | OOD-Param Success ↑ | OOD-Material Success ↑ | OOD-Tool Success ↑ | Spillage ↓ | Jam / Instability ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Reactive PPO | No | No | No |  |  |  |  |  |  |
| RNN-PPO | No | No | No |  |  |  |  |  |  |
| Domain Randomization PPO | No | No | No |  |  |  |  |  |  |
| Fixed-Probe + PPO | Yes | No | No |  |  |  |  |  |  |
| Material Class + Expert Routing | Yes | Weak (discrete) | No |  |  |  |  |  |  |
| Ours w/o Uncertainty | Yes | Yes | No |  |  |  |  |  |  |
| **Probe-Then-Act (Ours)** | **Yes** | **Yes** | **Yes** |  |  |  |  |  |  |
| Privileged Upper Bound | Optional | Oracle | Optional |  |  |  |  |  |  |

---

## Ablation Table Template
| Variant | Learned Probe | Tactile Input | Belief Latent `z` | Uncertainty `sigma` | Teacher-Student | OOD-Material Success ↑ | Spillage ↓ | Calibration Error ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Full model | Yes | Yes | Yes | Yes | Yes |  |  |  |
| No probing | No | Yes | No | No | Yes |  |  |  |
| Random probing | Random | Yes | Yes | Yes | Yes |  |  |  |
| No tactile | Yes | No | Yes | Yes | Yes |  |  |  |
| No uncertainty | Yes | Yes | Yes | No | Yes |  |  |  |
| No teacher-student | Yes | Yes | Yes | Yes | No |  |  |  |

---

## OOD Evaluation Axes
The paper must include **multiple OOD settings**. At minimum:

1. **OOD-Parameter**
   - unseen viscosity / yield / friction / internal material coefficients

2. **OOD-Material Family**
   - train on subset of materials, test on held-out material families

3. **OOD-Tool Geometry**
   - train on tool set A, test on held-out tool shape B

4. **OOD-Container Geometry**
   - unseen target container shape / size / depth

5. **OOD-Sensor Perturbation**
   - visual noise, tactile bias, force noise, latency

---

## Metrics
### Primary Metrics
- **Task Success Rate**
- **Transferred Mass / Effective Fill**
- **Spillage Ratio**
- **Failure Rate due to Contact Instability**

### Secondary Metrics
- **Execution Time**
- **Action Smoothness**
- **Probe Cost**
- **Uncertainty Calibration Error**
- **Generalization Gap** between ID and OOD

### Important Rule
Do **not** report only reward. T-RL reviewers will care more about task-level robustness metrics than a single training reward curve.

---

## Required Figures for the Paper
1. **Method Overview Figure**
   - probe phase → belief inference → task execution → risk-aware action selection

2. **Benchmark Figure**
   - material families, tools, containers, tasks

3. **Main Result Figure**
   - OOD performance bars or line plots across settings

4. **Tactile / Belief Visualization Figure**
   - probe traces and latent belief clustering or belief trajectories

5. **Failure Analysis Figure**
   - examples of spill, jam, collapse, and recovery behavior

---

## Required Deliverables from the Team
### Deliverable D1 — Environment Package
- stable Genesis environments for Task A and Task B
- deterministic reset utilities
- configurable material randomization
- logging for particles / mass / contact / spillage

### Deliverable D2 — Baseline Package
- Reactive PPO
- RNN-PPO
- Domain Randomization PPO
- Fixed-Probe + PPO

### Deliverable D3 — Method Package
- learned probe policy
- latent physical belief encoder
- uncertainty head
- risk head
- teacher-student training pipeline

### Deliverable D4 — Evaluation Package
- automatic OOD benchmark runner
- standardized metrics export
- plotting scripts
- seed sweep and statistical summary

### Deliverable D5 — Paper Package
- abstract v1 / intro v1
- figure drafts
- main result tables
- appendix experiment details

---

## Team Execution Plan
### Workstream 1 — Environment & Physics
**Owner:** Simulation lead

Tasks:
- build `scoop-and-transfer` environment
- build `level-and-fill` environment
- verify stable multi-physics stepping
- expose hidden material parameters for logging only
- implement spillage / transferred-mass estimators

### Workstream 2 — Sensors & Observation Stack
**Owner:** Perception lead

Tasks:
- camera observation interface
- tactile interface with `KinematicContactProbe`
- optional `ElastomerDisplacement`
- synchronized probe trace serialization
- observation normalization and caching

### Workstream 3 — Baselines
**Owner:** RL lead

Tasks:
- Reactive PPO
- RNN-PPO
- Domain Randomization PPO
- Fixed-Probe baseline
- reproducible training config and seed control

### Workstream 4 — Main Method
**Owner:** Method lead

Tasks:
- learned probe policy
- latent belief encoder
- uncertainty estimation head
- risk-aware task policy
- teacher-student distillation

### Workstream 5 — Evaluation & Writing
**Owner:** Paper lead

Tasks:
- benchmark protocol
- OOD split definition
- tables and plots
- failure case curation
- draft sections and appendix

---

## Suggested 8-Week Milestone Plan
### Week 1–2
- finalize task definitions
- build stable environments
- verify metric correctness
- produce sanity-check videos

### Week 3
- complete Reactive PPO and RNN-PPO
- obtain first ID learning curves

### Week 4
- complete Domain Randomization and Fixed-Probe baselines
- define OOD splits formally

### Week 5
- implement probe policy + belief encoder
- run first end-to-end training

### Week 6
- add uncertainty / risk head
- run ablations

### Week 7
- full seed sweep
- collect figures and tables
- freeze experiment protocol

### Week 8
- write paper draft
- assemble appendix
- finalize reviewer-facing narrative

---

## Non-Negotiable Writing Rules
1. **Do not claim sim-to-real.** We do not have hardware evidence.
2. **Do not oversell Genesis speed as the paper contribution.** Genesis is the platform, not the scientific novelty.
3. **Do not write this as a pure benchmark paper unless the method is weak.** Preferred framing is **method + benchmark**.
4. **Do not present semantic material classification as the main innovation.** The main idea is **actionable physical belief under hidden physics**.
5. **Do not hide negative results.** If some materials fail badly, turn that into a failure analysis section.

---

## Reviewer-Facing Narrative
### The story we want reviewers to believe
- Hidden physical properties are a core bottleneck for robot manipulation.
- Purely reactive policies generalize poorly.
- Passive memory helps, but explicit active probing is better.
- Belief-conditioned control is more robust than material-class routing.
- Uncertainty matters because failure under hidden physics is asymmetric and safety-relevant.

### The story we must avoid
- “We made a policy in Genesis and it worked.”
- “This is sim-to-real without real-world validation.”
- “More compute and more domain randomization solve everything.”

---

## Submission Notes for T-RL
T-RL accepts **regular papers** and **survey papers**. Official author guidance states that accepted regular papers may be invited for presentation at selected conferences, and currently lists transfer windows for **IROS 2026** and **CASE 2026** through **April 30, 2026**. The journal scope strongly favors contributions with clear theoretical or practical significance, and benchmark-style work is explicitly welcomed when it enables reproducibility and comparison.

**Practical implication for us:**
- write for **journal quality**, not conference-paper quality;
- include strong supplementary material;
- release reproducible configs, seeds, and evaluation scripts if possible;
- if we miss the near-term conference presentation window, the journal submission still remains valid.

---

## Immediate Action Items (send this to the team)
1. Freeze the project name as **Probe-Then-Act**.
2. Implement **Task A** first, **Task B** second.
3. Use **MPM-based materials** as the default path.
4. Keep **SPH.Liquid** only if coupling and stability are satisfactory.
5. Finish **Reactive PPO**, **RNN-PPO**, and **Fixed-Probe** before touching fancy models.
6. Define **OOD splits** before training the final method.
7. Every experiment must log:
   - success
   - transferred mass
   - spillage
   - contact instability
   - execution time
8. Save videos for both success and failure cases from day one.

---

## Minimal Success Criteria
The project is worth writing up only if **Probe-Then-Act** achieves at least two of the following:
- clearly better **OOD-Material** performance than RNN-PPO,
- clearly lower **spillage** than all non-oracle baselines,
- clearly better **OOD-Tool** transfer than Fixed-Probe,
- meaningfully calibrated uncertainty that correlates with failure risk.

If none of these hold, pivot the project before paper writing.

---

## References (official sources)
1. IEEE Robotics and Automation Society, **Transactions on Robot Learning (T-RL)** — Purpose, mission, and scope.  
   https://www.ieee-ras.org/publications/t-rl/
2. IEEE Robotics and Automation Society, **T-RL Information for Authors**.  
   https://www.ieee-ras.org/publications/t-rl/information-for-authors/
3. Genesis documentation, **Beyond Rigid Bodies**.  
   https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/beyond_rigid_bodies.html
4. Genesis documentation, **Sensors**.  
   https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/sensors.html
5. Genesis documentation, **Manipulation with Two-Stage Training**.  
   https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/manipulation.html
6. Genesis documentation, **Config System**.  
   https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/config_system.html
7. Genesis documentation, **Non-rigid Coupling**.  
   https://genesis-world.readthedocs.io/en/latest/user_guide/advanced_topics/solvers_and_coupling.html
