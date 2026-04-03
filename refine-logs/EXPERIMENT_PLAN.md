# Experiment Plan

> Input for ARIS `/experiment-bridge`. Save as `refine-logs/EXPERIMENT_PLAN.md`.

**Problem**: Robot manipulation policies fail under hidden material variation because they cannot infer material properties from single observations, leading to spillage, unstable contact, and poor OOD generalization.

**Method Thesis**: Active probing (1-3 short diagnostic actions) + latent physical belief inference (z, sigma) + uncertainty-aware policy conditioning enables robust cross-material robot tool use.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|-------|----------------|----------------------------|---------------|
| C1: Active probing helps under hidden physics | Core novelty — probing before action | M7 > M1, M2 on OOD-Material (success + spill) | B1, B2 |
| C2: Explicit belief beats passive memory | Distinguishes from RNN-based approaches | M7 > M2 on OOD-Material or OOD-Tool, not just one seed | B1, B3 |
| C3: Uncertainty improves failure avoidance | Justifies the sigma + risk head | M7 > M6 in spill/contact failure | B1, B4 |
| C4: Method generalizes beyond training setup | Paper-level claim for T-RL | Improvement on >= 2 OOD splits, not only ID | B1, B5 |

## Experiment Blocks

### Block 0: Environment Validation
- **Claim tested**: None (infrastructure)
- **Task**: Scoop-and-Transfer, ID split
- **Systems**: Scripted baseline (oracle scoop motion)
- **Metrics**: success_rate, transfer_efficiency, spill_ratio
- **Success criterion**: Scripted policy achieves > 0% transfer; metrics are plausible
- **Status**: DONE — sanity check passed, 400 particles stable
- **Priority**: DONE

### Block 1: Main Result Table
- **Claim tested**: C1, C2, C3, C4
- **Task**: Scoop-and-Transfer
- **Split**: ID + OOD-Material + OOD-Tool + OOD-Sensor
- **Compared systems**: M1 (Reactive PPO), M2 (RNN-PPO), M3 (DomainRand PPO), M4 (Fixed-Probe+PPO), M5 (Material Router), M6 (Ours no-uncertainty), M7 (Probe-Then-Act), M8 (Privileged Teacher)
- **Metrics**: Primary: success_rate, transfer_efficiency, spill_ratio, contact_failure_rate. Secondary: episode_length
- **Setup details**:
  - All methods: PPO, lr=3e-4, gamma=0.99, n_steps=128, batch=64, 10M steps
  - Teacher: privileged obs (ground-truth material params)
  - Student methods: student obs only (proprio + tactile + camera)
  - Seeds: 3 exploratory, 5 for paper
- **Success criterion**: M7 beats M1, M2, M3 on at least 2 OOD splits by > 5% success or > 10% spill reduction
- **Failure interpretation**: If M7 = M2, probing adds no value → rethink probe design or increase hidden physics gap
- **Priority**: MUST-RUN

### Block 2: Baseline Reproduction
- **Claim tested**: C1 (lower bound established)
- **Task**: Scoop-and-Transfer, ID split
- **Compared systems**: M1, M2, M3, M4, M8
- **Setup**: Same hyperparams, 3 seeds each
- **Success criterion**: All baselines learn above random; teacher clearly beats reactive
- **Decision gate**: If M1 cannot learn above random → simplify task/reward
- **Priority**: MUST-RUN (prerequisite for Block 1)

### Block 3: Teacher-Student Distillation
- **Claim tested**: Teacher-student pipeline works
- **Task**: Scoop-and-Transfer, ID split
- **Systems**: M8 (teacher) → behavior cloning → RL fine-tune → M7 (student)
- **Setup**: 100K teacher demos, BC 100 epochs, fine-tune 2M steps
- **Success criterion**: Student performance within 80% of teacher on ID
- **Priority**: MUST-RUN (prerequisite for main method)

### Block 4: Ablation Study
- **Claim tested**: C1, C2, C3 (component isolation)
- **Task**: Scoop-and-Transfer, OOD-Material split
- **Compared systems**: Full M7, No-probe, Random-probe, No-tactile, No-uncertainty, No-teacher-student
- **Metrics**: success_rate, spill_ratio, calibration_error
- **Success criterion**: Each component contributes measurably (> 3% on primary metric)
- **Priority**: MUST-RUN

### Block 5: OOD Generalization
- **Claim tested**: C4
- **Task**: Scoop-and-Transfer
- **Splits**: OOD-Material (snow held out), OOD-Tool (spatula), OOD-Sensor (noise/blur)
- **Systems**: M1, M2, M3, M7
- **Success criterion**: Generalization gap (ID-OOD) is smaller for M7 than baselines
- **Priority**: MUST-RUN

### Block 6: Second Task (Level-and-Fill)
- **Claim tested**: Method is not single-task overfitting
- **Task**: Level-and-Fill, ID + OOD-Material
- **Systems**: M1, M2, M6, M7, M8
- **Success criterion**: M7 advantage holds on second task
- **Priority**: NICE-TO-HAVE (only if Scoop-Transfer results are strong)

## Run Order

| Milestone | Goal | Blocks | Decision Gate | GPU-hours |
|-----------|------|--------|---------------|-----------|
| M0: Sanity | Env works | B0 | Scripted policy transfers particles? | 0.5h (DONE) |
| M1: Baselines | Lower bounds | B2 | All learn above random? Teacher > reactive? | ~24h |
| M2: Distillation | Pipeline | B3 | Student within 80% of teacher? | ~8h |
| M3: Main method | Full PTA | B1 (partial) | M7 beats M2 on OOD? | ~16h |
| M4: Ablations | Components | B4 | Each component matters? | ~12h |
| M5: Full eval | All splits | B1, B5 | Table fills, >= 2 OOD wins? | ~20h |
| M6: Second task | Generality | B6 | Advantage holds? | ~12h |

## Compute Budget
- **Total estimated GPU-hours**: ~92h
- **Hardware**: 1x RTX 4090 (24GB), Vast.ai available for parallel seeds
- **Biggest bottleneck**: M1 baseline training (5 methods x 3 seeds x ~1.5h = 22.5h)
- **Parallelization strategy**: Train M1-M4 sequentially on local GPU, use Vast.ai for seed sweeps

## Risks
- **Risk**: Genesis MPM too slow for 10M steps → **Mitigation**: Reduce to 2M steps, increase n_envs
- **Risk**: PPO cannot learn scooping → **Mitigation**: Simplify reward, add reward shaping, try SAC
- **Risk**: Teacher not strong enough → **Mitigation**: Give teacher more privileged info, train longer
- **Risk**: OOD gap too small to show method value → **Mitigation**: Increase material diversity, widen parameter ranges
- **Risk**: Single GPU bottleneck → **Mitigation**: Use `/vast-gpu` for parallel training

## Method Configuration Reference

| Method | ID | Probe | Belief | Uncertainty | Obs Type | Notes |
|--------|-----|-------|--------|-------------|----------|-------|
| Reactive PPO | M1 | No | No | No | Student | Simplest baseline |
| RNN-PPO | M2 | No | No | No | Student + history | Tests passive memory |
| DomainRand PPO | M3 | No | No | No | Student | Tests randomization alone |
| Fixed-Probe + PPO | M4 | Scripted | No | No | Student + probe traces | Tests probing value |
| Material Router | M5 | Scripted | Discrete | No | Student | Tests discrete vs continuous |
| Ours no-unc | M6 | Learned | Yes | No | Student | Ablation |
| **Probe-Then-Act** | **M7** | **Learned** | **Yes** | **Yes** | **Student** | **Full method** |
| Privileged Teacher | M8 | Optional | Oracle | Optional | Teacher (privileged) | Upper bound |
