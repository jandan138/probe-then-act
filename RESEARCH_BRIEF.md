# Research Brief

> Document-based input for ARIS `/research-lit` and `/novelty-check`.

## Problem Statement

Robotic tool use on deformable and flowable materials is challenging because key physical properties of the manipulated medium are often hidden and cannot be inferred reliably from a single observation. Policies trained reactively on instantaneous observations fail under cross-material variation, leading to spillage, unstable contact, and poor out-of-distribution generalization. Current approaches either ignore hidden physics entirely (reactive policies), rely on passive memory that may not capture the right physical cues (RNN-based policies), or brute-force the problem with domain randomization which trades robustness for peak performance.

The missing piece is an **active information gathering** mechanism: a robot should first perform short diagnostic probing actions to estimate the hidden material properties, then condition its manipulation policy on the inferred physical belief and its uncertainty.

## Background
- **Field**: Robot Learning, Reinforcement Learning
- **Sub-area**: Manipulation under hidden physical properties, active system identification, tactile sensing, deformable object manipulation
- **Key papers I've read**:
  - Tactile exploration for material identification (Fishel & Loeb, ICRA 2012)
  - Learning to manipulate deformable objects with model-based RL (Shi et al., CoRL 2022)
  - DiffTactile and TacSL (tactile simulation frameworks)
  - Genesis simulator documentation (MPM, SPH, coupled solvers)
  - Teacher-student learning for manipulation (NVIDIA IsaacGym examples)
  - Domain randomization for sim-to-real (Tobin et al., IROS 2017)
  - POMDP-based manipulation with belief tracking (Kaelbling & Lozano-Pérez)
- **What I already tried**: Project design phase complete; edge-push environment, wrappers, and baseline/runtime paths are implemented; training is in progress
- **What didn't work**: Scoop-and-transfer as the mainline transfer task; Gate 4 still not passed on edge-push

## Constraints
- **Compute**: 1x RTX 4090 (24GB VRAM) local, Vast.ai available for scaling
- **Timeline**: 27 days to submission (deadline April 30, 2026)
- **Target venue**: IEEE Transactions on Robot Learning (T-RL), with IROS 2026 / CASE 2026 conference transfer window

## What I'm Looking For
- [x] Novelty verification: confirm no 2025-2026 concurrent work blocks our contribution
- [x] Literature survey: active probing + belief-conditioned manipulation + deformable materials
- [ ] ~~New research direction from scratch~~
- [ ] ~~Improvement on existing method~~

## Domain Knowledge
- **Core hypothesis**: Active probing + latent physical belief inference + uncertainty-aware control improves robust cross-material robot tool use more than reactive RL, recurrent history-based RL, domain randomization alone, or discrete material classification + policy routing.
- **Method name**: Probe-Then-Act
- **Pipeline**: (1) Probe Policy → (2) Physical Belief Encoder → (3) Task Policy conditioned on belief + uncertainty → (4) Risk/Safety Head
- **Simulator**: Genesis with MPM materials (Sand, Snow, ElastoPlastic, Liquid) + Franka Panda
- **Tasks**: Scoop-and-Transfer (primary), Level-and-Fill (secondary)
- **Training**: Two-stage teacher-student: privileged RL teacher → student distillation + RL fine-tuning
- **7 baselines**: Reactive PPO, RNN-PPO, Domain Randomization PPO, Fixed-Probe+PPO, Material Classification+Expert Routing, Ours w/o Uncertainty, Privileged Teacher Upper Bound
- **5 OOD axes**: OOD-Parameter, OOD-Material Family, OOD-Tool Geometry, OOD-Container Geometry, OOD-Sensor Perturbation

## Non-Goals
- Sim-to-real transfer (simulation only, no real hardware)
- Genesis showcase paper
- Pure application demo without methodological depth
- Foundation model for materials

## Existing Results
Current state is no longer pre-implementation. The edge-push mainline has active engineering results, but the scientific program is still only partially validated:
- Gate 0 passed for edge-push scripted feasibility
- Gate 2/3 passed for implementation and smoke testing
- Gate 4 remains partial: learning pipeline improved, but tiny-task pass criteria are still unmet
- M7 core runtime path is implemented, but no validated M7 claim should be made yet

Canonical design and protocol documents:
- `docs/00_foundation/00_PROJECT_BRIEF.md` — Full research vision and method design
- `docs/00_foundation/01_REPO_BLUEPRINT.md` — Repository architecture (~100 files planned)
- `docs/10_protocols/02_EXECUTION_PLAYBOOK.md` — 8-week execution plan with exit criteria
- `docs/10_protocols/03_EXPERIMENT_PROTOCOL.md` — 4 hypotheses, 8 methods, experiment matrix, metrics
