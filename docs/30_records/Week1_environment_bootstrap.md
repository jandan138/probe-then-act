# Week 1 -- Environment Bootstrap

**Date**: 2026-04-03 ~ 2026-04-04
**Owner**: Environment Lead (AI-assisted)

## Objective

Build minimal tasks and verify simulation stability.

## What Was Completed

### 1. Genesis Environment Setup
- Python 3.11.14 with PyTorch 2.11.0+cu126 on RTX 4090 (WSL2)
- Genesis imported successfully via `PYOPENGL_PLATFORM=osmesa`
- MPM smoke test: 8182 particles, 50 steps, 0 NaN

### 2. Scoop-and-Transfer Environment
- Full implementation following GraspEnv pattern from Genesis examples
- Scene: Franka Panda + source container + target container + MPM particles
- Rigid-MPM coupling enabled on all rigid entities (`needs_coup=True`)
- Action space: 7D (delta EE pose + gripper), DLS IK
- Observation space: 37D (joint pos/vel + EE pose + finger pos + step fraction)
- Reward: transfer_mass - spill_penalty - time_penalty + success_bonus
- Metrics: success_rate, transfer_efficiency, spill_ratio

Scene geometry:
- Source container at (0.5, 0.0, 0.05), 0.15 x 0.15, wall height 0.08 m
- Target container at (0.5, 0.35, 0.05), 0.12 x 0.12, wall height 0.10 m
- 400 sand particles initialised at (0.5, 0.0, 0.10) as a 0.10x0.10x0.04 block
- Robot at origin, EE home position: (0.306, 0.0, 0.587)

### 3. Sanity Check Results
```
ScoopTransfer Environment Sanity Check
  Steps:    100
  Material: sand
  Total particles: 400

  PASS: Environment is stable -- no NaN/Inf detected

  Reward: mean=-0.4122  std=0.1757  min=-0.5010  max=-0.0010
```

### 4. Code Scaffold
- 126 Python files + 22 YAML configs created
- Full project structure per `docs/00_foundation/01_REPO_BLUEPRINT.md`
- All modules have real type signatures with NotImplementedError bodies

### 5. PPO Training Verification
- SB3 PPO successfully created and ran 32 training steps
- Obs=37D, Act=7D confirmed compatible

### 6. Literature & Novelty Check
- No direct competitor found (2024-2026)
- Closest: SCONE (CoRL 2023), AdaptiGraph (RSS 2024), ASID (ICLR 2024)
- Full report: `docs/50_reports/06_NOVELTY_CHECK_REPORT.md`

### 7. Scripted Baselines

Four scripted baselines were implemented (`pta/scripts/run_scripted_baseline.py`)
and evaluated (5 episodes each, 200-step horizon, sand material):

| Sequence | Method | Transfer Eff. | Spill Ratio |
|----------|--------|---------------|-------------|
| A_scoop_deposit | Joint-waypoint sweep source-to-target | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| B_probe_scoop | 3 probe taps + sweep | 0.0010 +/- 0.0012 | 0.9985 +/- 0.0012 |
| C_random | Uniform random actions | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |
| D_noop | Zero actions (settling baseline) | 0.0000 +/- 0.0000 | 1.0000 +/- 0.0000 |

Full per-episode results: `results/tables/scripted_baselines.csv`

Key observations:
- Sequence B (probe+scoop) achieved non-zero transfer (1/400 particles)
  in 2 out of 5 episodes -- the only baseline to reach the target at all.
- All sequences show spill_ratio near 1.0 due to the particle-settling
  AABB issue (see Issue 4 below).
- The D_noop baseline confirms that spill_ratio=1.0 is the natural
  settling equilibrium, not caused by robot actions.

Sequences A and B use direct joint-position waypoints via `set_qpos` with
fine interpolation because the default IK action scale (0.01 m/step) is
too conservative for the gripper to reach the particle layer within 200
steps.  The EE starts at (0.306, 0.0, 0.587) and needs to reach
(0.5, 0.0, ~0.08) -- a distance of ~0.55 m requiring ~550 steps at full
IK effort.

## Evidence

| Artifact | Path |
|----------|------|
| Sanity check script | `pta/scripts/sanity_check_env.py` |
| Scripted baselines script | `pta/scripts/run_scripted_baseline.py` |
| Baseline results CSV | `results/tables/scripted_baselines.csv` |
| Environment implementation | `pta/envs/tasks/scoop_transfer.py` |
| Scene builder | `pta/envs/builders/scene_builder.py` |
| Config files | `pta/configs/env/scoop_transfer/*.yaml` |
| Novelty report | `docs/50_reports/06_NOVELTY_CHECK_REPORT.md` |
| Experiment plan | `refine-logs/EXPERIMENT_PLAN.md` |

## Exit Criteria Checklist

- [x] Environment resets without NaNs -- PASSED (100 consecutive steps)
- [x] At least 100 consecutive steps run stably -- PASSED
- [~] Task metrics change in sensible directions -- PARTIAL (see Issue 4)
  - transfer_efficiency correctly increases (0 -> 0.0025) when a particle
    enters the target AABB
  - spill_ratio saturates at 1.0 for all baselines due to particle settling
    below the source AABB z-threshold -- not yet discriminative
- [x] One Franka scene loads -- PASSED
- [x] One scoop tool loads -- PASSED (gripper as tool for v1)
- [x] One source + target container load -- PASSED
- [x] One MPM material runs stably -- PASSED (Sand, 400 particles)
- [ ] One debug rollout video -- PENDING (headless mode, need to capture frames)
- [x] Initial metrics defined -- DONE (success_rate, transfer_efficiency, spill_ratio)

## Issues Encountered

1. **WSL2 OpenGL**: Resolved with `PYOPENGL_PLATFORM=osmesa`
2. **libcuda.so path**: Resolved with `LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH`
3. **PyTorch version**: Upgraded from 2.6.0 to 2.11.0 for Genesis compatibility
4. **Particle settling below source AABB** (OPEN, HIGH PRIORITY):
   The source container base plate (0.005 m thick) does not fully contain
   MPM particles. Particles settle through/around it to rest on the ground
   at z~0.01-0.03, below the source AABB z-minimum of 0.04. This makes
   spill_ratio converge to 1.0 within ~50 steps regardless of robot
   actions. Fix options: (a) lower AABB z-min to 0.0 or (b) thicken the
   base plate.
5. **Reachability gap with default action scale** (OPEN, HIGH PRIORITY):
   The DLS IK + PD controller combination is too conservative for the EE
   to traverse the 0.55 m from home to the source container within the
   200-step horizon. Additionally, the PD controller cannot converge to
   low-z arm configurations due to gravity compensation limitations at
   full extension. Fix options: (a) increase action_scale_pos to 0.05+,
   (b) add a pre-approach phase, or (c) use absolute EE targets.
6. **substep_dt warning**: `0.0005 > suggested 0.0003125` -- simulation
   runs but may need tuning for stability.

## Next Steps (Week 2)

1. **Fix source AABB z-range** to capture settled particles (Issue 4).
   This is the highest priority since it blocks meaningful spill_ratio
   measurement.
2. **Fix action scale / reachability** (Issue 5) to make the
   scoop-and-transfer task solvable within the 200-step horizon via the
   standard action interface.
3. **Implement metric overlay on debug video** -- render particle
   positions with source/target AABB boundaries visible.
4. **Validate metrics against visual inspection** -- export video of
   Sequence B episodes where transfer_efficiency > 0 and verify the
   metric counts agree with visible particle positions.
5. **Re-run scripted baselines** after Issues 4 and 5 are fixed to
   establish meaningful baseline numbers.
6. **Train M1 (Reactive PPO) and M2 (RNN-PPO) baselines** once the
   action scale is fixed.
7. **Define OOD splits formally**.
