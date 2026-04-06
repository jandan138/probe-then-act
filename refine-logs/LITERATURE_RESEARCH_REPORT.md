# Literature Research Report: Solving the Scooping Training Bottleneck

> Multi-angle parallel research, 2026-04-06. Five agents investigated reward design, demo-guided RL, action spaces, Genesis MPM feasibility, and PPO tuning.

## Executive Summary

**Root cause of 0% success is NOT just reward design or PPO tuning — it's physical infeasibility.**

A Franka Panda parallel-jaw gripper **cannot physically scoop granular material**. The jaw geometry is flat, designed for grasping discrete objects, not bulk material. Every successful scooping paper in the literature uses a **custom scoop end-effector**. This single issue explains why no amount of reward shaping or hyperparameter tuning produces success.

Additionally, five compounding problems were identified:

| # | Problem | Severity | Fix Difficulty |
|---|---------|----------|----------------|
| 1 | **Gripper can't scoop** — need custom scoop tool | CRITICAL | Medium (new URDF) |
| 2 | **7-D action space at 500Hz** — needle-in-haystack | HIGH | Easy (reduce dims + action repeat) |
| 3 | **ent_coef=0.01** — entropy divergence (std 1→2.1) | HIGH | Easy (set to 0.0) |
| 4 | **Approach reward dominates scoop by 10x** | HIGH | Easy (rebalance magnitudes) |
| 5 | **No demos / curriculum** — pure exploration fails | MEDIUM | Medium (scripted demos) |

## Detailed Findings

### 1. Physical Feasibility (genesis-researcher)

- **No published paper** uses Genesis MPM for RL-based scooping
- **No Genesis official example** of robot + MPM scooping exists
- **GranularGym (RSS 2023)** is the closest reference: Franka + **custom scoop attachment** + granular bed RL
- **DiffSkill (ICLR 2022)** uses custom tools (spatula, pusher) for deformable manipulation
- Parallel-jaw gripper is physically implausible for bulk material scooping
- Genesis MPM-rigid coupling is marked "experimental" (IPC coupler WIP)

**Action required**: Design and integrate a scoop end-effector URDF/MJCF before any further RL training.

### 2. Action Space (action-space-researcher)

Current 7-D delta EE at 500Hz is the worst possible choice for scooping:
- 7 dimensions × 200 steps = 1400 decisions per episode
- Random exploration probability of hitting correct scoop trajectory ≈ 0
- Orientation (roll/pitch/yaw) is a strong geometric prior for scooping — should be fixed per phase

**Recommended changes**:
- Reduce to **3-D position only** `(dx, dy, dz)` with phase-dependent fixed orientation
- Add **action repeat** (policy at 20Hz, physics at 500Hz) → ~40 meaningful decisions per episode
- Or use **ProDMP** to output trajectory parameters instead of per-step actions

### 3. PPO Hyperparameters (ppo-researcher)

- **ent_coef=0.01 is 10x too high** — SB3 default is 0.0 for continuous control
- With 7 action dims, entropy bonus is summed across dims → 7x effective pressure
- Action std grew from 1.0 to 2.1 = entropy bonus overpowering policy gradient
- **use_sde=True** (state-dependent exploration) is designed for exactly this type of task
- **log_std_init=-1.0** starts with std=0.37 instead of 1.0

**Recommended PPO config**:
```python
PPO("MlpPolicy", env,
    ent_coef=0.0,          # was 0.01
    use_sde=True,          # coherent exploration
    sde_sample_freq=4,
    policy_kwargs=dict(log_std_init=-1.0),
)
```

### 4. Reward Design (reward-researcher)

Current reward magnitude mismatch:
- Approach: -0.1 × dist → ~-0.5/step (always available, easy to optimize)
- Scoop: 0.3 × depth → ~0.05/step (narrow precondition, rarely triggered)
- **Approach is 10x easier to harvest than scoop**

Literature consensus:
- PlasticineLab, FluidLab, DiffSkill all found **RL struggles** with deformable manipulation
- Most use **differentiable optimization** for the manipulation primitive, not RL
- Excavation RL papers use **curriculum + large scoop rewards (10x approach)**

**Recommended rebalancing**:
```python
approach:  -0.01 * dist    # reduce to guidance-only (was -0.1)
scoop:      3.0 * depth    # 10x increase (was 0.3)
lift:       5.0 * frac     # 10x increase (was 0.5)
transfer:  10.0 * frac     # 10x increase (was 1.0)
success:   50.0            # large terminal bonus (was 5.0)
```

### 5. Demo-Guided RL (demo-researcher)

Pure PPO exploration fails for scooping — this is well-established in the literature. Options:

| Method | # Demos Needed | Complexity | Compatibility |
|--------|---------------|------------|---------------|
| **Residual Policy Learning** | 0 (scripted base) | Low | PPO ✓ |
| **DAPG (auxiliary BC loss)** | 10-25 | Medium | PPO ✓ |
| **BC + RL fine-tune** | 10-25 | Medium | PPO ✓ |
| **RLPD (demos in replay)** | 10-25 | Medium | SAC only |

**Top recommendation**: Residual policy learning — write a scripted scooping controller as `pi_base`, learn corrections `pi_residual` via RL. No demo collection/storage needed, continuous exploration guidance.

## Prioritized Action Plan

### Phase A: Physical Fix (MUST DO FIRST)
1. **Design scoop end-effector** — simple concave mesh, attach to Franka link 7
2. **Update URDF/MJCF** — add scoop geometry, mass, inertia
3. **Verify with scripted trajectory** — confirm particles can be scooped and transferred
4. **Adjust coup_friction** — may need increase from 0.1 for particles to stay in scoop

### Phase B: Training Infrastructure Fix
5. **Reduce action space to 3-D** — position only, phase-dependent orientation
6. **Add action repeat** — policy at 20Hz (action_repeat=25), physics at 500Hz
7. **Fix PPO config** — ent_coef=0.0, use_sde=True, log_std_init=-1.0
8. **Rebalance reward magnitudes** — scoop >> approach

### Phase C: Learning Strategy
9. **Write scripted scooping controller** — parametric waypoint trajectory for the scoop tool
10. **Implement residual policy learning** — pi = pi_base + pi_residual
11. **Train Teacher with residual policy** — should converge much faster
12. **If needed**: Fall back to DAPG with 10-25 scripted demos

### Phase D: Evaluation
13. **Re-evaluate M1 decision gate** — Teacher success_rate > 0?
14. **If pass**: Proceed to M2 (distillation) per EXPERIMENT_PLAN

## Estimated Timeline

| Phase | Work | GPU Time | Wall Time |
|-------|------|----------|-----------|
| A: Scoop tool | URDF + scripted verify | ~1h | 4-6h |
| B: Infra fix | Code changes | 0 | 2-3h |
| C: Residual RL | Train Teacher v3 | ~15h | ~15h |
| D: Eval | OOD evaluation | ~3h | ~3h |
| **Total** | | **~19h** | **~24-27h** |

## Key References

- **GranularGym** (RSS 2023) — Franka + scoop + granular RL (closest reference)
- **DiffSkill** (ICLR 2022) — Tool-use deformable manipulation with differentiable MPM
- **PlasticineLab** (NeurIPS 2021) — MPM benchmark; RL struggles, grad-opt wins
- **DAPG** (RSS 2018) — Demo-augmented policy gradient, 25 demos
- **DEMO3** (ICML 2025) — 5 demos sufficient for multi-stage tasks
- **Residual Policy Learning** (Silver 2018, Johannink ICRA 2019)
- **gSDE** — State-dependent exploration for PPO (built into SB3)
- **CoRL 2024 Action Space Study** — Control frequency is most impactful hyperparameter
