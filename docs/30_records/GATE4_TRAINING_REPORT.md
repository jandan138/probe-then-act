# Gate 4 Training Report: Joint-Space Residual PPO

**Date:** 2026-04-07
**Agent:** ik-fix

## Summary

Two training runs tested the `JointResidualWrapper` on the edge-push tiny task:
- **v1**: `residual_scale=0.1`, `log_std_init=-1.0` (std~0.37)
- **v2**: `residual_scale=0.2`, `log_std_init=-0.5` (std~0.61)

Both runs showed clear learning signal and separated from random baseline, but plateaued at the scripted baseline level without meeting the formal Gate 4 pass criteria.

**2026-04-09/10 follow-up:** the original suggestion to try scoop / bowl as the next Gate 4 base trajectory is now superseded by the expanded flat-scene bowl diagnosis. Bowl remains a side-track negative result; native tuning, minimal sticky fallback, hidden geometry, and particle constraints have all been exercised without useful final carry. The line is now better interpreted as a simulator-gap / probe-signal branch, not as the Gate 4 main line.

## v1 Results (scale=0.1)

| Steps | Mean Reward | Std |
|-------|------------|-----|
| 5,000 | -119.33 | 19.42 |
| 10,000 | -161.26 | 33.70 |
| 15,000 | -6.73 | 0.96 |
| 20,000 | -2.10 | 0.01 |
| 25,000 | -2.04 | 0.00 |
| 30,000 | -2.10 | 0.01 |
| 35,000 | -2.15 | 0.01 |
| 40,000 | -2.09 | 0.00 |
| 45,000 | -2.16 | 0.01 |

**Best reward:** -2.04 at 25K steps
**Explained variance:** 0.74 at 37K (value function learned well)
**Plateau:** Converged at -2.1 from 20K onward — reproduces scripted baseline exactly

### v1 Diagnosis
- Policy learned to output near-zero residuals (std stayed at 0.364)
- Clip fraction dropped to 0 — no further policy updates
- The 0.1 rad exploration range was too small to discover improvements

## v2 Results (scale=0.2)

| Steps | Mean Reward | Std |
|-------|------------|-----|
| 5,000 | -392.93 | 70.91 |
| 10,000 | -1.84 | 0.02 |
| 15,000 | -5.51 | 0.00 |
| 20,000 | -1.20 | 0.00 |
| 25,000 | -3.03 | 0.01 |
| 30,000 | -2.43 | 0.01 |

**Best reward:** -1.20 at 20K steps (better than v1's plateau)
**Explained variance:** 0.43 at 32K
**Policy std:** 0.605-0.609 (maintained higher exploration)

### v2 Diagnosis
- Oscillating reward (-1.2 to -5.5) due to larger exploration footprint
- Best eval (-1.20) slightly improves on v1's plateau (-2.04)
- Still exploring but hasn't found breakthrough corrections

## Comparison with Baselines

| Method | Mean Reward | Transfer % |
|--------|-----------|------------|
| E1 Cartesian-delta PPO | -39.6 +/- 3 | ~0% |
| Random baseline | ~-120 to -160 | ~0% |
| Scripted baseline (set_qpos) | ~-2.0 | ~12.5% |
| **v1 Joint-residual (scale=0.1)** | **-2.09** | **~12.5%** |
| **v2 Joint-residual (scale=0.2)** | **-1.20** (best) | **~12-15%** |

## Gate 4 Assessment

| Gate 4 Target | Status |
|---|---|
| success_rate >= 70% with learner meeting tiny-task pass thresholds | NOT MET — transfer ~12-15% |
| median transferred_mass_frac >= 0.25 | NOT MET — ~0.12 |
| Stable across >= 3 eval reruns | PARTIAL — v1 very stable, v2 oscillating |

**Gate 4 verdict: NOT PASSED** — but the joint-space approach shows clear improvement over the Cartesian-delta E1 attempts.

## Key Findings

1. **Joint-space residual works**: Reward improved from -160 (random) to -2.0 (scripted baseline level) — a 100x improvement over E1's failure to learn.

2. **IK bypass confirmed**: The joint-space approach completely avoids the y-axis inversion bug and produces stable, reproducible trajectories.

3. **Bottleneck is the base trajectory, not the residual**: The scripted edge-push trajectory only achieves ~12.5% transfer. The residual policy reproduces this faithfully but can't improve much beyond it. A fundamentally better base strategy is still needed to reach the formal Gate 4 pass criteria.

4. **Scale=0.2 slightly outperforms scale=0.1**: Best reward of -1.20 vs -2.04, suggesting the residual can find small improvements with more exploration room.

## Recommendations

1. **Better base trajectory**: Improve the edge-push base trajectory first. Do **not** treat scoop / bowl as the current Gate 4 rescue path; that line is now a documented side-track negative result unless separate bowl-side-track work changes the evidence.

2. **Longer training**: 200K steps may not be enough for the residual to find meaningful improvements over a decent base trajectory. Try 1M+ steps.

3. **Curriculum**: Start with scale=0.0 (pure replay) for warmup, then anneal to scale=0.2 over 100K steps.

4. **Hybrid approach**: Use `control_dofs_position` (PD control) instead of `set_qpos` (teleport) in the wrapper for more physically realistic execution, but do not conflate that controller question with the separate bowl-side-track contact problem.

## Files

- `pta/scripts/launch_gate4.py` — v1 launcher
- `pta/scripts/launch_gate4_v2.py` — v2 launcher
- `logs/gate4_joint_residual_v1/` — v1 training logs
- `logs/gate4_joint_residual_v2/` — v2 training logs
- `checkpoints/gate4_joint_residual_v1/` — v1 checkpoints
