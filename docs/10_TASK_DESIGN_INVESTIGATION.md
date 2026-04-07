# Gate 0.5: Task Design Investigation

> **Date:** 2026-04-07
> **Purpose:** Determine whether the edge-push task has sufficient material discriminability for the paper.

---

## 1. Why This Investigation Was Needed

After the 48-hour sprint fixed the control stack, we discovered:
- The scripted baseline only achieves 12.6% transfer on sand (below 30% Gate 4 threshold)
- The earlier 42.2% (Gate 0) was from a different particle position (y=0.03 → y=-0.03 bugfix)
- No multi-material edge-push data existed — unclear if materials discriminate at all

**Core concern:** If a fixed push achieves ~the same transfer on all materials, then "probe-then-act" has no advantage over reactive control — the paper's hypothesis is untestable.

---

## 2. Material Sweep (Original Config, particle y=-0.03)

Same fixed scripted trajectory (Sequence E) on all materials, 5 episodes each:

| Material | Transfer % | Spill % | Observation |
|----------|-----------|---------|-------------|
| Snow | 22.3 ± 0.03 | 8.3 | Lightest (ρ=200-600), easiest to push |
| Liquid | 13.8 ± 0.05 | 9.5 | MPM liquid, not real water — has viscosity |
| Sand | 12.6 ± 0.05 | 9.8 | Mid-range friction/density |
| ElastoPlastic | 0.0 ± 0.00 | 5.8 | Too stiff/cohesive, push deforms but doesn't displace |

**Gap:** 22.3pp (Snow - ElastoPlastic). Materials DO discriminate, but nothing reaches 30%.

---

## 3. Scooping Feasibility Test

Tested scoop-lift-traverse-deposit (flat containers, same level) with 4 variants × 3 materials:

**Result: 0% transfer for ALL materials and ALL variants.**

MPM particles have no adhesion force to rigid bodies — only contact friction. During horizontal traverse, gravity pulls all particles off the scoop.

**Phase-wise retention (useful insight):**

| Material | Captured at LIFT_LOW | Retained at LIFT_FULL | Survived TRAVERSE |
|----------|---------------------|-----------------------|-------------------|
| Sand | 572/2400 (24%) | 0 | 0 |
| Snow | 613/2400 (26%) | 39 (1.6%) | 0 |
| ElastoPlastic | 13/2454 (0.5%) | 1114 (45%!) | 0 |

**Key insight:** ElastoPlastic barely gets captured (low insertion) but has extreme retention once lifted (blob behavior). This is the opposite of Sand. The capture/retention signature is **highly material-discriminative** — useful as a probe signal even though transfer fails.

**Conclusion:** Scooping is dead for material transfer in Genesis MPM. But scoop-insertion as a **probe action** (tap, press, partial insert) can reveal material properties.

---

## 4. Geometry Sweep (Config A-E)

Tested 5 geometric configurations × 3 materials × 3 episodes:

| Config | Change | Sand | Snow | EP | Gap |
|--------|--------|------|------|-----|-----|
| A (baseline) | particle y=-0.03 | 12.6% | 22.3% | 0.0% | 22.3pp |
| B (target closer) | target y: 0.15→0.10 | 9.4% | 20.0% | 0.0% | 20.0pp |
| C (lower platform) | platform z: 0.15→0.10 | 0.0% | 0.0% | 0.0% | 0pp |
| **D (particles closer)** | **particle y: -0.03→0.02** | **32.2%** | **87.3%** | **69.6%** | **55.1pp** |
| E (D + B combined) | particle y=0.02, target y=0.10 | 27.4% | 79.7% | 24.9% | 54.8pp |

**Winner: Config D.** One parameter change (`particle_pos y: -0.03 → 0.02`) solves everything:
- All 3 materials above 30%
- 55pp material gap
- Unexpected ordering: Snow > ElastoPlastic > Sand

Config B (target closer) slightly hurts — scattered sand particles miss the closer target.
Config C (lower platform) catastrophically fails — platform too low for scoop to reach.
Config E (D+B) is slightly worse than D alone.

---

## 5. Understanding Config D: Why y=0.02 Works

### Geometry

```
Platform: y ∈ [-0.075, +0.075], open edge at y = +0.075
Particles (Config D): center at y = 0.02, extends y ∈ [-0.01, +0.05]
Distance from particle front to open edge: 0.075 - 0.05 = 0.025m (2.5cm)
Distance from particle center to open edge: 0.075 - 0.02 = 0.055m (5.5cm)
```

At y=-0.03 (old config), particles were 10.5cm from edge — the 3-pass push couldn't move enough material that far. At y=0.02, particles are 5.5cm from edge — much more reachable.

### Material ranking explanation

- **Snow (87.3%):** Low density (ρ~300), low friction. Scoop pushes snow easily, particles fly off edge with momentum. Snow is "slippery" — almost all particles get pushed.
- **ElastoPlastic (69.6%):** Deforms as a cohesive blob. The scoop compresses and displaces the entire mass, which slides off the edge as a unit. Moderate transfer because the blob has internal resistance.
- **Sand (32.2%):** Granular, mid-friction (μ~0.5), mid-density (ρ~1600). Scoop pushes surface particles but deeper layers resist via friction. Only top layers transfer. **This is the hardest material for the push strategy.**

---

## 6. Is Config D Still "Gravity Does All the Work"?

### The concern
If particles are 2.5cm from the edge, do they just fall off on their own without the robot doing anything?

### Evidence needed
A no-op baseline test: run Config D with zero robot action and measure how many particles auto-fall.

### Physics reasoning
- Particles start at z=0.20, platform surface at z=0.15
- After settling (~50 steps), particles rest on the platform
- Particle front edge is 2.5cm from the open edge
- Sand particles have friction μ~0.5 with the platform (coup_friction=0.5)
- With no external force, settled particles should NOT slide off the edge — friction + flat surface = static equilibrium

**However, this must be verified empirically.** The no-op baseline (Sequence D in the scripted baseline script) measures exactly this.

### What would invalidate Config D
If the no-op baseline on sand shows >5% transfer → particles auto-fall → task is gravity-trivial → Config D is invalid.
If no-op shows <5% transfer → robot action is necessary → Config D is valid.

**This test must be run before proceeding.**

### Empirical verification (2026-04-07)

No-op baseline (Sequence D: zero robot action, 200 steps settle) on Config D:

| Material | Transfer % | Spill % |
|----------|-----------|---------|
| Sand | **0.0%** | 0.0% |
| Snow | **0.0%** | 0.0% |
| ElastoPlastic | **0.0%** | 0.0% |

Random action baseline (Sequence C: uniform random 7D actions) on sand:
- Transfer: **0.0%**, Spill: 0.0%

**Config D is validated.** Particles do NOT auto-fall. Zero transfer without intentional pushing. The robot's action is necessary and sufficient — this is NOT a gravity-trivial task.

The 32-87% transfer from the scripted push is entirely due to the robot's pushing strategy, not passive gravity. Different materials respond differently to the same push → material-adaptive control is meaningful.

---

## 7. Recommendations

1. **Run no-op baseline on Config D** for sand, snow, elastoplastic — verify robot is necessary
2. If valid: **Config D is the new default** (already applied to scene_builder.py)
3. **Train on sand** (hardest: 32%) as the training distribution
4. **Evaluate on snow/EP** (OOD: 87%/69% with scripted) — expect reactive policy to underperform teacher
5. Use **scoop-insertion as probe signal** — the capture-retention pattern is material-discriminative

---

## 8. Files Produced

| File | Content |
|---|---|
| `results/edge_push_material_sweep/sand.csv` | Sand baseline (5 eps) |
| `results/edge_push_material_sweep/snow.csv` | Snow baseline (5 eps) |
| `results/edge_push_material_sweep/elastoplastic.csv` | EP baseline (5 eps) |
| `results/edge_push_material_sweep/liquid.csv` | Liquid baseline (5 eps) |
| `results/edge_push_material_sweep/geometry_sweep.csv` | Config A-E × 3 materials |
| `results/scoop_material_sweep/scoop_sweep.csv` | Scoop variants × 3 materials |
| `results/scoop_material_sweep/scoop_snapshots.csv` | Per-phase retention data |
| `pta/scripts/edge_push_geometry_sweep.py` | Geometry sweep script |
| `pta/scripts/scoop_material_sweep.py` | Scoop sweep script |
