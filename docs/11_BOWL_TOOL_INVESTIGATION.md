# Bowl-Shaped Tool Investigation

> **Date:** 2026-04-08
> **Purpose:** Evaluate whether a bowl/ladle-shaped tool could enable scooping in Genesis MPM

---

## 1. Problem Statement

Current flat scoop (`panda_scoop.xml`) has 3 walls + open front. During horizontal traverse, ALL particles fall out because:
- MPM has no particle-rigid adhesion (only friction coupling)
- Open front provides no containment during lateral motion
- Scooping task was abandoned: 0% transfer for all materials

**Proposed fix:** Close the front to make a bowl/cup → particles physically trapped inside.

---

## 2. Technical Feasibility: CONFIRMED

### Current scoop MJCF already uses multi-box geometry

File: `/home/zhuzihou/dev/Genesis/genesis/assets/xml/franka_emika_panda/panda_scoop.xml`

```xml
<body name="scoop" pos="0 0 0.107" quat="0.3826834 0 0 0.9238795">
  <geom name="scoop_bottom" type="box" pos="0 0.04 0.03" size="0.05 0.04 0.008" />
  <geom name="scoop_back"   type="box" pos="0 0.005 0.04" size="0.05 0.008 0.03" />
  <geom name="scoop_left"   type="box" pos="-0.042 0.04 0.04" size="0.008 0.04 0.03" />
  <geom name="scoop_right"  type="box" pos="0.042 0.04 0.04" size="0.008 0.04 0.03" />
</body>
```

Interior dimensions: ~8cm wide × 8cm deep × 5cm tall. Angled bottom (0.2 rad / 11.5°).

### Modification: Add ONE front wall

```xml
<geom name="scoop_front" type="box"
      pos="0 0.075 0.04"
      size="0.05 0.008 0.03"
      material="scoop_metal"
      group="3"/>
```

This closes the open front → creates a rectangular bowl. **~5 lines of XML, no code changes needed.**

Genesis handles this natively: all `<geom>` elements within the `<body>` share the same rigid body and are all coupled to MPM via `needs_coup=True, coup_friction=3.0`.

---

## 3. Physics Analysis

### Why a bowl changes the game

With a closed container:
- **Lift:** Particles can't fall out the bottom (it's solid) or sides (walls). They CAN slosh out the **top** if accelerated too fast.
- **Traverse:** The critical variable becomes **speed**. Inertial force during acceleration pushes particles against the trailing wall. If the force exceeds the angle of repose limit, particles overflow the top.

### Material-dependent traverse speed

The maximum safe traverse acceleration before overflow:

```
a_max = g × tan(φ) × (wall_height / bowl_depth)
```

Where φ = angle of repose:

| Material | φ (angle of repose) | Density (kg/m³) | Max safe speed* | Behavior in bowl |
|----------|--------------------|-----------------|-----------------| --- |
| Sand | ~30° | 1400-1800 | ~0.5 m/s | Granular, fills gaps, overflows if tilted >30° |
| Snow | ~35° | 200-600 | ~0.4 m/s | Some cohesion, slightly more stable |
| ElastoPlastic | ~40° | 1000-1500 | ~0.3 m/s | Cohesive blob, deforms but doesn't scatter |
| Liquid | ~0° | 1000 | Very slow / fails | No angle of repose, sloshes freely |

*At 5cm wall height, 8cm bowl depth

**Key insight:** Each material has a different optimal traverse speed. This is exactly the kind of material-dependent strategy that Probe-Then-Act is designed to exploit.

### Training story with bowl

1. **Probe phase:** Tap/press into material → infer stiffness/friction → estimate angle of repose
2. **Scoop phase:** Insert bowl (tilt forward), push into material, tilt back to capture
   - Insertion depth depends on material: sand = easy, elastoplastic = needs more force
3. **Lift phase:** Raise bowl. Particles contained by walls + bottom. Minor spill from top if too fast.
4. **Traverse phase:** Move to target. **Speed = key decision variable.**
   - Too fast → overflow → spill
   - Too slow → episode timeout
   - Optimal speed is material-dependent
5. **Pour phase:** Tilt bowl to deposit into target.

### Comparison with edge-push

| Aspect | Edge-push | Bowl scooping |
|--------|-----------|---------------|
| Transfer mechanism | Push off cliff, gravity drops to target | Scoop, carry, pour |
| Material discrimination | 55pp gap (Sand 32%, Snow 87%) | Expected: insertion depth + speed + angle all vary |
| Validated? | YES (Config D) | NO (needs experiment) |
| Implementation effort | Done | 2-3 hours |
| Paper narrative | Simple geometric task | Matches "tool use" title better |
| Risk | Low (proven) | Medium (physics unknown) |
| Can serve as Task B? | Already Task A | Natural complement |

---

## 4. Implementation Path

### Minimum viable test (~2 hours)

1. **Copy and modify** `panda_scoop.xml` → `panda_bowl.xml` (add front wall, possibly raise all walls)
2. **Add** `tool_type="bowl"` option to `scene_builder.py`
3. **Write** scripted bowl trajectory (tilt-insert-tilt-lift-traverse-pour)
4. **Test** on sand at 3 traverse speeds (0.2, 0.5, 1.0 m/s)
5. **Measure** transfer efficiency and spill ratio

### Files to modify
- `assets/` or Genesis assets: Create `panda_bowl.xml`
- `pta/envs/builders/scene_builder.py`: Add `tool_type="bowl"` path
- `pta/scripts/run_scripted_baseline.py`: Add bowl-specific waypoint sequence

---

## 5. Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| MPM particles leak through box gaps | Medium | Experiment fails | Increase wall thickness, raise `mpm_grid_density` |
| Bowl too heavy for Franka PD control | Low | Robot can't lift | Use `set_qpos()` (already validated) |
| No material discrimination in bowl task | Low | Wasted 2-3 hours | Edge-push is already validated as backup |
| Particles don't enter bowl during insertion | Medium | Can't scoop | Adjust insertion angle, increase bowl opening |

---

## 6. Recommendation

**Worth a quick experiment (2-3 hours) as a parallel investigation.** Should NOT block the edge-push main line.

If successful, it becomes **Task B** in the paper:
- Task A: Edge-push (material-adaptive push force/speed)
- Task B: Bowl scooping (material-adaptive traverse speed)

Two tasks strengthen the paper's generalizability claim: "Our method works across different manipulation tasks, not just one specific geometry."

---

## 7. Decision Status

- [ ] Quick experiment (modify MJCF, test on sand)
- [ ] If works: integrate as Task B
- [ ] If fails: document as negative result, focus on edge-push only
