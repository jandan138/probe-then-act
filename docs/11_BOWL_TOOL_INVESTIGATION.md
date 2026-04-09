# Bowl-Shaped Tool Investigation

> **Date:** 2026-04-08
> **Last updated:** 2026-04-09
> **Purpose:** Evaluate whether a bowl/ladle-shaped tool could enable scooping in Genesis MPM

> **Execution runbook:** See [docs/12_BOWL_TRANSPORT_DIAGNOSIS_RUNBOOK.md](12_BOWL_TRANSPORT_DIAGNOSIS_RUNBOOK.md) for the flat-only diagnosis procedure and non-interference rules.

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

## 4. Historical Implementation Path

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

## 6. Current Recommendation

The quick bowl experiment has already been completed, and the flat-only diagnosis now dominates the decision.

Current recommendation:

- keep bowl as a **side-track only**, not the current Task B;
- if this line is continued, exhaust **Genesis-native contact-quality tuning first**;
- only if native tuning still fails, allow a clearly labeled **post-step sticky fallback** for engineering demonstration;
- do **not** let bowl work block the edge-push main line.

---

## 7. Decision Status

- [x] Quick experiment completed (MJCF + bowl trajectory + flat-only diagnosis)
- [ ] Integrate as Task B (not supported by current flat-scene results)
- [x] Current result documented as a negative finding; edge-push remains the main line

---

## 8. 2026-04-08 Flat-Only Diagnosis Update

### 8.1 Hidden scene mismatch was real

The early "~40.8% transfer" bowl smoke test was **not** directly comparable to the flat bowl sweep.
`run_scripted_baseline.py --tool-type bowl` inherited the SceneBuilder defaults, while the diagnosis script used an explicit flat scene.

| Path | task_layout | particle_pos | particle_size | total_particles |
|------|-------------|--------------|---------------|-----------------|
| Baseline bowl default | `edge_push` | `(0.55, 0.02, 0.20)` | `(0.12, 0.06, 0.03)` | `1728` |
| Flat bowl diagnosis | `flat` | `(0.50, 0.00, 0.12)` | `(0.10, 0.10, 0.03)` | `2400` |

This means the earlier bowl smoke result was effectively a **different physical task**. All diagnosis below uses the explicit `flat` bowl scene only.

Source: `results/scoop_debug/20260408T133215Z/config_sand/merged_scene_configs.json`

### 8.2 Three-stage retention diagnosis

We ran the bowl diagnosis in three stages:

1. `lift_only_hold` — lift to `lift_low`, then hold still
2. `lift_full_hold` — lift to `lift_full`, then hold still
3. `traverse_slow` — move to traverse midpoint, then hold

For **sand** and **snow**, we ran the full diagnosis (`3` repeats, `120` hold steps).  
For **elastoplastic**, we stopped at the smoke pass (`1` repeat, `20` hold steps) because the failure mode was already clear at the static stage.

| Material | Repeats | lift_only_hold end `n_on_tool` | lift_full_hold end `n_on_tool` | traverse_slow end `n_on_tool` | Diagnosis |
|----------|---------|--------------------------------|--------------------------------|-------------------------------|-----------|
| Sand | 3 | `156, 156, 156` | `159, 159, 159` | `0, 0, 0` | **Static retention works; transport fails** |
| Snow | 3 | `62, 61, 61` | `50, 50, 50` | `0, 0, 0` | **Static retention works; transport fails** |
| ElastoPlastic | 1 smoke | `0` | `0` | `0` | **Failure already happens before transport** |

Interpretation:

- **Sand:** the bowl can carry a meaningful number of particles while stationary, but loses everything by slow traverse midpoint.
- **Snow:** same qualitative pattern as sand, just with fewer particles retained at rest.
- **ElastoPlastic:** this is not mainly a traverse-speed problem. The bowl fails to capture or hold material even before transport.

### 8.3 Minimal transport scan

We then ran the smallest useful scan on the dynamic-failure branch: slow vs mid traverse speed, with and without extra back-tilt during transport.

#### Sand

| Combo | `lift_full_n_on_tool` | `mid_traverse_n_on_tool` | final transfer | dropped midway |
|-------|------------------------|--------------------------|----------------|----------------|
| `slow_base` | `165` | `0` | `0.0000` | yes |
| `mid_base` | `166` | `0` | `0.0000` | yes |
| `slow_backtilt` | `123` | `17` | `0.0017` | no |
| `mid_backtilt` | `123` | `14` | `0.0017` | no |

#### Snow

| Combo | `lift_full_n_on_tool` | `mid_traverse_n_on_tool` | final transfer | dropped midway |
|-------|------------------------|--------------------------|----------------|----------------|
| `slow_base` | `35` | `0` | `0.0000` | yes |
| `mid_base` | `35` | `0` | `0.0000` | yes |
| `slow_backtilt` | `83` | `3` | `0.0004` | no |
| `mid_backtilt` | `82` | `0` | `0.0000` | yes |

Interpretation:

- Extra back-tilt **does help midpoint retention a little** for sand and snow.
- But the help is far too small: final transfer remains effectively zero in the flat task.
- This is strong evidence that the next engineering pass for sand/snow should start with **native contact-quality tuning** and then, if native carry becomes nonzero, revisit **trajectory dynamics** (speed profile, acceleration, sustained back-tilt). It is **not** a reason to launch another blind material sweep.

### 8.4 Updated conclusion

Current status is **not** "bowl works but needs more training." The diagnosis says something more specific:

1. **Sand and snow are dynamic-transport failures.** The bowl can statically retain particles, but the current traverse phase loses them before delivery.
2. **ElastoPlastic is a capture / geometry / contact failure.** It does not reach a meaningful retained state even before traverse.
3. Under the explicit flat scene, **no tested setting produced useful final transfer**. The bowl path is therefore **not yet viable as Task B**.

### 8.5 Recommended next moves if this line is continued

The diagnosis now supports a stricter engineering order than the original quick experiment:

1. **Native Genesis contact-quality tuning first** for the sand / snow dynamic-failure branch.
2. **Guarded post-step sticky fallback second** only if native tuning still cannot keep useful final retention.
3. **Capture-first fixes for elastoplastic** remain separate; do not treat elastoplastic as the same problem as sand / snow transport.

For now, this remains a **parallel negative result** and should **not block the edge-push main line**.

---

## 9. 2026-04-09 Engineering-Priority Implementation Status

This section records both the current engineering order and what has already been wired into the repo, so the bowl side-track is not confused with either the mainline physics result or a stale pre-implementation plan.

### 9.1 Canonical status before any new work

- **Sand / snow:** static retention exists, but dynamic transport collapses during traverse.
- **ElastoPlastic:** failure happens earlier, at capture / contact / geometry time.
- Therefore, bowl is **not currently viable as Task B** under the explicit flat scene.

### 9.2 Phase 1 — Genesis-native contact-quality tuning

Goal: test whether the existing bowl can be stabilized **without** adding fake adhesion.

**Implementation status:** the repo now exposes the bowl-only gating path for this phase. The scene builder accepts bowl-only contact-quality flags, but all of them remain **default-off** and only activate when `tool_type="bowl"`, `task_layout="flat"`, and the explicit bowl enable flag is on. This was implemented specifically to avoid changing edge-push defaults.

Engineering order:

1. expose the missing Genesis knobs in the repo configuration path;
2. rerun the flat-only config dump + retention + minimal scan on **sand first**;
3. extend to snow only if sand shows a real signal;
4. keep elastoplastic out of this phase except for a cheap smoke check.

Parameters to expose and sweep first:

- `enable_CPIC` in `SceneBuilder._create_scene()`;
- robot-side `coup_friction` in `SceneBuilder._add_robot()`;
- `coup_softness` on the bowl rigid material;
- `substeps` in the scene config;
- `sdf_cell_size` and optional `sdf_min_res` / `sdf_max_res` on the bowl rigid material;
- keep `mpm_grid_density=128` as the baseline, and only raise it if leakage / wall-resolution evidence appears.

Primary success gate for this phase:

- **do not** judge by midpoint retention alone;
- require clearly nonzero `final_n_on_tool` and clearly nonzero final transfer in the explicit flat scene;
- if native tuning still ends with effective zero final carry, stop calling this a pure tuning problem.

Repo touch points for this phase:

- `pta/envs/builders/scene_builder.py` — expose CPIC and rigid-coupling knobs;
- `pta/scripts/bowl_transport_diagnosis.py` — keep the flat-only config dump + retention + scan as the evaluation harness;
- `pta/scripts/run_scripted_baseline.py` — keep trajectory changes secondary until contact-quality tuning is exhausted.

Implemented bowl-only knobs now include:

- `bowl_contact_quality_enabled`
- `bowl_enable_cpic`
- `bowl_substeps_override`
- `bowl_robot_coup_friction`
- `bowl_robot_coup_softness`
- `bowl_robot_sdf_cell_size`
- `bowl_robot_sdf_min_res`
- `bowl_robot_sdf_max_res`

### 9.3 Phase 2 — Post-step sticky fallback

Trigger: Phase 1 fails to produce useful final carry, but we still want a stable bowl-carry engineering demo.

Definition: a **local, non-physical retention fallback** that acts after each physics step during carry.

**Implementation status:** the first guarded version of this fallback is now wired into the repo. It remains **default-off**, and the runtime gate is intentionally narrow: it only becomes available for `tool_type="bowl"`, `task_layout="flat"`, and only during the explicit `carry` phase. This prevents accidental bleed-through into edge-push or other default paths.

Recommended first implementation:

1. add the fallback immediately after `scene.step()` in `pta/envs/tasks/scoop_transfer.py`;
2. detect particles that are inside or just above a bowl-local carry region;
3. during the carry phase only, project those particles back into the bowl volume and damp or zero the outward velocity component;
4. disable the fallback during capture and pour so the cheat is limited to transport stabilization.

Implemented bowl-only task flags now include:

- `bowl_sticky_fallback_enabled`
- `bowl_sticky_top_slack`
- `bowl_sticky_detection_margin`
- `bowl_sticky_velocity_damping`
- `bowl_sticky_zero_outward_velocity`
- `bowl_sticky_max_snap`
- `bowl_sticky_region_min`
- `bowl_sticky_region_max`

If the local post-step version is still too weak, the next stronger variants are:

- bowl-local hidden retention geometry in `panda_bowl.xml`;
- Genesis particle constraints or direct particle position / velocity setters;
- custom carry-phase force fields.

### 9.4 Reporting rule

If Phase 2 is used, document it as a **sticky fallback / retention fallback**, not as native Genesis bowl transport.

Do not mix native-tuning and sticky-fallback results into the same headline claim.

**Implementation status:** `pta/scripts/bowl_transport_diagnosis.py` now records both the **requested** config and the **effective runtime** config in its config report outputs. This was added because bowl-only gating means a requested override is not always the same thing as a setting that actually became active at runtime.

Current bowl diagnosis outputs now distinguish:

- `requested_scene_config`
- `effective_scene_runtime`
- `requested_task_config`
- `effective_task_runtime`

This makes it explicit, for example, when a non-flat bowl scene requested bowl tuning flags but those flags did **not** actually activate under the runtime gate.

### 9.5 Final decision gate

- **Native tuning succeeds:** bowl remains a simulator-native side-track and may be studied further.
- **Only sticky fallback succeeds:** bowl can be kept as an engineering demo, but not as physics-faithful evidence.
- **Neither succeeds:** freeze bowl as a negative result and keep all mainline effort on edge-push.
