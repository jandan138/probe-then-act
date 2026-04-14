# Bowl Transport Diagnosis Runbook

> **Date:** 2026-04-08  
> **Last updated:** 2026-04-09  
> **Owner:** Scoop / bowl side-track only  
> **Purpose:** Run flat-only bowl transport diagnosis without touching the edge-push main line.

---

## 1. Scope and non-interference rules

This runbook is for the **scoop / bowl diagnosis track only**.

Do **not**:
- modify edge-push scene defaults,
- stop or restart edge training processes,
- write outputs into edge training directories,
- run multiple heavy scoop jobs in parallel.

Do **only**:
- use `pta/scripts/bowl_transport_diagnosis.py`,
- use a fresh output root under `results/scoop_debug/<timestamp>/`,
- run one heavy simulation job at a time,
- keep all tests in the explicit `flat` bowl scene.

---

## 2. What this runbook is trying to answer

We are not chasing a high score first. We are answering this simpler question:

1. **Static retention question:** if the bowl is lifted and held still, do particles stay in the bowl?
2. **Dynamic transport question:** if static retention is OK, do particles fall out mainly during traverse?

Decision rule:
- **Lift holds keep particles, traverse loses them** → root cause is mainly **dynamic transport**.
- **Lift holds already lose particles fast** → root cause is mainly **geometry / contact / boundary setup**.

---

## 3. Environment setup

Run all commands from the repository root:

```bash
source /home/zhuzihou/dev/Genesis/.venv/bin/activate
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/home/zhuzihou/dev/probe-then-act:$PYTHONPATH
```

Create one isolated output root for this diagnosis pass:

```bash
export RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
export RUN_ROOT="results/scoop_debug/${RUN_TS}"
mkdir -p "$RUN_ROOT"
```

---

## 4. Pre-flight safety check

Record the current training state before starting any scoop job:

```bash
pgrep -af "run_all_experiments|train_baselines|train_teacher|train_m7|bowl_transport_diagnosis|bowl_feasibility_sweep"
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
git status --short
```

If the edge main line is already under heavy load, wait for a quieter window before launching the next scoop diagnosis command.

---

## 5. Step A — Dump the merged scene config first

This proves what scene is actually being run.

```bash
python pta/scripts/bowl_transport_diagnosis.py \
  --mode config \
  --material sand \
  --seed 7 \
  --output-dir "$RUN_ROOT/config_sand"
```

Expected outputs:
- `results/scoop_debug/<timestamp>/config_sand/merged_scene_configs.json`
- `results/scoop_debug/<timestamp>/config_sand/run_metadata_start.json`
- `results/scoop_debug/<timestamp>/config_sand/run_metadata_end.json`

Quick check:

```bash
python - <<'PY'
import json
import os
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "config_sand" / "merged_scene_configs.json"
data = json.loads(path.read_text())
for name in ["baseline_bowl", "flat_bowl"]:
    cfg = data[name]["scene_config"]
    env = data[name]["env"]
    print(f"[{name}]")
    print("  task_layout:", cfg["task_layout"])
    print("  particle_pos:", cfg["particle_pos"])
    print("  particle_size:", cfg["particle_size"])
    print("  tool_type:", cfg["tool_type"])
    print("  total_particles:", env["total_particles"])
    print()
PY
```

Pass condition for this step:
- `flat_bowl.scene_config.task_layout == "flat"`
- `tool_type == "bowl"`
- the dumped config is saved before any retention or scan comparison.

---

## 6. Step B — Retention smoke test (cheap, required)

Run the three-group retention diagnosis once on sand before any longer sweep:

```bash
python pta/scripts/bowl_transport_diagnosis.py \
  --mode retention \
  --material sand \
  --seed 7 \
  --repeats 1 \
  --hold-steps 20 \
  --output-dir "$RUN_ROOT/retention_smoke_sand"
```

Expected outputs:
- `retention/lift_only_hold/trial_01/metrics.csv`
- `retention/lift_full_hold/trial_01/metrics.csv`
- `retention/traverse_slow/trial_01/metrics.csv`
- `retention/retention_summary.csv`
- keyframes: `lift_end.png`, `hold_mid.png`, `hold_end.png`

Quick summary:

```bash
python - <<'PY'
import csv
import os
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "retention_smoke_sand" / "retention" / "retention_summary.csv"
rows = list(csv.DictReader(path.open()))
for row in rows:
    print(
        row["diagnosis_group"],
        "start=", row["start_n_on_tool"],
        "mid=", row["mid_n_on_tool"],
        "end=", row["end_n_on_tool"],
        "spill=", row["end_spill_ratio"],
        "drop_to_zero=", row["dropped_to_zero"],
    )
PY
```

Interpretation:
- if `lift_only_hold` and `lift_full_hold` keep substantial `n_on_tool`, the bowl can statically retain material;
- if `traverse_slow` collapses while the two lift holds stay alive, transport dynamics are the main problem.

---

## 7. Step C — Full retention diagnosis (3 repeats)

If the smoke run completes cleanly and does not disturb edge training, expand to the required 3 repeats:

```bash
python pta/scripts/bowl_transport_diagnosis.py \
  --mode retention \
  --material sand \
  --seed 7 \
  --repeats 3 \
  --hold-steps 120 \
  --output-dir "$RUN_ROOT/retention_full_sand"
```

Primary file to inspect:

```bash
python - <<'PY'
import csv
import os
from collections import defaultdict
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "retention_full_sand" / "retention" / "retention_summary.csv"
rows = list(csv.DictReader(path.open()))
groups = defaultdict(list)
for row in rows:
    groups[row["diagnosis_group"]].append(row)
for group, vals in groups.items():
    ends = [float(v["end_n_on_tool"]) for v in vals]
    spills = [float(v["end_spill_ratio"]) for v in vals]
    print(group)
    print("  end_n_on_tool:", ends)
    print("  end_spill_ratio:", spills)
PY
```

Decision gate after this step:
- **Static OK, traverse bad** → proceed to Step D (minimal scan).
- **Static already bad** → stop chasing speed first; document geometry/contact failure and do not launch long sweeps yet.

---

## 8. Step D — Minimal transport scan (4 combinations only)

This is the smallest useful scan for the dynamic-failure branch.

```bash
python pta/scripts/bowl_transport_diagnosis.py \
  --mode scan \
  --material sand \
  --seed 7 \
  --output-dir "$RUN_ROOT/scan_sand"
```

The current scan includes these four combinations:
- `slow_base`
- `mid_base`
- `slow_backtilt`
- `mid_backtilt`

Quick summary:

```bash
python - <<'PY'
import csv
import os
from pathlib import Path
path = Path(os.environ["RUN_ROOT"]) / "scan_sand" / "scan" / "scan_summary.csv"
rows = list(csv.DictReader(path.open()))
for row in rows:
    print(
        row["name"],
        "lift=", row["lift_full_n_on_tool"],
        "mid=", row["mid_traverse_n_on_tool"],
        "final_transfer=", row["final_transfer_efficiency"],
        "final_spill=", row["final_spill_ratio"],
        "dropped_midway=", row["dropped_midway"],
    )
PY
```

What to look for:
- Does back-tilt increase `mid_traverse_n_on_tool`?
- Does slower traverse preserve particles through the midpoint?
- Is there any combination where particles do **not** drop to zero midway?

---

## 9. Step E — Extend to other materials only after sand is understood

Do **not** jump into all materials before sand gives a clean diagnosis.

### 9.1 Retention smoke on snow and elastoplastic

```bash
for MAT in snow elastoplastic; do
  python pta/scripts/bowl_transport_diagnosis.py \
    --mode retention \
    --material "$MAT" \
    --seed 7 \
    --repeats 1 \
    --hold-steps 20 \
    --output-dir "$RUN_ROOT/retention_smoke_${MAT}"
done
```

### 9.2 Full retention on snow; elastoplastic only if smoke is unexpectedly positive

```bash
for MAT in snow; do
  python pta/scripts/bowl_transport_diagnosis.py \
    --mode retention \
    --material "$MAT" \
    --seed 7 \
    --repeats 3 \
    --hold-steps 120 \
    --output-dir "$RUN_ROOT/retention_full_${MAT}"
done
```

For elastoplastic, only promote beyond the smoke pass if the smoke result unexpectedly shows meaningful static retention. Otherwise keep elastoplastic documented as a separate capture/contact branch and do not spend longer retention budget here yet.

### 9.3 Minimal scan only on materials that survive static retention

```bash
for MAT in snow; do
  python pta/scripts/bowl_transport_diagnosis.py \
    --mode scan \
    --material "$MAT" \
    --seed 7 \
    --output-dir "$RUN_ROOT/scan_${MAT}"
done
```

This loop is sequential, so it stays within the “one heavy scoop job at a time” rule.

Do **not** run the minimal transport scan on elastoplastic by default. Elastoplastic should only enter this scan branch if a prior smoke or retention pass shows that it can actually reach a meaningful retained state before traverse.

---

## 10. Overnight run policy

Only start a longer overnight run if **both** are true:

1. edge training is still healthy and not visibly slowed by scoop diagnostics;
2. sand diagnosis shows at least one promising signal, for example:
   - lift holds retain particles,
   - or one scan combo keeps `mid_traverse_n_on_tool > 0`,
   - or one combo reaches a clearly nonzero final transfer.

Recommended overnight sequence:

```bash
for MAT in sand snow elastoplastic; do
  python pta/scripts/bowl_transport_diagnosis.py \
    --mode retention \
    --material "$MAT" \
    --seed 7 \
    --repeats 3 \
    --hold-steps 120 \
    --output-dir "$RUN_ROOT/overnight_retention_${MAT}" || break

  python pta/scripts/bowl_transport_diagnosis.py \
    --mode scan \
    --material "$MAT" \
    --seed 7 \
    --output-dir "$RUN_ROOT/overnight_scan_${MAT}" || break
done
```

Do **not** use `pta/scripts/bowl_feasibility_sweep.py` as the default overnight command while edge is active. That script writes to the fixed shared path `results/bowl_feasibility_sweep/` and is better treated as a later-stage tool after this diagnosis gate is passed.

---

## 11. What counts as a successful diagnosis pass

This diagnosis run is complete if either conclusion is supported by saved outputs:

### Outcome A — Dynamic transport is the main failure mode
Evidence pattern:
- `lift_only_hold` retains particles,
- `lift_full_hold` retains particles,
- `traverse_slow` loses most particles,
- scan shows back-tilt and/or slower speed improves midpoint retention.

### Outcome B — Geometry/contact is the main failure mode
Evidence pattern:
- `lift_only_hold` or `lift_full_hold` already loses particles quickly,
- traverse is not the first place where retention collapses,
- scan on speed/back-tilt does not rescue midpoint retention.

---

## 12. Deliverables checklist

At the end of the run, report these artifacts:

1. `merged_scene_configs.json` proving the explicit flat bowl scene.
2. `retention_summary.csv` for the three-group diagnosis.
3. `scan_summary.csv` for the minimal transport scan.
4. The exact commands used.
5. Start/end UTC timestamps.
6. GPU snapshot before and after.
7. Whether edge training was impacted: `yes` or `no`.

---

## 13. Suggested one-line team summary

Use this wording when reporting status:

> This pass is not trying to maximize score yet; it is isolating whether bowl failure happens because particles cannot stay in the tool at rest, or because they are lost mainly during transport.

---

## 14. Follow-on engineering ladder (native first, fallback second)

This section only applies **after** the diagnosis above is complete and the bowl line is explicitly being continued as a side-track.

### 14.1 Native-first rule

Before adding any fake adhesion or sticky logic, first exhaust the Genesis-native contact-quality knobs.

This means:

1. keep using the explicit `flat` bowl scene;
2. expose missing contact-quality knobs in the repo config path;
3. rerun the same `config -> retention -> scan` sequence with those knobs changed;
4. judge success by **final carry**, not by midpoint improvement alone.

### 14.2 Phase 1 — Native contact-quality tuning

Expose and sweep these knobs first:

- `enable_CPIC` in the MPM options path;
- robot-side bowl `coup_friction`;
- bowl `coup_softness`;
- `substeps`;
- bowl `sdf_cell_size` and, if needed, `sdf_min_res` / `sdf_max_res`;
- keep `mpm_grid_density=128` as the default baseline and only raise it if there is direct evidence of wall-resolution failure.

Current repo status:

- these bowl-only knobs are now wired through the repo config path;
- they remain **default-off**;
- they only become active for the explicit flat bowl path, not for edge-push defaults.

Recommended engineering order:

1. **sand only**: run one small native-tuning pass;
2. if sand shows a real end-state signal, repeat on snow;
3. treat elastoplastic as a separate capture/contact branch, not as the same transport problem.

Success gate for Phase 1:

- clearly nonzero `final_n_on_tool`, and
- clearly nonzero final transfer,
- in the explicit flat scene,
- without adding any post-step particle correction.

If the scan only improves `mid_traverse_n_on_tool` but final carry still collapses, do **not** declare Phase 1 a success.

**2026-04-10 recorded outcome:** this exact failure pattern has now occurred in the sand runs. Native tuning improved some midpoint carry signals, but all tested native configurations still ended with `final_n_on_tool = 0`. Therefore, the bowl side-track has officially crossed the runbook threshold for **Phase 1 native failure**.

### 14.3 Phase 2 trigger — sticky fallback is allowed only after native failure

Open the fallback path only if all are true:

1. the flat-scene diagnosis still says sand / snow are dynamic-transport failures;
2. Phase 1 native tuning still cannot produce useful final carry;
3. the goal has shifted from physics-faithful diagnosis to stable engineering carry.

### 14.4 Phase 2 — Post-step sticky fallback

Preferred first fallback:

1. add a local post-step hook immediately after `scene.step()` in `pta/envs/tasks/scoop_transfer.py`;
2. define a bowl-local carry region;
3. during the carry phase only, project or clamp particles back into that region and damp the outward velocity component;
4. leave capture and pour unmodified, so the fallback is scoped to transport stabilization.

Current repo status:

- this first guarded fallback path is now implemented in `pta/envs/tasks/scoop_transfer.py`;
- activation still requires explicit bowl flags plus the explicit `carry` phase;
- the default edge-push path does not enable it.

**2026-04-10 recorded outcome:** a minimal sticky-fallback validation has now been run on sand, and it also failed the final-carry gate (`final_n_on_tool = 0`). This does **not** invalidate Phase 2 as a category; it only means the first guarded sticky variant is too weak.

**Later 2026-04-10 recorded outcome:** two heavier fallbacks were then exercised on sand:

- hidden retention geometry in `panda_bowl.xml`;
- Genesis particle constraints.

Hidden geometry still ended with `final_n_on_tool = 0` in scan runs. Particle constraints produced the first weak nonzero terminal signal in retention, but scan-time `final_n_on_tool` still remained `0` and spill stayed high. The bowl line therefore still lacks meaningful final carry, even after progressing beyond minimal sticky fallback.

The heavier fallbacks that have now been exercised are:

- hidden retention geometry in `panda_bowl.xml`;
- direct particle position / velocity setters or particle constraints via Genesis internals.

The main remaining heavier fallback class is:

- custom carry-phase force fields.

The next execution step should now be selected from these heavier fallbacks, still under bowl-only isolation and still reported separately from native Genesis bowl transport.

At this point, however, it is no longer enough to record “which fallback comes next.” The workflow should also explicitly record the emerging interpretation: the bowl can often achieve calm retention, but the simulator still fails to produce a robust, transportable load through dynamic carry.

### 14.5 Reality-gap note

Current evidence suggests a simulator-vs-reality gap at the **carried-load formation** level:

- static bowl retention can exist;
- midpoint retention can sometimes be improved;
- but no tested variant has yet produced robust final carry in the flat task.

This means the dominant problem is no longer just “needs more friction” or “needs a slightly deeper bowl.” The stronger interpretation is that the simulated material does not naturally form a stable transported packet under bowl carry, even when local retention is improved.

**2026-04-11 phase-diagnostic addition:** the dedicated phase run shows that the dominant loss happens at **carry onset**. In the current sand diagnosis, `n_on_tool` falls from `122` at `lift_full` to `1` at `carry_early`, and later phases only handle a tiny residue. This means the main question is no longer “does pour kill the payload?” but “does dynamic carry immediately deconfine the load?”

Recommended next discriminator:

- rerun the same phase diagnostic with a **much higher-wall bowl variant**;
- if `carry_early` survival rises sharply, shallow/open-mouth overflow is a major contributor;
- if `carry_early` still collapses, then the stronger explanation is that the simulated material never forms a stable transported packet, even with stronger wall containment.

### 14.6 Reporting rule

If Phase 2 is used, label outputs as **sticky fallback / retention fallback**.

Do **not** merge those runs into the same headline result table as native Genesis bowl transport.

Current repo status:

- the diagnosis script now writes both requested and effective runtime config views;
- use the effective runtime section, not just the requested override section, when verifying whether bowl tuning or sticky fallback actually became active.

### 14.7 Suggested output naming

Use separate run labels under `results/scoop_debug/<timestamp>/`:

- `native_tune_<material>/...`
- `sticky_fallback_<material>/...`

This keeps side-track evidence auditable and avoids mixing the two regimes.
