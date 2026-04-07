# Next Steps Plan: Config D + Paper Critical Path

> Internal execution plan for `probe-then-act`.
>
> **Last updated:** 2026-04-07 (Day 4 of 27)
> **Deadline:** 2026-04-30 (23 days remaining)
> **Prerequisite:** Gate 0.5 material sweep completed

---

## 0. Situation Assessment (Updated)

### What we now know

1. **Config D solves the material discriminability problem:**

   | Material | Before (y=-0.03) | Config D (y=0.02) |
   |----------|------------------|--------------------|
   | Snow | 22.3% | **87.3%** |
   | ElastoPlastic | 0.0% | **69.6%** |
   | Sand | 32.2% | **32.2%** |
   | Gap | 22.3pp | **55.1pp** |

   All materials > 30%. Material gap is 55 percentage points. One config change (`particle_pos y: -0.03 → 0.02`) was enough.

2. **Scooping is dead:** MPM has no particle-rigid adhesion. 0% transfer for all materials during horizontal traverse. However, **capture-phase retention IS material-discriminative** (useful as probe signal).

3. **反直觉排序 (Snow > ElastoPlastic > Sand):**
   - ElastoPlastic particles slide as a cohesive blob → 69.6%
   - Sand scatters granularly → only 32.2%
   - This supports the paper: simple heuristics ("lighter = easier") are wrong

4. **Joint-space residual control is validated** (from 48hr sprint).

5. **Paper method is 0% implemented** (all stubs).

### Config D is now the default
Applied to `pta/envs/builders/scene_builder.py`: `particle_pos: (0.55, 0.02, 0.20)`

---

## 1. Immediate: Gate 4 Retest with Config D (Day 5, ~4 hours)

### Step 1: Verify scripted baseline on Config D
Run scripted baseline (Sequence E) on sand with new config. Expect ~32% transfer.

### Step 2: Train residual PPO v3
- JointResidualWrapper with `residual_scale=0.1`
- Sand material, 200K steps
- Baseline should be ~32% (sand). Goal: maintain or improve.

### Step 3: Gate 4 evaluation
- `success_rate >= 70%` (sand: 32% > 30% threshold → success=1 per episode)
- `transferred_mass_frac >= 0.25` (32% > 25% → pass)
- Stable across 3 eval runs

**Expected: Gate 4 PASSES for sand.** Config D's 32% baseline already exceeds both thresholds.

---

## 2. Paper Scope: Focused 3-Method Paper (Confirmed)

| Method | Role | What it shows | Effort |
|---|---|---|---|
| **M1: Reactive PPO** | Lower bound | Fixed strategy fails on OOD materials | Ready (retrain with Config D) |
| **M8: Privileged Teacher** | Upper bound | Knowing material = best performance | Ready (train_teacher.py works) |
| **M7: Probe-Then-Act** | Our method | Probing infers material → adapts strategy | Need: belief encoder (~3 days) |

### Core paper story

> Train on sand (32% scripted baseline). Test on snow (87%) and elastoplastic (69.6%).
> - M1 (reactive) learns a sand-specific strategy → fails on snow/elastoplastic
> - M8 (teacher, knows material) → adapts per-material → good everywhere
> - M7 (probe-then-act) → infers material from probe → approaches teacher performance
> - Gap: M7 >> M1 on OOD, M7 ≈ M8

### Key paper claims
1. "A fixed manipulation strategy's performance varies by 55pp across materials (32-87%)" — Config D data
2. "Active probing enables material-adaptive control without privileged information" — M7 vs M1
3. "Cross-material Genesis MPM benchmark with 3 material families" — benchmark contribution

### Material split design
- **ID (training):** Sand (hardest for scripted baseline: 32%)
- **OOD-Material:** Snow (87%), ElastoPlastic (69.6%) — unseen material families
- **OOD-Params:** Sand with extreme E/nu/rho — same family, shifted params

Training on the hardest material (sand) means the learned policy must generalize UP to easier materials. This is a stronger claim than training on an easy material.

---

## 3. Implementation Roadmap (Days 5-22)

### Phase A: Gate 4 Pass + Baseline (Days 5-8, 4 days)

**A1. Gate 4 retest** (Day 5)
- Verify Config D baseline
- Train residual PPO on sand, confirm Gate 4 pass
- If pass: proceed

**A2. M1 Reactive PPO baseline** (Days 6-7)
- JointResidualWrapper, no probe, no privileged obs
- Train on sand, 500K steps, 3 seeds
- Evaluate on sand, snow, elastoplastic

**A3. M8 Teacher baseline** (Days 7-8)
- JointResidualWrapper + PrivilegedObsWrapper
- Train on sand, 500K steps, 3 seeds
- Evaluate on all materials

### Phase B: Core Method (Days 8-13, 5 days)

**B1. Probe Phase Design** (Day 8)
- Add probe phase to episode: first 3 steps = scripted tap/press
- Probe at 3 positions (center, left, right of material)
- Record (action, delta_obs) as probe trace
- Scoop capture-retention data is material-discriminative → use as probe signal

**B2. Latent Belief Encoder** (Days 9-10)
- File: `pta/models/belief/latent_belief_encoder.py`
- Input: probe traces (3 × obs_dim) → 2-layer MLP → z (16D) + log_sigma (16D)
- Trained end-to-end with task policy

**B3. Belief-Conditioned Task Policy** (Day 11)
- Append z to observation → standard PPO policy
- JointResidualWrapper action space unchanged (7D residual)

**B4. Training Script** (Day 12)
- M7 training: probe phase → encode belief → task policy
- Train on sand, 500K-1M steps, 3 seeds

**B5. Ablations** (Day 13)
- No-Probe: skip probe phase, z=zeros → shows probing matters
- No-Belief: probe but don't encode → shows belief encoding matters

### Phase C: Evaluation (Days 14-16, 3 days)

- Run M1, M7, M8 + ablations on:
  - ID: Sand (training distribution)
  - OOD-Material: Snow, ElastoPlastic
  - OOD-Params: Sand with E=1e3 (soft), E=1e5 (hard)
- 10 episodes × 3 seeds per condition
- Produce `results/main_results.csv`

### Phase D: Paper Writing (Days 16-22, 7 days)

```
/paper-plan → /paper-figure → /paper-write → /paper-compile → /auto-paper-improvement-loop
```

### Phase E: Buffer + Submission (Days 22-27, 5 days)

---

## 4. Key Risks and Mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| Residual PPO doesn't improve beyond 32% on sand | Medium | Wider residual scale, curriculum annealing |
| M7 ≈ M1 (belief doesn't help) | Medium | Ensure probe signal is rich; verify belief z encodes material |
| M7 fails on OOD (snow/elastoplastic) | Low | Config D shows 55pp gap → large signal to learn |
| Training too slow | Medium | Vast.ai for parallel seeds |
| Paper timeline too tight | Medium | Start outline Day 14; ARIS pipeline for writing |

---

## 5. Go/No-Go Checkpoints

### Checkpoint 1: Day 5 (Gate 4 retest)
- Sand transfer ≥ 30% with Config D? → Expected YES (scripted = 32%)

### Checkpoint 2: Day 13 (Method trained)
- M8 > M1 on OOD? (Teacher with privileged info beats reactive?)
- M7 > M1 on OOD? (Probe-then-act beats reactive?)
- YES → write full paper
- M8 > M1 but M7 ≈ M1 → debug belief encoder
- M8 ≈ M1 → task design problem, escalate

### Checkpoint 3: Day 16 (Results table)
- Clear M7 advantage on ≥ 1 OOD split? → write paper
- No clear advantage → write benchmark paper (narrower claims)

---

## 6. Files to Modify/Create

### Done
- **MODIFIED** `pta/envs/builders/scene_builder.py` — Config D particle position

### Phase A (Days 5-8)
- **RUN** Gate 4 retest on Config D
- **RUN** M1 + M8 baselines

### Phase B (Days 8-13)
- **IMPLEMENT** `pta/models/belief/latent_belief_encoder.py`
- **IMPLEMENT** `pta/models/policy/task_policy.py` (or: just append z to obs)
- **MODIFY** `pta/envs/tasks/scoop_transfer.py` — add probe phase
- **CREATE** `pta/scripts/train_m7.py`
- **CREATE** `pta/scripts/train_baselines.py`

### Phase C (Days 14-16)
- **CREATE** `pta/scripts/run_ood_eval.py`
- **CREATE** `results/main_results.csv`

---

## 7. Deprioritized Items

- Level-and-Fill task (Task B)
- Scooping as transfer mechanism (confirmed dead — MPM no adhesion)
- M2/M3/M4/M5 baselines
- Risk head / uncertainty calibration
- Learned probe policy (use scripted probes)
- Vision/tactile encoders (proprioception only)
- 5-seed sweeps (use 3 seeds)

---

## 8. Key Discovery Log

| Date | Finding | Impact |
|---|---|---|
| 04-07 AM | IK y-axis inversion is DLS coupling artifact, not Genesis bug | Gate 2 passed |
| 04-07 AM | `control_dofs_position()` PD fails (0.68m z-divergence) | JointResidualWrapper created |
| 04-07 PM | 42% was old config (y=0.03); current (y=-0.03) gives 12.6% | No trajectory mismatch |
| 04-07 PM | Material sweep: Sand 12.6%, Snow 22.3%, EP 0.0% | Edge-push needs geometry fix |
| 04-07 PM | Scooping 0% all materials (no MPM adhesion) | Scooping dead for transfer |
| **04-07 PM** | **Config D (y=0.02): Sand 32%, Snow 87%, EP 69%** | **Task viable, Gate 4 passable** |
| 04-07 PM | Capture-phase retention is material-discriminative | Useful as probe signal |

---

## 9. One-Sentence Plan

> Apply Config D (particle y=0.02), pass Gate 4 on sand (Day 5), train M1+M8 baselines (Days 6-8), implement belief encoder + probe-conditioned policy (Days 8-13), evaluate on 3 materials (Days 14-16), write paper with ARIS (Days 16-22).
