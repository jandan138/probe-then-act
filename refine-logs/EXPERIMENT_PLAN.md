# Experiment Plan

**Problem**: Corrected OOD evaluation plus M7 component ablations show the current Probe-Then-Act family does not improve broad OOD robustness over reactive PPO (M1). M7 only improves on elastoplastic.

**Method Thesis (2026-04-29 update)**: After user-approved narrative pivot, the paper now claims a **narrow but defensible scope** under the **recoverable-deformation hypothesis**: active probing helps when probe-induced perturbations relax back on the task timescale (viscoelastic media), and hurts when they do not (granular non-cohesive media). The 5-split asymmetry is reframed as predicted by the hypothesis, not as broad robustness. Paper compiled at `paper/main.pdf` (9 pages, IEEE T-RL).

**Date**: 2026-04-29

## Direction Decision

- **Completed strategy**: Option 1, Ablation-First Diagnostic.
- **Decision source**: post-ablation result-to-claim review on 2026-04-29.
- **Rationale**: Broad PTA claims are contradicted by corrected OOD results, and ablations do not repair or localize the degradation enough to support a paper-facing robustness claim.
- **Completed scope**: `m7_noprobe` and `m7_nobelief` were trained for seeds `42`, `0`, and `1`; corrected resumable OOD v2 was rerun for the ablations.
- **Execution backend**: R001 finished locally; R002-R006 finished through the DLC handoff; R007 ablation OOD finished locally. DLC workers remain bounded compute workers only.
- **Go / no-go gate**: **NO-GO** for broad PTA robustness. Do not proceed to M2, elastoplastic confirmation, or paper-writing unless the user explicitly chooses a narrowed salvage path.

## Post-Ablation Verdict

| System | All-OOD Transfer Delta vs M1 | All-OOD Spill Delta vs M1 | All-OOD Success Delta vs M1 | Interpretation |
|---|---:|---:|---:|---|
| M7 full | -0.0858 | +0.0894 | 0.0000 | Worse overall, one unstable elastoplastic win |
| M7 no-probe | -0.2733 | +0.1365 | -0.3333 | Probe removal damages the M7 family |
| M7 no-belief | -0.1279 | +0.0693 | -0.2167 | Belief removal does not repair broad OOD |

The ablations support only a metric-scoped internal mechanism statement: probing helps relative to `m7_noprobe`, while belief helps transfer/success relative to `m7_nobelief` but not all-OOD spill. They do not support broad robustness over M1.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: Current broad robustness claim is not supported | Prevents overclaiming | Result-to-claim verdict recorded from complete corrected OOD table | B0 |
| C2: M7 degradation can be localized to probe, belief, or training instability | Needed before salvage | Ablations show metric-scoped component effects but do not identify a single removable module that repairs M1 regressions | B1 |
| C3: Explicit belief is better than passive memory | Only keep if M2 is run | M7 or a repaired variant beats M2 on OOD material with success and spill evidence | B2 |
| C4: A narrower material-specific PTA claim may exist | Salvage path if broad claim fails | Elastoplastic gain repeats across more seeds and is explained by ablations | B3 |

## Paper Storyline

- Main paper must not claim broad PTA robustness from the current run.
- Appendix can report the negative corrected OOD and ablation result as a diagnostic if the project pivots to a failure-analysis story.
- Experiments cut by default: M2/RNN, M6/uncertainty, OOD-tool, OOD-sensor, second-task validation, and paper-writing. Revive only after an explicit new hypothesis is chosen.

## Experiment Blocks

### Block 0: Result-to-Claim Postmortem

- **Claim tested**: C1
- **Why this block exists**: Establish an honest stopping point before paper writing.
- **Dataset / split / task**: Corrected OOD v2, edge-push, `id_sand`, `ood_snow`, `ood_elastoplastic`, `ood_sand_soft`, `ood_sand_hard`.
- **Compared systems**: M1, M7, M8 one-seed reference.
- **Metrics**: transfer, spill, success, failed episodes.
- **Setup details**: Existing completed run, 35/35 expected rows.
- **Success criterion**: Verdict recorded and claims narrowed.
- **Failure interpretation**: If not recorded, downstream paper writing will overclaim.
- **Table / figure target**: Internal finding, not paper main table unless pivoting to a negative study.
- **Priority**: DONE

### Block 1: M7 Component-Damage Ablations

- **Claim tested**: C2
- **Why this block exists**: M7 is worse than M1 on most splits; isolate whether probe execution, latent belief conditioning, or PPO wrapper complexity is the source.
- **Dataset / split / task**: Same edge-push training on sand, same corrected OOD v2 splits.
- **Compared systems**: M1, M7 full, `m7_noprobe`, `m7_nobelief`.
- **Metrics**: transfer, spill, success; primary diagnostic is M7 ablation delta against full M7 and M1.
- **Setup details**: Train `train_m7.py --ablation no_probe` and `--ablation no_belief` for seeds `42,0,1`, `500000` timesteps, residual scale `0.05`; rerun resumable OOD v2.
- **Success criterion**: One ablation clearly explains the degradation or preserves the elastoplastic gain while improving snow/parameter splits.
- **Result**: Not met. All M7 variants underperform M1 on all-OOD average transfer/spill; no-probe is much worse, and no-belief remains below M1.
- **Failure interpretation**: The current PTA wrapper/mechanism is not a viable broad-robustness paper contribution.
- **Table / figure target**: Ablation / diagnostic table.
- **Priority**: DONE, no-go

### Block 2: Passive-Memory Baseline Check

- **Claim tested**: C3
- **Why this block exists**: The original explicit-belief claim is impossible without M2.
- **Dataset / split / task**: Same edge-push training and corrected OOD v2 splits.
- **Compared systems**: M1, M2 RNN-PPO, M7 full or repaired M7.
- **Metrics**: transfer, spill, success.
- **Setup details**: First run a seed-42 smoke test with `train_student.py --method rnn_ppo`; confirm checkpoint naming and evaluator support before full 3-seed training. Add M2 to OOD evaluator only after smoke checkpoint loads.
- **Success criterion**: M7/repaired M7 beats M2 on OOD-material in both success and spill, not just one seed.
- **Failure interpretation**: If M2 matches or beats M7, remove explicit-belief superiority from the paper.
- **Table / figure target**: Main comparison only if M7 is repaired; otherwise diagnostic appendix.
- **Priority**: CUT unless a new explicit-belief claim is approved.

### Block 3: Elastoplastic Confirmation

- **Claim tested**: C4
- **Why this block exists**: Elastoplastic is the only positive split, but current evidence is seed-unstable.
- **Dataset / split / task**: `ood_elastoplastic` only, plus ID sanity.
- **Compared systems**: M1, M7 full, best ablation from Block 1.
- **Metrics**: transfer, spill, success; paired seed deltas.
- **Setup details**: Add at least two more seeds if compute allows; otherwise rerun seed 42/0/1 with fixed deterministic settings to check reproducibility.
- **Success criterion**: Positive transfer/spill/success deltas on most seeds, not only the aggregate mean.
- **Failure interpretation**: If unstable, treat the elastoplastic win as noise and pivot.
- **Table / figure target**: Narrow material-specific claim figure only if confirmed.
- **Priority**: DEFERRED; only if explicitly pivoting to a narrow dynamics-adaptation story.

### Block 4: Failure-Mode Inspection

- **Claim tested**: C2/C4
- **Why this block exists**: M7's largest regressions are snow and sand parameter shifts; understanding behavior may suggest a repair.
- **Dataset / split / task**: `ood_snow`, `ood_sand_soft`, `ood_sand_hard` rollouts from M1/M7.
- **Compared systems**: M1, M7 full, best ablation.
- **Metrics**: trajectory videos, transfer/spill over episode, probe-phase displacement, final particle distribution.
- **Setup details**: Save 3 representative videos per split/method after Block 1 OOD eval.
- **Success criterion**: Identify a concrete failure mechanism, e.g. probe perturbation degrades initial particle geometry or belief conditioning biases residuals.
- **Failure interpretation**: If no clear mechanism, avoid mechanism claims and pivot.
- **Table / figure target**: Qualitative diagnostic panel.
- **Priority**: DEFERRED; run only after an explicit failure-analysis pivot is approved.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Freeze result-to-claim verdict | Record completed OOD table and verdict | `claim_supported=no` recorded | Done | None |
| M1 | Determine whether M7 components hurt | Train `m7_noprobe` / `m7_nobelief`, seeds 42/0/1 | Any variant beats M1 or explains M7 damage? | Done | Gate failed |
| M2 | Rerun OOD with ablations | Resume OOD v2 after ablation checkpoints exist | Exact rows complete, compare to M1/M7 | Done | `65` rows complete, no failed episodes |
| M3 | Decide whether M2 is worth running | M2 seed-42 smoke training and eval support | Checkpoint loads and quick eval works | Cut | No broad PTA claim remains to compare |
| M4 | Run M2 if continuing PTA claim | M2 seeds 42/0/1 + OOD support | M7/repaired M7 beats M2? | Cut | Revive only with new explicit-belief hypothesis |
| M5 | Confirm elastoplastic if still promising | Add seeds or deterministic rerun | Stable positive deltas? | Deferred | Only for a narrow dynamics-adaptation pivot |

## Compute and Data Budget

- The ablation decision cost is complete: 6 ablation training runs plus one expanded OOD eval.
- No additional compute is approved by default after the no-go verdict.
- If a new hypothesis is approved, the biggest bottleneck remains Genesis eval/training runtime and process-level memory growth.
- Mitigation for any future runs: keep resumable OOD enabled, keep local cron paused unless explicitly re-enabled, and use PAI-DLC only for bounded worker jobs.

## Risks and Mitigations

- **Risk**: M7 ablations are also worse than M1. **Status**: realized; pivot away from PTA robustness.
- **Risk**: M2 implementation/eval is incompatible with current OOD runner. **Mitigation**: run seed-42 smoke and add evaluator support before full training.
- **Risk**: Elastoplastic gain is not stable. **Mitigation**: do not use it as a paper claim unless it repeats across seeds.
- **Risk**: Timeline too short for full salvage. **Mitigation**: do not start paper-writing without a new supported claim.

## Final Checklist

- [x] Main OOD result is complete and verdict recorded
- [x] Direction selected: Option 1, ablation-first diagnosis
- [x] Ablation OOD result is complete
- [x] Post-ablation no-go verdict is recorded
- [x] **New pivot direction explicitly chosen (2026-04-29):** narrowed scope under the **recoverable-deformation hypothesis** — active probing helps when probe-induced perturbations relax back on the task timescale (viscoelastic), and hurts when they do not (granular non-cohesive). No additional compute used; existing experimental matrix is sufficient.
- [x] **Passive-memory baseline (M2/RNN) cut** — not part of the narrowed scope.
- [x] **Uncertainty / M6 contribution removed** from paper claims.
- [x] **Paper writing pipeline executed (2026-04-29):** `/paper-plan` → `/paper-figure` → `/paper-write` → `/paper-compile` → `paper/main.pdf` (9 pages, IEEE T-RL).
- [ ] Optional: `/auto-paper-improvement-loop` polish (≤2 GPT review rounds).
- [ ] Manual: IEEE author block + copyright form, supplementary release of code+checkpoints before 2026-04-30 deadline.
