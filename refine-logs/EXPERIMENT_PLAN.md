# Experiment Plan

**Problem**: Corrected OOD evaluation shows the current Probe-Then-Act (M7) does not improve broad OOD robustness over reactive PPO (M1). M7 only improves on elastoplastic and is worse on ID, snow, and sand parameter shifts.

**Method Thesis**: The next research cycle should diagnose whether the active probe / belief mechanism contains a recoverable material-specific benefit, or whether the paper should pivot away from broad Probe-Then-Act robustness claims.

**Date**: 2026-04-26

## Direction Decision

- **Selected strategy**: Option 1, Ablation-First Diagnostic.
- **Decision source**: post-result-to-claim strategy discussion on 2026-04-26.
- **Rationale**: Broad PTA claims are contradicted by corrected OOD results; ablations are the smallest decisive step that can distinguish salvageable mechanism from failed wrapper/training complexity.
- **Immediate scope**: run only `m7_noprobe` and `m7_nobelief` for seeds `42`, `0`, and `1`, then rerun corrected resumable OOD v2.
- **Execution backend**: local cron/screen may continue single-GPU work, but PAI-DLC is approved for bounded ablation train/eval jobs after the repos are uploaded to DSW. DLC workers must not run ARIS, cron, opencode, Claude, Codex, or Auto-repo orchestration.
- **Go / no-go gate**: proceed to M2, elastoplastic confirmation, or paper-facing claims only if ablations explain or repair the M7 degradation. Otherwise pivot away from broad PTA robustness.

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1: Current broad robustness claim is not supported | Prevents overclaiming | Result-to-claim verdict recorded from complete corrected OOD table | B0 |
| C2: M7 degradation can be localized to probe, belief, or training instability | Needed before salvage | `no_probe` / `no_belief` ablations identify which component changes ID/OOD regressions | B1 |
| C3: Explicit belief is better than passive memory | Only keep if M2 is run | M7 or a repaired variant beats M2 on OOD material with success and spill evidence | B2 |
| C4: A narrower material-specific PTA claim may exist | Salvage path if broad claim fails | Elastoplastic gain repeats across more seeds and is explained by ablations | B3 |

## Paper Storyline

- Main paper must not claim broad PTA robustness from the current run.
- Appendix can report the negative corrected OOD result as a diagnostic if the project pivots.
- Experiments intentionally cut for now: OOD-tool, OOD-sensor, uncertainty calibration, second-task validation, and M2 full baseline until ablation evidence justifies retaining a PTA paper story.

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
- **Failure interpretation**: If all M7 variants underperform M1, the current PTA wrapper/mechanism is not a viable paper contribution.
- **Table / figure target**: Ablation / diagnostic table.
- **Priority**: MUST-RUN

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
- **Priority**: MUST-RUN if continuing PTA paper; otherwise cut.

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
- **Priority**: NICE-TO-HAVE after Block 1.

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
- **Priority**: MUST-RUN for diagnosis, not necessarily for paper.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | Freeze result-to-claim verdict | Record completed OOD table and verdict | `claim_supported=no` recorded | Done | None |
| M1 | Determine whether M7 components hurt | Train `m7_noprobe` / `m7_nobelief`, seeds 42/0/1 | Any variant beats M1 or explains M7 damage? | High, comparable to M7 training | Another OOM / slow Genesis; selected as the next approved direction |
| M2 | Rerun OOD with ablations | Resume OOD v2 after ablation checkpoints exist | Exact rows complete, compare to M1/M7 | Medium-long | More optional rows increase eval time |
| M3 | Decide whether M2 is worth running | M2 seed-42 smoke training and eval support | Checkpoint loads and quick eval works | Medium | sb3-contrib / recurrent eval mismatch |
| M4 | Run M2 if continuing PTA claim | M2 seeds 42/0/1 + OOD support | M7/repaired M7 beats M2? | High | If negative, original paper direction ends |
| M5 | Confirm elastoplastic if still promising | Add seeds or deterministic rerun | Stable positive deltas? | Medium | Gain may vanish |

## Compute and Data Budget

- Total must-run cost before decision: ablation training for 6 runs plus one expanded OOD eval.
- Additional cost if continuing original paper: M2 smoke + 3-seed training + evaluator support.
- Biggest bottleneck: Genesis eval/training runtime and process-level memory growth.
- Mitigation: keep resumable OOD enabled; rely on cron restarts locally; use PAI-DLC for parallel ablation workers when available; avoid adding optional splits until ablations justify them.

## Risks and Mitigations

- **Risk**: M7 ablations are also worse than M1. **Mitigation**: pivot away from PTA robustness; write a negative finding or switch method.
- **Risk**: M2 implementation/eval is incompatible with current OOD runner. **Mitigation**: run seed-42 smoke and add evaluator support before full training.
- **Risk**: Elastoplastic gain is not stable. **Mitigation**: do not use it as a paper claim unless it repeats across seeds.
- **Risk**: Timeline too short for full salvage. **Mitigation**: prioritize Block 1 and use result-to-claim again before any paper writing.

## Final Checklist

- [x] Main OOD result is complete and verdict recorded
- [x] Direction selected: Option 1, ablation-first diagnosis
- [ ] Novelty is isolated by ablations
- [ ] Passive-memory baseline is available if belief claim is retained
- [ ] Uncertainty contribution is either tested or removed from claims
- [ ] Nice-to-have runs are separated from must-run runs
