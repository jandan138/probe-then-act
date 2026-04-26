# Ablation-First Diagnostic Strategy

## Decision

On 2026-04-26, after corrected OOD v2 and result-to-claim review, the project selected **Option 1: Ablation-First Diagnostic** as the next research direction.

## Problem Anchor

- Corrected OOD v2 does not support the original broad Probe-Then-Act robustness claim.
- M7 improves only on `ood_elastoplastic`, while underperforming M1 on ID, snow, and sand parameter shifts.
- The current evidence does not isolate whether the regression comes from the probe phase, latent belief conditioning, or general M7 training/wrapper instability.

## Selected Path

Run the smallest decisive ablation package before any paper writing or broader experiment expansion:

- Train `m7_noprobe` seeds `42`, `0`, and `1`.
- Train `m7_nobelief` seeds `42`, `0`, and `1`.
- Rerun corrected resumable OOD v2 after checkpoints exist.
- Re-run result-to-claim before any paper-facing PTA claim.

## Deferred Paths

- Elastoplastic-only claim: deferred until ablations show the elastoplastic gain is mechanistic rather than seed noise.
- Negative/failure-analysis paper: fallback if ablations do not repair or explain M7's regressions.
- M2/RNN baseline: deferred unless the project keeps an explicit-belief-vs-passive-memory claim after ablations.

## Go / No-Go Gate

- **Go**: at least one ablation explains M7 damage or preserves the elastoplastic benefit while improving ID/snow/sand parameter splits enough to justify a narrowed claim.
- **No-go**: all M7 variants remain worse than M1 on broad OOD, or elastoplastic remains seed-unstable. In that case, pivot away from broad PTA robustness.

## Execution Boundary

The approved immediate execution unit is `R001` through `R007` in `refine-logs/EXPERIMENT_TRACKER.md`. Do not launch M2, extra elastoplastic seeds, uncertainty diagnostics, or paper writing until this gate is resolved.
