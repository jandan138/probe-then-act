# Paper Plan

**Title**: Probe-Then-Act: When Active Probing Helps Context-Conditioned Reinforcement Learning under Hidden Dynamics
**One-sentence contribution**: Probe-Then-Act shows, on a controlled cross-material benchmark, that a deliberately minimal in-episode probing phase can improve context-conditioned RL on a recoverable elastoplastic OOD split while hurting on four irreversible or partly irreversible splits, identifying recoverability as a candidate boundary condition for active probing.
**Venue**: NeurIPS
**Type**: empirical/diagnostic
**Date**: 2026-04-30
**Page budget**: 9 pages to Conclusion, excluding references and appendix
**Section count**: 6 main sections plus appendix

## Claims-Evidence Matrix

| Claim | Evidence | Status | Section |
|-------|----------|--------|---------|
| Explicit probing can help context-conditioned RL when the elicited response is recoverable. | Elastoplastic split: 60.7% transfer for M7 vs 46.0% for M1; spill 39.3% vs 54.0%; success +16.7pp. | Narrowly supported | §1, §5 |
| Probing can be harmful when it irreversibly perturbs the state. | M7 underperforms M1 by 13--22pp on the other four splits. | Supported descriptively | §1, §5, §6 |
| Probe and encoder are necessary for the elastoplastic gain in this architecture. | No-probe and no-belief ablations both fall below the reactive baseline on elastoplastic. | Supported for this split and architecture | §5, Appendix |
| Passive privileged material parameters are not sufficient in the sand-only residual-control setup. | M8 collapses to 0% transfer on elastoplastic across three seeds. | Supporting observation only | §5, Appendix |
| The benchmark is a reproducible hidden-dynamics stress test. | Five splits, 55pp scripted-transfer spread, 3-seed protocol, ablation infrastructure. | Supported | §1, §4, Appendix |

## Structure

### §0 Abstract
- State hidden dynamics and context inference as the problem.
- Present explicit probing as an active data-collection budget.
- Reveal the asymmetric result early: one elastoplastic win and four losses.
- End with benchmark and boundary-condition framing.

### §1 Introduction
- Open on hidden dynamics in RL, not robotics tool use.
- Explain why history/context policies may fail when normal rollouts do not elicit useful information early enough.
- Introduce explicit probing as a minimal active intervention with possible state-perturbation cost.
- Preview the recoverable-response interpretation.
- Contributions: conditional empirical finding, benchmark, and boundary-condition hypothesis.

### §2 Related Work
- Context-conditioned RL and meta-RL.
- Active system identification and information-gathering control.
- Embodied probing and physical exploration.
- Deformable-material manipulation benchmarks.

### §3 Method
- Reuse shared method.
- Maintain minimal-instantiation wording.

### §4 Experiments
- Reuse shared experiments.
- Emphasize sand-only training as extrapolative hidden-dynamics generalization.

### §5 Results
- Present results as diagnostic evidence, not leaderboard claims.
- Keep all numbers unchanged.
- Keep M8 as supporting observation.

### §6 Conclusion and Limitations
- Fold discussion into a compact NeurIPS-style conclusion.
- Restate the recoverable-deformation hypothesis as interpretive.
- Name limitations: one positive material family, high variance, fixed probe, sim-state observations, no sim-to-real.

## Figure Plan

| ID | Type | Description | Data Source | Priority |
|----|------|-------------|-------------|----------|
| Fig 1 | Hero + effect plot | PTA pipeline plus per-split M7-M1 delta; skim reader sees the one positive split and four negative splits immediately. | Existing `fig1_hero.pdf` | HIGH |
| Fig 2 | Heatmap | Transfer efficiency by method and split, emphasizing asymmetric material response. | Existing generated figure | HIGH |
| Fig 3 | Dot plot | Ablation metrics with per-seed dispersion on elastoplastic. | Existing generated figure | HIGH |
| Fig 4 | Paired-seed slope chart | Matched M7 vs M1 seed differences by split. | Existing generated figure | MEDIUM |
| Table 1 | Main results | M1/M7/M8 transfer, spill, success, retained material. | Existing table | HIGH |
| Table 2 | Ablation | Probe and encoder ablations on elastoplastic. | Existing table | HIGH |

## Citation Plan

- §1 Introduction: cite existing hidden-dynamics/domain-randomization/meta-RL/context-inference references from `shared/references.bib`; do not invent new BibTeX.
- §2 Related Work: reuse existing keys for PEARL/VariBAD/meta-RL/system-identification/active perception/deformable benchmarks if present. If a needed NeurIPS/ICLR/ICML citation is missing, add only after verifying metadata.
- §3-§5: keep current citations unless wording introduces a new comparison.

## Reviewer Feedback Targets

- A NeurIPS reviewer should understand by page 1 that the paper is a scoped diagnostic study, not a broad robustness claim.
- The asymmetric result must be visible before the method section.
- The appendix must provide enough protocol detail for reproducibility without padding the main text.

## Next Steps

- Rewrite NeurIPS Introduction.
- Rewrite NeurIPS Related Work.
- Create NeurIPS-specific Conclusion.
- Expand NeurIPS Appendix.
- Compile and run automated review.
