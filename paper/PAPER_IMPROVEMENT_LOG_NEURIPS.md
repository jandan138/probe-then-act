# Paper Improvement Log - NeurIPS Polish Pass

Date: 2026-05-01

Worktree: `/home/zhuzihou/dev/probe-then-act/.worktrees/paper-polish-review`

Branch: `paper-polish-review`

Venue entry: `paper/venues/neurips/main.tex`

## Score Progression

| Stage | External scores | Verdict | Key changes |
|-------|-----------------|---------|-------------|
| Round 0 baseline | Primary 6/10; GLM 5/10; Kimi 5/10; adversarial GLM 4/10 | Almost to Reject, depending on reviewer strictness | Baseline after NeurIPS rewrite |
| Round 1 polish | Not re-scored | Text-level risk reduction | Softer title, abstract, recoverability framing, M8 scope, ablation wording, benchmark attribution, simulator-state limitation |

## External Review Panel

One primary review agent and four auxiliary research reviewers were dispatched. Four returned substantive reviews; one auxiliary reviewer returned an empty result. Reviewers were instructed not to edit files and to respect the evidence constraints: one positive elastoplastic OOD split, four negative splits, M8 as single-seed diagnostic, and recoverability as interpretive rather than validated.

### Primary Reviewer - 6/10, Almost

Key findings:

- Recoverability still occasionally read as a causal rule rather than an interpretive hypothesis.
- M8 wording was still too strong for a single-seed diagnostic.
- The ablation phrasing "both components are necessary" overstated what a 3-seed split-specific ablation can establish.
- Benchmark attribution language was too clean: the simple task reduces confounds but does not eliminate them.
- Observation realism should be more reviewer-facing because aggregate particle statistics are simulator-derived.

Smallest recommended fixes:

- Soften recoverability language in abstract, results, and conclusion.
- Narrow all M8 claims to this single-seed sand-only privileged-parameter baseline.
- Rename or soften the ablation paragraph and Table 2 caption around necessity.
- Replace strong benchmark attribution language with "reduces confounds".
- Add "simulator-state diagnostic benchmark" language.

### GLM Reviewer - 5/10, Borderline

Key findings:

- The central +14.7pp elastoplastic gain is statistically fragile under 3 seeds and high variance.
- Recoverability is post-hoc without controlled validation.
- ML/RL novelty is thin unless the paper leans into the diagnostic negative result.
- Observations include simulator-derived privileged statistics.
- M8 speculation should be narrowed.

Smallest recommended fixes:

- Change "supports a recoverable-response hypothesis" to "is consistent with a recoverability interpretation".
- Treat recoverability as post-hoc interpretation rather than a validated principle.
- Add stronger wording that the evidence is directional and underpowered.

### Kimi Reviewer - 5/10, Weak Reject to Borderline

Key findings:

- Title and abstract still implied broad "material-adaptive" success despite 4/5 negative splits.
- Recoverability received too much narrative weight for an unvalidated interpretation.
- M8 was labeled as single-seed but still carried too much comparative weight.
- Figure and caption framing should make the 4/5 negative result visible.
- Novelty should be positioned as a conditional empirical finding, not an architectural claim.

Smallest recommended fixes:

- Rewrite the title as a question about when active context acquisition helps.
- Open the abstract with "we study when active probing helps" rather than proposing a broadly adaptive method.
- Add explicit "conditional, not broad, robustness" language to the figure caption.

### Adversarial GLM Reviewer - 4/10, Reject

Key findings:

- The positive result may be statistically indistinguishable from zero at the current seed count.
- Recoverability is a post-hoc narrative without independent manipulation.
- M8 can be read as a red herring if framed as evidence that probing beats parameters.
- The ablation table is elastoplastic-only and should not imply general component necessity.
- "Belief" terminology and simulator-state observations create reviewer risk.

Smallest recommended fixes:

- Cut the M8 comparative claim that passive parameters failed while PTA's interaction trace succeeded.
- Downgrade recoverability from explanatory hypothesis to interpretive lens.
- Make the No-Belief versus M1 0.3pp gap explicitly uninterpretable at this seed count.

### Empty Auxiliary Review

The `research-kimi26` auxiliary reviewer returned no substantive content. It was not used for decisions.

## Fixes Implemented

1. Changed the NeurIPS title from a broad material-adaptive framing to a diagnostic question: `Probe-Then-Act: When Does Active Context Acquisition Help Under Hidden Dynamics?`
2. Rewrote the abstract to open with "we study when active probing helps" and to call the benchmark diagnostic.
3. Changed the abstract ablation claim from "show ... needed" to "suggest ... contribute to this split-specific gain".
4. Changed the abstract recoverability claim from "supports a hypothesis" to "is consistent with, but does not validate, a recoverable-response interpretation".
5. Changed the abstract M8 claim from "collapses, suggesting interaction traces are more useful" to a narrow single-seed diagnostic failure of this sand-only parameter-conditioned baseline.
6. Changed the introduction heading from "Recoverability as an explanatory hypothesis" to "Recoverability as an interpretive lens".
7. Softened the recoverability paragraph to state that the account remains post-hoc and under-constrained.
8. Added explicit "conditional, not broad, robustness" wording to the Figure 1 caption.
9. Changed the scope paragraph to call this a "simulator-state diagnostic" reading.
10. Changed benchmark design language from "attributed to material-adaptive behavior" to "reduces, but does not eliminate, task-engineering confounds".
11. Changed material-spread language from "forcing methods to adapt" to "making a single push profile insufficient for uniformly high scripted transfer".
12. Changed Results framing from a recoverability claim to compatibility with a recoverability-based interpretation.
13. Changed the M8 paragraph heading from "Privileged-parameter collapse" to "Single-seed privileged-parameter failure".
14. Removed the sentence comparing passive parameters directly against PTA's interaction trace as a general lesson.
15. Changed the ablation paragraph heading from "Both components are necessary" to "Both components contribute to the observed elastoplastic gain".
16. Added the No-Belief versus M1 0.3pp gap caveat as too small to interpret at this seed count.
17. Added that a proper recoverability test would independently vary relaxation time or damping while holding information content and task difficulty fixed.
18. Changed the conclusion to say recoverability is consistent but not validated, post-hoc, and under-constrained.
19. Changed the limitations paragraph to label the setup a simulator-state diagnostic benchmark.
20. Updated the shared ablation table caption to avoid causal necessity language.
21. Cleaned stale shared figure include captions that still mentioned statistical significance, p-values, and broad competitiveness.

## Verification

Commands run from `paper/`:

- `make neurips`
- `rg -n "Warning|undefined|Overfull|Citation" "venues/neurips/build/main.log" || true`
- Python check for abstract word count and LaTeX aux page labels
- `git status --short --branch`

Results:

- Build completed successfully: `venues/neurips/build/main.pdf`.
- Critical log pattern check returned no matches for `Warning|undefined|Overfull|Citation`.
- Abstract word count: 229 words.
- Conclusion label: page 9.
- Total PDF pages from `main.aux`: 14.
- `pdfinfo` was unavailable in this environment, so page count was verified from LaTeX aux output.

## PDFs

- Baseline PDF: `paper/venues/neurips/build/main_round0_original.pdf`
- Polished PDF: `paper/venues/neurips/build/main_round1_polished.pdf`
- Current PDF: `paper/venues/neurips/build/main.pdf`

These PDFs are in the build directory and are not necessarily tracked by git.
