# NeurIPS Figure Order Design

## Goal

Restore the NeurIPS main-text figure sequence so Fig. 1 is the \PTA{} workflow/architecture figure, while keeping the uploaded 2D result-summary image in the main text but later than Fig. 2.

## Approved Figure Order

1. Fig. 1: \PTA{} workflow/architecture (`fig1_workflow.pdf`) in the Introduction.
2. Fig. 2: Genesis scene schematic (`fig5_scene_schematic.pdf`) in Experimental Setup.
3. Fig. 3: absolute transfer heatmap (`fig2_main_comparison.pdf`) in Results.
4. Fig. 4: per-split delta plot (`fig1_effect_delta.pdf`) in Results.
5. Fig. 5: uploaded 2D result-summary image (`fig0_teaser.png`) in late Results as an after-the-fact qualitative summary, not as an opening teaser.

## Narrative Rules

- The Introduction must introduce the method using the workflow figure, not a result-summary image.
- The uploaded 2D image stays in the NeurIPS main text but appears only after quantitative evidence has been shown.
- The uploaded 2D image caption must frame it as a qualitative summary of the information--intervention tradeoff, not as new evidence or a broad robustness claim.
- Claim discipline remains unchanged: \PTA{} has one positive OOD elastoplastic split and negative performance on the other four splits; M8 is a single-seed diagnostic; recoverability is an interpretive lens rather than a validated law.

## Verification Requirements

- `make all` succeeds.
- NeurIPS `.aux` maps `fig:pta_workflow` to figure 1, `fig:scene_schematic` to figure 2, `fig:main` to figure 3, `fig:effect_delta` to figure 4, and `fig:teaser` to figure 5.
- `sec:conclusion:end` remains on NeurIPS page 9.
- Source grep finds no stale claims that M8 is averaged across three seeds and no stale dagger/magenta-callout/error-bar heatmap wording.
- `git diff --check` is clean.
