# Experiment Tracker

**Current decision (2026-04-26):** Option 1 selected. Advance `m7_noprobe` and `m7_nobelief` ablations first; do not launch M2, elastoplastic expansion, or paper writing until ablation OOD evidence is available.

**Execution backend note:** R001 is running locally in screen. Future R002-R006 may run through PAI-DLC after the repos are uploaded to DSW. DLC truth lives in `results/dlc/jobs.jsonl` and `results/dlc/runs/*.json`; avoid submitting a duplicate `m7_noprobe seed=42` job unless local R001 is intentionally abandoned or isolated.

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R000 | M0 | Record result-to-claim verdict | M1 vs M7 vs M8 | All corrected OOD v2 | transfer, spill, success | MUST | DONE | Verdict: original claims not supported |
| R001 | M1 | Train no-probe ablation | `m7_noprobe` seed 42 | train sand | eval reward, checkpoint | MUST | RUNNING | Screen `aris_m7_noprobe_s42`, PID `1518354`, log `logs/orchestration/train_m7_noprobe_seed42.log`; command `python pta/scripts/train_m7.py --ablation no_probe --seed 42 --total-timesteps 500000 --residual-scale 0.05` |
| R002 | M1 | Train no-probe ablation | `m7_noprobe` seed 0 | train sand | eval reward, checkpoint | MUST | TODO | Same command with `--seed 0` |
| R003 | M1 | Train no-probe ablation | `m7_noprobe` seed 1 | train sand | eval reward, checkpoint | MUST | TODO | Same command with `--seed 1` |
| R004 | M1 | Train no-belief ablation | `m7_nobelief` seed 42 | train sand | eval reward, checkpoint | MUST | TODO | `python pta/scripts/train_m7.py --ablation no_belief --seed 42 --total-timesteps 500000 --residual-scale 0.05` |
| R005 | M1 | Train no-belief ablation | `m7_nobelief` seed 0 | train sand | eval reward, checkpoint | MUST | TODO | Same command with `--seed 0` |
| R006 | M1 | Train no-belief ablation | `m7_nobelief` seed 1 | train sand | eval reward, checkpoint | MUST | TODO | Same command with `--seed 1` |
| R007 | M2 | Evaluate with ablations | OOD v2 resumable | all current splits | transfer, spill, success | MUST | TODO | Optional ablation checkpoints will add rows automatically |
| R008 | M3 | Smoke passive-memory baseline | `rnn_ppo` seed 42 | train sand + quick eval | checkpoint loads, quick metrics | CONDITIONAL | TODO | Verify sb3-contrib and OOD runner support before full M2 |
| R009 | M4 | Full passive-memory baseline | `rnn_ppo` seeds 42/0/1 | all corrected OOD v2 | transfer, spill, success | CONDITIONAL | TODO | Only if continuing explicit-belief claim |
| R010 | M5 | Confirm elastoplastic signal | M1/M7/best ablation extra seeds | `ood_elastoplastic` | paired deltas | NICE | TODO | Only if Block 1 suggests salvage |
