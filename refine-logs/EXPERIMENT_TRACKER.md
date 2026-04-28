# Experiment Tracker

**Current decision (2026-04-26):** Option 1 selected. Advance `m7_noprobe` and `m7_nobelief` ablations first; do not launch M2, elastoplastic expansion, or paper writing until ablation OOD evidence is available.

**Execution backend note:** R001 finished locally. R002-R006 are the remaining ablation training jobs and may run through PAI-DLC. DLC truth lives in `results/dlc/jobs.jsonl` and `results/dlc/runs/*.json`; do not submit a duplicate `m7_noprobe seed=42` job unless intentionally isolating results in a separate root.

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R000 | M0 | Record result-to-claim verdict | M1 vs M7 vs M8 | All corrected OOD v2 | transfer, spill, success | MUST | DONE | Verdict: original claims not supported |
| R001 | M1 | Train no-probe ablation | `m7_noprobe` seed 42 | train sand | eval reward, checkpoint | MUST | DONE | Local run complete. Best eval `25735.26 +/- 1.74 @440k`; final eval `23634.73 +/- 2.03 @500k`; final checkpoint `checkpoints/m7_pta_noprobe_seed42/m7_pta_final.zip`; best checkpoint exists under `checkpoints/m7_pta_noprobe_seed42/best/`. |
| R002 | M1 | Train no-probe ablation | `m7_noprobe` seed 0 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc14uard6mq7vsw`; final checkpoint `checkpoints/m7_pta_noprobe_seed0/m7_pta_final.zip`; `num_timesteps=500352`. |
| R003 | M1 | Train no-probe ablation | `m7_noprobe` seed 1 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc15e9y4qc12v0j`; final checkpoint `checkpoints/m7_pta_noprobe_seed1/m7_pta_final.zip`; `num_timesteps=500352`. |
| R004 | M1 | Train no-belief ablation | `m7_nobelief` seed 42 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc15o9jiielquzg`; final checkpoint `checkpoints/m7_pta_nobelief_seed42/m7_pta_final.zip`; `num_timesteps=500352`. |
| R005 | M1 | Train no-belief ablation | `m7_nobelief` seed 0 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc15y94wa65t7c8`; final checkpoint `checkpoints/m7_pta_nobelief_seed0/m7_pta_final.zip`; `num_timesteps=500352`. |
| R006 | M1 | Train no-belief ablation | `m7_nobelief` seed 1 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc16i8bnuxurvgb`; final checkpoint `checkpoints/m7_pta_nobelief_seed1/m7_pta_final.zip`; `num_timesteps=500352`. |
| R007 | M2 | Evaluate with ablations | OOD v2 resumable | all current splits | transfer, spill, success | MUST | TODO | Optional ablation checkpoints will add rows automatically |
| R008 | M3 | Smoke passive-memory baseline | `rnn_ppo` seed 42 | train sand + quick eval | checkpoint loads, quick metrics | CONDITIONAL | TODO | Verify sb3-contrib and OOD runner support before full M2 |
| R009 | M4 | Full passive-memory baseline | `rnn_ppo` seeds 42/0/1 | all corrected OOD v2 | transfer, spill, success | CONDITIONAL | TODO | Only if continuing explicit-belief claim |
| R010 | M5 | Confirm elastoplastic signal | M1/M7/best ablation extra seeds | `ood_elastoplastic` | paired deltas | NICE | TODO | Only if Block 1 suggests salvage |
