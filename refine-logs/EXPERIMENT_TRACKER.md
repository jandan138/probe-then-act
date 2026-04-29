# Experiment Tracker

**Current decision (2026-04-29):** Ablation OOD is complete and the post-ablation result-to-claim gate rejects broad Probe-Then-Act robustness. Pivot away from the current PTA paper claim unless a new, explicitly approved salvage hypothesis is chosen.

**Execution backend note:** R001 finished locally; R002-R006 finished through the DLC handoff; R007 ablation OOD finished locally. Local ARIS cron remains paused.

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| R000 | M0 | Record result-to-claim verdict | M1 vs M7 vs M8 | All corrected OOD v2 | transfer, spill, success | MUST | DONE | Verdict: original claims not supported |
| R001 | M1 | Train no-probe ablation | `m7_noprobe` seed 42 | train sand | eval reward, checkpoint | MUST | DONE | Local run complete. Best eval `25735.26 +/- 1.74 @440k`; final eval `23634.73 +/- 2.03 @500k`; final checkpoint `checkpoints/m7_pta_noprobe_seed42/m7_pta_final.zip`; best checkpoint exists under `checkpoints/m7_pta_noprobe_seed42/best/`. |
| R002 | M1 | Train no-probe ablation | `m7_noprobe` seed 0 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc14uard6mq7vsw`; final checkpoint `checkpoints/m7_pta_noprobe_seed0/m7_pta_final.zip`; `num_timesteps=500352`. |
| R003 | M1 | Train no-probe ablation | `m7_noprobe` seed 1 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc15e9y4qc12v0j`; final checkpoint `checkpoints/m7_pta_noprobe_seed1/m7_pta_final.zip`; `num_timesteps=500352`. |
| R004 | M1 | Train no-belief ablation | `m7_nobelief` seed 42 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc15o9jiielquzg`; final checkpoint `checkpoints/m7_pta_nobelief_seed42/m7_pta_final.zip`; `num_timesteps=500352`. |
| R005 | M1 | Train no-belief ablation | `m7_nobelief` seed 0 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc15y94wa65t7c8`; final checkpoint `checkpoints/m7_pta_nobelief_seed0/m7_pta_final.zip`; `num_timesteps=500352`. |
| R006 | M1 | Train no-belief ablation | `m7_nobelief` seed 1 | train sand | eval reward, checkpoint | MUST | DONE | DLC resume job `dlc16i8bnuxurvgb`; final checkpoint `checkpoints/m7_pta_nobelief_seed1/m7_pta_final.zip`; `num_timesteps=500352`. |
| R007 | M2 | Evaluate with ablations | OOD v2 resumable | all current splits | transfer, spill, success | MUST | DONE | Completed `65` per-seed rows and `25` aggregate rows in `results/ood_eval_per_seed.csv` / `results/main_results.csv`; all failed episode counts are `0`; ablations used `m7_pta_final.zip` checkpoints. |
| R008 | M3 | Smoke passive-memory baseline | `rnn_ppo` seed 42 | train sand + quick eval | checkpoint loads, quick metrics | CONDITIONAL | CUT | Do not run automatically after the post-ablation no-go; only revive if a new explicit-belief claim is approved. |
| R009 | M4 | Full passive-memory baseline | `rnn_ppo` seeds 42/0/1 | all corrected OOD v2 | transfer, spill, success | CONDITIONAL | CUT | Current PTA method does not have a broad robustness claim to compare against M2. |
| R010 | M5 | Confirm elastoplastic signal | M1/M7/best ablation extra seeds | `ood_elastoplastic` | paired deltas | NICE | DEFERRED | Only run if explicitly pivoting to a narrow dynamics-adaptation story; current EP signal is seed-unstable. |
| R011 | M2 | Record post-ablation verdict | M1 vs M7 variants | corrected OOD v2 | claim support | MUST | DONE | Result-to-claim verdict: `claim_supported=no` for broad PTA robustness; component ablations support only an internal-mechanism statement. |
