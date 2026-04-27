# Checkpoint Manifest

Date: 2026-04-26

Checkpoints are not stored in normal Git. The repo provides a manifest/bundle builder so a fresh DSW/DLC machine can retrieve the exact artifacts without bloating Git history.

Current shortest path: do not upload checkpoints before training. The DSW side can start the five remaining ablation jobs without baseline checkpoints because those jobs train new `m7_noprobe` and `m7_nobelief` models.

## Build Manifest

```bash
cd /home/zhuzihou/dev/probe-then-act
source /home/zhuzihou/dev/Genesis/.venv/bin/activate
python scripts/build_checkpoint_manifest.py \
  --manifest results/dlc/checkpoint_manifest.json \
  --archive results/dlc/checkpoint_bundle.tar.gz
```

By default the script searches:

- Main probe repo: `/home/zhuzihou/dev/probe-then-act`
- Stage-D worktree: `/home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d`

## Required Minimal Bundle

These are required only for corrected OOD replay and the current `m7_noprobe seed=42` ablation resume path:

- `checkpoints/m1_reactive_seed42/best/best_model.zip`
- `checkpoints/m1_reactive_seed0/best/best_model.zip`
- `checkpoints/m1_reactive_seed1/best/best_model.zip`
- `checkpoints/m7_pta_seed42/best/best_model.zip`
- `checkpoints/m7_pta_seed0/best/best_model.zip`
- `checkpoints/m7_pta_seed1/best/best_model.zip`
- `checkpoints/m8_teacher_seed42/best/best_model.zip`
- `checkpoints/m7_pta_noprobe_seed42/best/best_model.zip`
- `checkpoints/m7_pta_noprobe_seed42/m7_pta_50000_steps.zip`

Local manifest probe on 2026-04-26 found all 9 required artifacts present: 4 under the main repo and 5 under the Stage-D worktree. It also tracks five optional future ablation outputs that are not expected to exist until DSW jobs finish:

- `checkpoints/m7_pta_noprobe_seed0/best/best_model.zip`
- `checkpoints/m7_pta_noprobe_seed1/best/best_model.zip`
- `checkpoints/m7_pta_nobelief_seed42/best/best_model.zip`
- `checkpoints/m7_pta_nobelief_seed0/best/best_model.zip`
- `checkpoints/m7_pta_nobelief_seed1/best/best_model.zip`

## 2026-04-27 DLC M7 Ablation Resume Points

The five DLC ablation jobs failed on shared-storage writes before final model save. The latest durable resume checkpoint for each run is:

- `checkpoints/m7_pta_noprobe_seed0/m7_pta_400000_steps.zip`
- `checkpoints/m7_pta_noprobe_seed1/m7_pta_400000_steps.zip`
- `checkpoints/m7_pta_nobelief_seed42/m7_pta_400000_steps.zip`
- `checkpoints/m7_pta_nobelief_seed0/m7_pta_400000_steps.zip`
- `checkpoints/m7_pta_nobelief_seed1/m7_pta_400000_steps.zip`

All five were verified with `PPO.load(...).num_timesteps == 400000`. See `docs/30_records/DLC_M7_ABLATION_RESUME_2026-04-27.md`.

The 400k checkpoints were resubmitted on 2026-04-27 as 400k-to-500k DLC resume jobs:

- `no_probe seed=0`: `dlc14uard6mq7vsw`
- `no_probe seed=1`: `dlc15e9y4qc12v0j`
- `no_belief seed=42`: `dlc15o9jiielquzg`
- `no_belief seed=0`: `dlc15y94wa65t7c8`
- `no_belief seed=1`: `dlc16i8bnuxurvgb`

Do not use the original `train_ablation` path to continue these runs; it starts PPO from scratch. Use `pta/scripts/resume_m7.py` with `--resume-from <400k zip>` and `--target-timesteps 500000`.

## Remote Retrieval

Preferred storage choices:

- GitHub Release asset on `jandan138/probe-then-act`.
- CPFS path mounted on the DSW machine.
- OSS/object storage URL available from the DSW network.

Set the DSW env file:

```bash
CHECKPOINT_BUNDLE_URL=https://github.com/jandan138/probe-then-act/releases/download/<tag>/checkpoint_bundle.tar.gz
CHECKPOINT_BUNDLE_SHA256=<sha256-when-known>
```

Then run:

```bash
scripts/download_artifacts.sh .env.dsw
scripts/smoke_remote.sh .env.dsw
```

If `CHECKPOINT_BUNDLE_URL` is empty, `download_artifacts.sh` still writes a manifest for whatever checkpoints are already present on disk.

For a fresh-from-scratch DSW handoff, leave `CHECKPOINT_BUNDLE_URL` empty and submit the five missing ablation jobs with:

```bash
python -m pta.scripts.dlc.submit_jobs \
  --suite ablation \
  --name pta_ablation \
  --variants no_probe no_belief \
  --seeds 42 0 1 \
  --skip no_probe:42
```

For the current 2026-04-27 state, those five jobs already reached durable 400k checkpoints and have been resubmitted through the resume path above.
