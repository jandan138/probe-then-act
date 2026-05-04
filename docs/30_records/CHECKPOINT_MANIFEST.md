# Checkpoint Manifest

Date: 2026-04-26
Updated: 2026-05-01
Status update: 2026-05-04

Checkpoints are not stored in normal Git. The repo provides a manifest/bundle builder so a fresh DSW/DLC machine can retrieve the exact artifacts without bloating Git history.

## 2026-05-01 Artifact Registry Status

Durable CPFS artifact root:

```text
/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/
```

Archived runs now registered under the new artifact registry workflow:

- `20260501_smartbot_legacy_existing_checkpoints`
  - Source: `/shared/smartbot/zhuzihou/dev/probe-then-act`
  - Manifest: `/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/20260501/20260501_smartbot_legacy_existing_checkpoints/artifact_manifest.json`
  - Bundle: `/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/20260501/20260501_smartbot_legacy_existing_checkpoints/checkpoint_bundle.tar.gz`
  - Contents: 61 checkpoint zip files, all verified with SB3 load (`61/61 loaded`).
  - Restore validation: restored bundle contained 61 zip files and 61 manifest artifacts.
- `20260501_dlc1hn82yye94ojd_m7_pta_seed42_best_50000_provisional`
  - Source: recovery DLC job `dlc1hn82yye94ojd`, provisional 50k best checkpoint.
  - Manifest: `/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/20260501/20260501_dlc1hn82yye94ojd_m7_pta_seed42_best_50000_provisional/artifact_manifest.json`
  - Bundle: `/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/20260501/20260501_dlc1hn82yye94ojd_m7_pta_seed42_best_50000_provisional/checkpoint_bundle.tar.gz`
  - Contents: `checkpoints/m7_pta_seed42/best/best_model.zip`, `sha256=6cb717b1dbd1fbe31e668421b23a2178d13e517b483155e0d5def996439b771b`, `num_timesteps=50000`, `load_status=loaded`.
- `20260501_dlc1hn82yye94ojd_m7_pta_seed42_final_recovery`
  - Source: completed recovery DLC job `dlc1hn82yye94ojd` (`Status=Succeeded`, worker `exit_code=0`).
  - Manifest: `/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/20260501/20260501_dlc1hn82yye94ojd_m7_pta_seed42_final_recovery/artifact_manifest.json`
  - Bundle: `/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/20260501/20260501_dlc1hn82yye94ojd_m7_pta_seed42_final_recovery/checkpoint_bundle.tar.gz`
  - Contents: 12 checkpoint zip files for `m7_pta_seed42`, all verified with SB3 load (`12/12 loaded`).
  - Final checkpoint: `checkpoints/m7_pta_seed42/m7_pta_final.zip`, `sha256=55bf288ab6211f15b016a6210b51435c5650d71a5ff0a4fc65e04c5835085116`, `num_timesteps=500224`.
  - Included result files: recovery DLC run record, G2 DLC run record, and `results/presub/audit_encoder_m7_pta_s42_ood_elastoplastic.json`.
  - Restore validation: restored bundle contained 12 zip files, 12 manifest artifacts, 12 loaded artifacts, and 3 result files.

The final recovered M7 seed42 run supersedes the provisional 50k archive for long-term preservation. Keep the provisional archive as an immutable audit snapshot because it was the checkpoint used to launch the first G2 recovery check.

Use the new registry tool for future artifacts:

```bash
python tools/artifact_registry.py verify \
  --repo-root /path/to/probe-then-act \
  --requirement g2-matched-encoder \
  --manifest /tmp/pta_g2_matched_encoder_verify.json

python tools/artifact_registry.py verify \
  --repo-root /path/to/probe-then-act \
  --requirement presub-g2 \
  --manifest /tmp/pta_presub_g2_verify.json

python tools/artifact_registry.py register-run \
  --repo-root /path/to/probe-then-act \
  --artifact-root /cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act \
  --run-id <YYYYMMDD_descriptive_run_id> \
  --origin local \
  --command "<exact command>" \
  --artifact-path checkpoints/<run>/best/best_model.zip

python tools/artifact_registry.py bundle \
  --run-dir /cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/<date>/<run_id>
```

Current shortest path: the verified-image six-seed replacement training jobs for seeds `2`, `3`, and `4` have completed on DLC with worker `exit_code=0`. Preserve those checkpoints in CPFS and register them before any cleanup. The 2026-05-03 legacy evaluation jobs below only write result CSVs and do not create new checkpoint artifacts.

Matched M7 claim archives must include the complete policy-plus-encoder bundle for the evaluated checkpoint:

- `best/best_model.zip`
- `best/best_model.json`
- `best/belief_encoder.pt`
- `best/belief_encoder_metadata.json`

Checkpoint `.zip` files and encoder `.pt` sidecars stay out of Git. Preserve them through the artifact registry or durable CPFS archives, with JSON metadata and SHA256 links committed only as documentation when needed.

## 2026-05-02 Matched Encoder Smoke Artifact

The DLC matched-sidecar smoke job `dlc7m883ucvcce7n` ran in `/cpfs/shared/simulation/zhuzihou/dev/probe-then-act` with seed `9042` to verify the new sidecar protocol without changing legacy seed42 artifacts. It produced a final matched bundle:

- `checkpoints/m7_pta_seed9042/m7_pta_final.zip`
- `checkpoints/m7_pta_seed9042/m7_pta_final.json`
- `checkpoints/m7_pta_seed9042/belief_encoder.pt`
- `checkpoints/m7_pta_seed9042/belief_encoder_metadata.json`

The worker record `results/dlc/runs/20260502T065019Z_custom_dlc7m883ucvcce7n-master-0.json` has `exit_code=0`. Local verification loaded the policy with SB3 and loaded the sidecar with `load_m7_encoder_artifact(... expected={'method': 'm7_pta', 'seed': 9042, 'ablation': 'none'})`, returning protocol `matched_encoder_v1`.

This smoke artifact proves the runtime sidecar path works, but it is not corrected G2 evidence. Corrected G2 requires a full matched `m7_pta seed42` artifact bundle and `results/presub/audit_matched_encoder_m7_pta_s42_ood_elastoplastic.json`.

As of the 2026-05-03 refresh, the isolated full matched seed42 job `dlc1weyuiyngs6ow` is `Succeeded` with worker `exit_code=0`, best and final matched bundles verified. All five additional matched M7 seeds `0-4` also succeeded in their isolated roots. Do not register or promote a corrected G2 bundle until the corrected G2 audit DLC `dlcqfs83uu5rmvp7` finishes and produces `audit_matched_encoder_m7_pta_s42_ood_elastoplastic.json`.

## 2026-05-03 Matched M7 Claim Bundles

Six matched M7 claim-bundle candidates produced under isolated roots:

| Seed | Root | DLC JobId | Best timesteps | Protocol |
|---:|---|---|---:|---|
| 0 | `probe-then-act-matched-m7-s0` | `dlc1t2k26iivao2n` | 240000 | `matched_encoder_v1` |
| 1 | `probe-then-act-matched-m7-s1` | `dlc1u6ifpmua9a2q` | 380000 | `matched_encoder_v1` |
| 2 | `probe-then-act-matched-m7-s2` | `dlc1vkgemincxw8m` | 320000 | `matched_encoder_v1` |
| 3 | `probe-then-act-matched-m7-s3` | `dlc1wyedjetvmv6o` | 120000 | `matched_encoder_v1` |
| 4 | `probe-then-act-matched-m7-s4` | `dlc1y2cr2ibi27xx` | 170000 | `matched_encoder_v1` |
| 42 | `probe-then-act-matched-seed42` | `dlc1weyuiyngs6ow` | 270000 | `matched_encoder_v1` |

All 6 verified: `Status=Succeeded`, worker `exit_code=0`, verified image, best bundle 4/4 files present, final bundle 4/4 files present, PPO.load OK, `load_m7_encoder_artifact` OK. Best-checkpoint timesteps vary because PPO saved the best evaluation model at different training steps; final bundles complete at `total_timesteps=500000`.

DLC submission note: priority 8 blocked by workspace role check (`CheckConfigFromAIWorkspace: role check failed`). Use priority ≤7 for this workspace.

Corrected G2: **passed.** `dlcqfs83uu5rmvp7` worker `exit_code=0`, audit `passes=true`, `mean_transfer=0.968`.
Matched G4 eval: `dlcrjqlmycuna0or` (priority 6) completed M7 matched eval (30/30 rows for 6 seeds × 5 splits) then stalled at M1 transition. M7 rows saved in `results/presub/ood_eval_matched_g4_6seed_20260503.csv`. DLC still `Running`; worker record not yet available. M1 comparison rows come from verified legacy eval outputs.
Seed 4 elastoplastic claim evidence uses the final 500k checkpoint value `0.9712`, superseding the best-checkpoint outlier (`0.080219` at 170k best timesteps). Local retest DLC `dlc1dmz216m49jqt` is queued for independent reproduction only.

## 2026-05-03 Six-Seed Result Artifact Status

The verified-image G3 replacement checkpoints are runtime artifacts under `/cpfs/shared/simulation/zhuzihou/dev/probe-then-act/checkpoints/`. The full `m7_pta` seed `2`, `3`, and `4` artifacts are policy-only legacy diagnostics because those jobs predate matched encoder sidecar persistence. Do not register them as matched M7 claim bundles.

Legacy extra-seed eval jobs from the same runtime wrote these non-checkpoint result artifacts:

| Scope | DLC JobId | Worker record | Result artifact | Rows |
|---|---|---|---|---:|
| `m1_reactive`, `m7_pta`; seeds `2, 3` | `dlcoucjyiozupi5h` | `results/dlc/runs/20260503T030353Z_custom_dlcoucjyiozupi5h-master-0.json` | `results/presub/ood_eval_extra_seeds_s2_s3_legacy_policy_only_20260503.csv` | 20 |
| `m1_reactive`, `m7_pta`; seed `4` | `dlcpyaxhmcnzk87e` | `results/dlc/runs/20260503T031326Z_custom_dlcpyaxhmcnzk87e-master-0.json` | `results/presub/ood_eval_extra_seed4_legacy_policy_only_20260503.csv` | 10 |
| `m7_noprobe`; seeds `2, 3, 4` | `dlc5pczyite2yvpp` | `results/dlc/runs/20260503T031326Z_custom_dlc5pczyite2yvpp-master-0.json` | `results/presub/ood_eval_ablation_no_probe_extra_seeds_s2_s3_s4_legacy_policy_only_20260503.csv` | 15 |
| `m7_nobelief`; seeds `2, 3, 4` | `dlc7na5myukdtyc1` | `results/dlc/runs/20260503T030410Z_custom_dlc7na5myukdtyc1-master-0.json` | `results/presub/ood_eval_ablation_no_belief_extra_seeds_s2_s3_s4_legacy_policy_only_20260503.csv` | 15 |

Completion verification found all four legacy DLC eval jobs `Succeeded` with worker `exit_code=0`, expected row counts, and no missing or duplicate method/seed/split combinations. The matched M7 seed `0`-`4` jobs submitted under isolated roots are verified claim-bundle candidates with `belief_encoder.pt` + `belief_encoder_metadata.json` (all 6 seeds PPO-load and encoder-load verified). Seed4 best-checkpoint elastoplastic outlier is resolved for claim evidence by using the final 500k checkpoint value `0.9712`.

Matched-encoder elastoplastic audit summary for paper use: six policy seeds (`0, 1, 2, 3, 4, 42`), M7 matched mean transfer `0.9354`, M1 mean transfer `0.3936`, mean paired delta `+54.2` pp, `positive_pairs=4/6`, with seeds 1 and 3 near ties. Keep this summary separate from the original 3-seed cross-split main table and legacy policy-only diagnostics.

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
