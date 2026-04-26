# Remote Reproduction Runbook

Date: 2026-04-26

This is the DSW/DLC handoff path for the Probe-Then-Act ablation-first diagnostic. The code repos are now on the personal GitHub remotes, while checkpoints are handled as release/storage artifacts because checkpoints are not stored in normal Git.

## Repos

- Probe repo: `git@github.com:jandan138/probe-then-act.git`
- Genesis runtime fork: `git@github.com:jandan138/Genesis.git`
- Genesis runtime commit: `0f82aa73f8c51a246cabdd4e5f3919224a62956b`
- Auto/status repo: `git@github.com:jandan138/Auto-claude-code-research-in-sleep.git`

## Fresh DSW Machine Bootstrap

```bash
mkdir -p /cpfs/shared/simulation/zhuzihou/dev
cd /cpfs/shared/simulation/zhuzihou/dev
git clone git@github.com:jandan138/probe-then-act.git
cd probe-then-act
cp .env.dsw.example .env.dsw
```

Edit `.env.dsw` only for machine-specific paths, DLC data sources, and checkpoint bundle location. Do not put secrets in it.

```bash
scripts/bootstrap_remote.sh .env.dsw
scripts/download_artifacts.sh .env.dsw
scripts/smoke_remote.sh .env.dsw
```

`bootstrap_remote.sh` clones `git@github.com:jandan138/Genesis.git` if `${GENESIS_ROOT}` is absent and installs the editable Genesis runtime plus probe requirements unless `BOOTSTRAP_SKIP_INSTALL=1`.

## Local WSL Smoke

```bash
cd /home/zhuzihou/dev/probe-then-act
cp .env.local.example .env.local
scripts/smoke_remote.sh .env.local
```

The smoke script checks Python imports, runs `pta/scripts/dlc/preflight_remote.sh`, and dry-runs the corrected OOD command through `pta/scripts/dlc/run_task.sh`.

## DLC Smoke And Ablation Jobs

After local smoke succeeds on the DSW machine:

```bash
PATH=/usr/bin:/bin PYTHON_BIN=python3 \
  bash pta/scripts/dlc/submit_ablation_sweep.sh --dry-run
```

Submit the six ablation jobs:

```bash
bash pta/scripts/dlc/submit_ablation_sweep.sh
```

Submit the corrected OOD ablation eval after the ablation checkpoints exist:

```bash
python -m pta.scripts.dlc.submit_jobs \
  --suite ood-ablation \
  --name pta_ood_ablation \
  --gpu-count 1
```

## Current Experiment Target

The current branch is the ablation-first diagnostic:

- Train `m7_noprobe` seeds `42,0,1`.
- Train `m7_nobelief` seeds `42,0,1`.
- Rerun corrected resumable OOD for `m7_noprobe` and `m7_nobelief`.

The local WSL machine was already running `m7_noprobe seed=42`; DSW can either resume from the bundled `m7_pta_noprobe_seed42/m7_pta_50000_steps.zip` checkpoint or rerun that seed from scratch.

## Artifact Policy

Checkpoints are not stored in normal Git. Use:

```bash
python scripts/build_checkpoint_manifest.py \
  --manifest results/dlc/checkpoint_manifest.json \
  --archive results/dlc/checkpoint_bundle.tar.gz
```

Upload `results/dlc/checkpoint_bundle.tar.gz` as a GitHub Release asset or copy it to CPFS/OSS, then set `CHECKPOINT_BUNDLE_URL` and optional `CHECKPOINT_BUNDLE_SHA256` in `.env.dsw`.

## Verification Commands

```bash
source /home/zhuzihou/dev/Genesis/.venv/bin/activate
pytest tests/test_remote_handoff_assets.py tests/test_dlc_shell_contract.py tests/test_dlc_submit_jobs.py -q
```

Expected result for the handoff layer: all tests pass.
