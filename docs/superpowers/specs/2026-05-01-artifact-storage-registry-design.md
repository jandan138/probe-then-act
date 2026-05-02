# Artifact Storage Registry Design

Date: 2026-05-01
Project: Probe-Then-Act

## Problem

Probe-Then-Act training artifacts are currently produced as ignored runtime files under `checkpoints/`, `logs/`, and `results/`. This kept Git lightweight, but it also allowed critical checkpoints to disappear when local worktrees, DSW workspaces, or temporary bundles were removed. The current pre-submission gate needs `checkpoints/m7_pta_seed42/best/best_model.zip`, but the original checkpoint is not recoverable from Git, GitHub releases, or the searched CPFS/tmp paths.

The project needs a durable artifact registry that treats checkpoints as first-class research evidence without storing large binary files in Git.

## Goals

- Persist every claim-critical checkpoint outside Git in a stable CPFS artifact root.
- Record enough metadata to identify the exact code, command, DLC job, environment, and checkpoint hashes used for each result.
- Verify artifacts before marking a run complete: file exists, SHA256 is recorded, and SB3 checkpoints load with expected `num_timesteps`.
- Produce portable checkpoint bundles for remote replay, paper audit, or GitHub Release/OSS upload.
- Keep the workflow simple enough to run after local jobs, DLC jobs, and recovery jobs.

## Non-Goals

- Do not commit checkpoint `.zip` files to Git.
- Do not build a database service.
- Do not retroactively fabricate missing artifacts; missing old checkpoints must be recorded as missing or regenerated with an explicit `recovered_by_retraining` provenance marker.

## Recommended Approach

Use a repo-local registry tool plus a CPFS-backed artifact root.

Artifact root:

```text
/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/
```

Run layout:

```text
<artifact-root>/<YYYYMMDD>/<run_id>/
  artifact_manifest.json
  command.txt
  env.json
  checkpoints/
  logs/
  results/
  checkpoint_bundle.tar.gz
```

Each `run_id` should include the method, seed, and origin, for example:

```text
20260501_dlc1hn82yye94ojd_m7_pta_seed42_recovery
```

## Components

### Artifact Registry Tool

Add `tools/artifact_registry.py` with subcommands:

- `scan`: inspect known required checkpoint paths and report present/missing artifacts.
- `verify`: load present SB3 checkpoints with `PPO.load`, collect `num_timesteps`, file size, and SHA256.
- `register-run`: copy or hardlink selected `checkpoints/`, `logs/`, and `results/` into the artifact root and write `artifact_manifest.json`.
- `bundle`: create `checkpoint_bundle.tar.gz` containing verified checkpoints plus the manifest.
- `restore`: unpack a bundle or copy registered artifacts back into a repo checkout.

### Manifest Schema

`artifact_manifest.json` should include:

- `schema_version`
- `project`
- `git_commit`
- `repo_root_at_registration`
- `run_id`
- `origin`: `local`, `dlc`, or `recovered_by_retraining`
- `dlc_job_id` and `dlc_display_name` when applicable
- `command`
- `env`: selected paths and runtime variables only, no secrets
- `artifacts`: list of entries with `logical_name`, `relative_path`, `storage_path`, `size_bytes`, `sha256`, `kind`, `required_for`, `num_timesteps`, and `load_status`
- `result_files`: CSV/JSON outputs linked to the checkpoints
- `created_at_utc`

### Required Artifact Sets

Maintain named requirements in the tool instead of scattering checkpoint lists through docs:

- `presub-g2`: `m7_pta_seed42/best/best_model.zip`
- `presub-extra-eval`: `m1_reactive_seed2`, `m1_reactive_seed3`, `m7_pta_seed2`, `m7_pta_seed3`
- `corrected-ood-replay`: M1/M7/M8 main checkpoints used for original OOD tables
- `ablation-replay`: no-probe/no-belief checkpoints and final 500K artifacts

## Data Flow

1. Training writes normal outputs under the active repo checkout.
2. Post-run registration copies verified outputs into the artifact root.
3. The manifest records both runtime source paths and durable storage paths.
4. Bundles are generated from the artifact root, not from temporary worktrees.
5. Future DSW/DLC jobs restore from a manifest or bundle, then verify before use.

## DLC Integration

After a DLC training job exits successfully, run a bounded post-run command:

```bash
python tools/artifact_registry.py register-run \
  --run-id "$DLC_RUN_ID" \
  --origin dlc \
  --dlc-job-id "$DLC_JOB_ID" \
  --requirement presub-g2
```

For existing `run_task.sh`, this should be added as an explicit mode or post-command step, not as hidden behavior that can mask training failures.

## Error Handling

- If a required artifact is missing, exit nonzero and print the missing logical names.
- If `PPO.load` fails, mark `load_status=failed` and exit nonzero for required artifacts.
- If CPFS copy fails, do not delete source artifacts.
- If an artifact was regenerated because the original was lost, require `origin=recovered_by_retraining` and record the recovery command and DLC job id.

## Testing

Add focused tests that use tiny fake files and monkeypatched `PPO.load`:

- manifest includes git commit, run id, command, paths, SHA256, and artifact status
- missing required checkpoint causes nonzero exit
- load failure causes nonzero exit for required artifacts
- bundle contains manifest and verified artifact paths
- restore recreates the expected checkpoint tree

## Acceptance Criteria

- Running `scan --requirement presub-g2` clearly reports whether `m7_pta_seed42` is available.
- Running `verify --requirement presub-g2` proves the checkpoint loads and records `num_timesteps`.
- A completed DLC recovery run can be registered into the artifact root with manifest and bundle.
- Future pre-submission gates consume artifacts from the registry or from a verified restored checkout, not from undocumented temporary paths.
