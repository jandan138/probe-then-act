# M7 Matched-Encoder Checkpoint Protocol Design

Date: 2026-05-02
Project: Probe-Then-Act

## Problem

M7 currently evaluates a policy checkpoint with a freshly constructed `LatentBeliefEncoder` inside `ProbePhaseWrapper`. The policy checkpoint is loaded from disk, but the encoder state that produced the latent belief vector `z` during training is not persisted with the policy.

The current G2 encoder-sensitivity audit therefore exposed a protocol vulnerability: a fixed `m7_pta seed42` PPO policy was paired with independently initialized evaluation encoders, and `ood_elastoplastic` transfer ranged from about 31% to 97%. That result is valid as a random-evaluation-encoder stress test, but it is not final evidence about the performance of a matched M7 system.

Important terminology: the current encoder is not trained by PPO. It is randomly initialized, frozen, and used as a fixed mapping while PPO learns a policy that reads its latent coordinates. The minimal checkpoint-consistent protocol is therefore a **matched-encoder checkpoint protocol**, not a claim that the encoder itself was optimized.

## Goals

- Persist the exact M7 belief encoder state that is paired with each future M7 policy checkpoint.
- Make normal M7 evaluation load the matched encoder by default.
- Fail fast when a full M7 policy checkpoint is missing its encoder sidecar, unless the caller explicitly requests random-encoder stress mode.
- Preserve old DLC runs as `policy-only legacy` diagnostics without mixing them into matched-encoder claims.
- Extend artifact registration so policy and encoder sidecars are archived together.
- Keep the implementation small and testable without requiring Genesis-heavy unit tests.

## Non-Goals

- Do not redesign M7 into an end-to-end trained encoder method in this change.
- Do not try to reconstruct old encoder weights from RNG state. That would be fragile and unsuitable as paper evidence.
- Do not stop the currently running policy-only DLC jobs. They remain useful for policy-seed variance auditing.
- Do not commit checkpoint `.zip` or encoder `.pt` files to Git.

## Current Code Diagnosis

The current data flow is:

```text
GenesisGymWrapper
  -> JointResidualWrapper
  -> ProbePhaseWrapper
  -> SB3 PPO MlpPolicy
```

`ProbePhaseWrapper` runs probe steps during `reset()`, encodes the collected probe traces with `LatentBeliefEncoder`, caches a latent vector `z`, and appends `z` to every observation. The PPO policy receives `[base_obs, z]` as a flat observation.

Current behavior creates mismatches:

- `pta/scripts/train_m7.py` constructs `ProbePhaseWrapper` without passing an encoder, so the wrapper constructs a fresh `LatentBeliefEncoder`.
- `train_m7.py` saves only SB3 policy checkpoints through `save_sb3_checkpoint`.
- `pta/scripts/run_ood_eval_v2.py` also constructs `ProbePhaseWrapper` without a saved encoder, so evaluation uses a new encoder.
- `EvalCallback` saves `best/best_model.zip` without saving a paired encoder sidecar.
- `train_m7.py` creates separate training and evaluation environments, so best-checkpoint selection can already involve a different encoder from the rollout environment unless construction is made explicit.

## Proposed Protocol

Every future full `m7_pta` policy checkpoint used as matched-encoder evidence must have a paired encoder artifact. The `no_probe` and `no_belief` ablations set `z` to zeros by design; they do not require encoder sidecars for semantic correctness, but their metadata should still state that no encoder is used.

For final checkpoints:

```text
checkpoints/m7_pta_seed<S>/m7_pta_final.zip
checkpoints/m7_pta_seed<S>/m7_pta_final.json
checkpoints/m7_pta_seed<S>/belief_encoder.pt
checkpoints/m7_pta_seed<S>/belief_encoder_metadata.json
```

For best checkpoints:

```text
checkpoints/m7_pta_seed<S>/best/best_model.zip
checkpoints/m7_pta_seed<S>/best/best_model.json
checkpoints/m7_pta_seed<S>/best/belief_encoder.pt
checkpoints/m7_pta_seed<S>/best/belief_encoder_metadata.json
```

For periodic checkpoints, the minimal implementation will not treat them as paper-facing matched-encoder artifacts. If a periodic checkpoint is later used for evaluation or claims, it must first receive the same paired encoder sidecar treatment as `best` and `final`.

The encoder metadata must include:

- `protocol`: `matched_encoder_v1`
- `method`: `m7_pta` or the M7 ablation run name
- `seed`
- `ablation`
- `trace_dim`
- `latent_dim`
- `hidden_dim`
- `num_layers`
- `n_probes`
- `paired_policy_path`
- `paired_policy_sha256`
- `stage`: `best`, `final`, or checkpoint step label
- `created_at_utc`
- `git_commit` when available
- DLC provenance for DLC claim artifacts: `dlc_job_id`, `dlc_display_name`, observed `JobSpecs[0].Image`, and command

The policy metadata must include matching links back to the encoder:

- `protocol`: `matched_encoder_v1`
- `encoder_mode`: `matched`
- `legacy_policy_only`: `false`
- `belief_encoder_path`
- `belief_encoder_sha256`
- `belief_encoder_metadata_sha256`

`belief_encoder.pt` should have a single explicit format: a Torch-serialized dictionary containing `format_version`, `state_dict`, and the encoder construction config. `belief_encoder_metadata.json` remains the source of truth for provenance, policy pairing, and claim status. Loaders must reject raw tensors, missing `state_dict`, or unknown format versions unless a migration path is explicitly added later.

Pairing must use repo-relative paths plus SHA256 hashes. Absolute paths are allowed only as runtime source-path provenance in manifests; they must not be the only identity binding between policy and encoder artifacts.

## Training Data Flow

`train_m7.py` should construct encoder state deliberately instead of letting each wrapper create an independent random encoder without persistence.

The minimal correct flow is:

1. Seed the run using the existing seed helper.
2. Create one canonical `LatentBeliefEncoder` config for the run.
3. Initialize the training encoder once.
4. Create the training environment with that encoder.
5. Create the eval environment with a copy of the same encoder state, not an independently initialized encoder.
6. Train PPO as before; the encoder remains frozen unless a future method explicitly changes that design.
7. When saving `best` and `final` policy checkpoints, save the paired encoder state and metadata at the same time.

The train and eval wrappers should not share the same Python module object if that creates vector-env ownership or device side effects. A state-dict copy is enough and easier to reason about.

The best-checkpoint path needs an explicit callback hook because SB3 `EvalCallback` saves `best_model.zip` internally. Implementation should either subclass/wrap `EvalCallback` or use a callback-on-new-best hook so `best_model.zip`, `best_model.json`, `belief_encoder.pt`, and `belief_encoder_metadata.json` are written as one accepted best-checkpoint event. A best policy checkpoint without its matching encoder sidecar is incomplete.

## Evaluation Data Flow

Normal full `m7_pta` evaluation should use matched-encoder mode by default:

```text
resolve policy checkpoint
resolve paired encoder sidecar
load encoder metadata
validate metadata against expected method, seed, ablation, latent_dim, n_probes
instantiate LatentBeliefEncoder
load state_dict on CPU or requested device
pass belief_encoder into ProbePhaseWrapper
evaluate PPO policy
```

The CLI/API contract should expose:

```text
--m7-encoder-mode {matched,random-stress}
--m7-random-encoder-seed <int>
```

`matched` is the default checkpoint-consistent evaluation mode. It is eligible for paper-facing claims only after replicated matched evaluations. `random-stress` is a diagnostic mode that intentionally creates a fresh encoder after setting the requested encoder seed.

If the full M7 checkpoint lacks an encoder sidecar, normal evaluation must fail with an explicit message such as:

```text
missing matched encoder artifact for m7_pta seed42; rerun with --m7-encoder-mode random-stress to run a diagnostic stress test
```

Random encoder evaluation remains useful, but it must be explicit and named as stress testing. Mode names:

- `matched`: default checkpoint-consistent evaluation mode
- `random-stress`: diagnostic mode that intentionally initializes a fresh encoder after setting the requested encoder seed

Evaluation outputs and resume keys must include encoder protocol identity. At minimum, result rows or output filenames must separate:

- `encoder_mode`
- `encoder_seed` for random-stress mode
- `encoder_artifact` for matched mode
- `encoder_sha256`
- `policy_checkpoint`
- `policy_sha256`
- `protocol`

Matched evaluation must not resume from or overwrite legacy/random-stress rows keyed only by `(method, seed, split)`.

## G2 Audit Split

The existing G2 result should be recorded as:

```text
G2 random-eval-encoder stress audit
```

The corrected audit should be recorded as:

```text
G2 matched-encoder checkpoint audit
```

The corrected G2 should first rerun `m7_pta seed42` on `ood_elastoplastic` using the matched encoder. After the one-seed audit is verified, additional M7 policy seeds can be trained and evaluated under the same protocol.

The matched encoder is part of the run identity. Performance should be interpreted over policy-plus-encoder seeds, not policy-only seeds. The seed42 corrected G2 is a protocol check; paper-facing matched M7 claims require multi-seed matched evaluations with uncertainty.

Old policy-only DLC jobs should be labeled:

```text
policy-only legacy diagnostics
```

They may be used for policy-seed variance diagnostics only when the evaluation encoder protocol is clearly stated. They must not be mixed with matched-encoder results in headline paper tables.

## Artifact Registry Changes

The artifact registry should treat encoder sidecars as first-class artifacts. Manifest entries should use `kind=belief_encoder` for `belief_encoder.pt` and `kind=belief_encoder_metadata` for metadata JSON.

For M7 matched-encoder requirements, a policy checkpoint is valid only if the paired encoder sidecar exists, loads with Torch on CPU, and metadata matches the policy checkpoint metadata.

Add a dedicated requirement set, for example `g2-matched-encoder`, that requires the full policy/encoder bundle for the checkpoint used in corrected G2:

```text
best/best_model.zip
best/best_model.json
best/belief_encoder.pt
best/belief_encoder_metadata.json
```

Registry verification must be type-specific:

- `kind=policy_checkpoint`: load with SB3 `PPO.load`.
- `kind=policy_metadata`: validate JSON schema and policy-to-encoder hash links.
- `kind=belief_encoder`: load with `torch.load(..., map_location="cpu")`, validate format version, and instantiate/load `LatentBeliefEncoder`.
- `kind=belief_encoder_metadata`: validate JSON schema, policy pairing fields, and SHA256 links.

The registry should archive only claim-critical artifacts by default:

- final policy checkpoint and metadata
- best policy checkpoint and metadata when used for evaluation
- paired `belief_encoder.pt`
- paired `belief_encoder_metadata.json`
- minimal logs and result JSON/CSV needed for claims

Intermediate checkpoints should not be registered unless a claim or debugging need requires them. Current observed storage is small: checkpoint zips are about 1.8 MB each, and a full M7 recovery archive with bundle is about 44 MB. Encoder sidecars are expected to be much smaller than the PPO checkpoint.

Runtime cleanup should keep CPFS bounded: retain final/best matched artifacts, worker records, minimal logs, and claim result files; do not register periodic 50k checkpoints by default; remove temporary restore/register directories only after manifest hashes and bundle restore validation pass. Legacy policy-only runs should keep a lightweight record label so cleanup cannot accidentally promote them into matched evidence.

## DLC Rollout Plan

After implementation, do not immediately launch a large matrix. Use a staged rollout:

1. Submit one smoke or micro DLC job that exercises M7 save/load artifact creation.
2. Verify job image is `pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang`.
3. Verify worker record `exit_code=0`.
4. Verify policy checkpoint loads with SB3.
5. Verify encoder sidecar loads with Torch and metadata matches the policy.
6. Verify normal matched eval fails on a policy-only legacy checkpoint and random-stress runs only when explicitly requested.
7. Register and bundle the run with the artifact registry.
8. Restore the bundle into an empty target and verify the restored policy-plus-encoder bundle can be used for matched evaluation.
9. Submit one full `m7_pta seed42` matched-encoder job.
10. Rerun G2 matched-encoder checkpoint audit.
11. If the one-seed path is clean, submit additional M7 seeds under the matched protocol.

The currently running 12 policy-only DLC jobs should continue. They should be documented as legacy and not counted as matched-encoder evidence.

## Testing Strategy

Use TDD with lightweight tests before implementation. Tests should avoid Genesis launches and PPO training.

Add focused tests for:

- Saving an M7 encoder sidecar from a wrapper-like object with `_belief_encoder.state_dict()`.
- Loading an encoder sidecar into a new `LatentBeliefEncoder` and reproducing state dict tensors exactly.
- Rejecting a full M7 evaluation when the policy checkpoint exists but the encoder artifact is missing.
- Requiring an explicit random-stress mode before a fresh encoder is allowed.
- Passing a loaded encoder into `ProbePhaseWrapper` during eval environment construction.
- Extending artifact registry requirements so M7 matched-encoder evidence includes both policy and encoder artifacts.
- Failing loudly on incompatible encoder metadata, such as mismatched `latent_dim`, `n_probes`, `ablation`, or seed.

Name these test targets in the implementation plan:

- `tests/test_m7_encoder_artifacts.py` for encoder sidecar save/load, tensor equality, and metadata validation.
- `tests/test_run_ood_eval_v2.py` for encoder mode defaults, sidecar resolution, policy-only failure, random-stress opt-in, and no-probe/no-belief zero-z behavior.
- `tests/test_pre_submission_audit.py` for separating random-eval-encoder stress from matched-encoder audit labels and outputs.
- `tests/test_artifact_registry.py` for the matched-M7 requirement set and type-specific validation.
- `tests/test_dlc_shell_contract.py` and `tests/test_dlc_submit_jobs.py` only if DLC command construction gains matched-encoder audit modes.

Edge cases to test include best-vs-final sidecar resolution, policy paths with and without `.zip`, missing policy metadata, missing encoder metadata, corrupt `belief_encoder.pt`, wrong state-dict shape, mismatched `paired_policy_sha256`, and legacy policy-only checkpoints that must not silently create a fresh encoder.

Do not add tests that run Genesis, real DLC, PPO training, or numeric transfer assertions. Those belong to post-implementation verification, not unit-level protocol checks.

## Acceptance Criteria

- A new M7 training run writes paired policy and encoder artifacts for final checkpoints.
- A best checkpoint used for evaluation has a paired encoder sidecar.
- Normal M7 evaluation loads the matched encoder by default.
- Normal M7 evaluation fails on policy-only legacy checkpoints unless explicit random-stress mode is requested.
- Artifact registration captures encoder sidecars and validates their metadata.
- Documentation separates old random-eval-encoder stress evidence from corrected matched-encoder evidence.
- The first new matched-encoder DLC job is verified end-to-end before launching a larger batch.
- Paper-facing matched M7 claims wait for multi-seed matched evaluations with uncertainty, not just the seed42 corrected G2 protocol check.
