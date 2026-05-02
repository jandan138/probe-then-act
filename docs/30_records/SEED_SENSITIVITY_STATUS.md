# Seed Sensitivity Status

Date: 2026-05-02

## Purpose

This record separates three questions that are easy to mix together:

- How many policy seeds are already represented in the current OOD table?
- Which pre-submission gates were actually run after the DLC/checkpoint recovery work?
- Which results currently show seed sensitivity, and which methods are not yet diagnosable from the available evidence?

The record is intentionally conservative. It documents measured sensitivity and observed high variance; it does not claim a root cause.

## Evidence Sources

- Main per-seed OOD table: `/cpfs/shared/simulation/zhuzihou/dev/probe-then-act/results/ood_eval_per_seed.csv`
- G1 probe-integrity outputs: `/cpfs/shared/simulation/zhuzihou/dev/probe-then-act/results/presub/audit_probe_*_seed123.json`
- G2 encoder-sensitivity output: `/cpfs/shared/simulation/zhuzihou/dev/probe-then-act/results/presub/audit_encoder_m7_pta_s42_ood_elastoplastic.json`
- New DLC worker records: `/cpfs/shared/simulation/zhuzihou/dev/probe-then-act/results/dlc/runs/*.json`
- Durable artifact manifests: `/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/20260501/*/artifact_manifest.json`

## Current Seed Coverage

The current paper-facing OOD matrix has 65 per-seed rows. Five splits are present for each listed seed: `id_sand`, `ood_elastoplastic`, `ood_snow`, `ood_sand_hard`, and `ood_sand_soft`.

| Method | Policy seeds present | Rows | Interpretation |
|---|---:|---:|---|
| `m1_reactive` | `0, 1, 42` | 15 | Three-policy-seed baseline evidence. |
| `m7_pta` | `0, 1, 42` | 15 | Three-policy-seed PTA evidence. |
| `m7_noprobe` | `0, 1, 42` | 15 | Three-policy-seed ablation evidence. |
| `m7_nobelief` | `0, 1, 42` | 15 | Three-policy-seed ablation evidence. |
| `m8_teacher` | `42` | 5 | Single teacher seed only; seed sensitivity cannot be assessed. |

The active six-seed strengthening step adds policy seeds `2`, `3`, and `4` for `m1_reactive`, `m7_pta`, `m7_noprobe`, and `m7_nobelief`. The first submission batch accidentally used the launcher default PyTorch image and was stopped. The replacement batch uses the previously verified Genesis/PTA image `pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang`. These jobs are not completion evidence until worker records and checkpoints verify success.

## Pre-Submission Gate Runs

| Gate | What ran | Seeds | Status |
|---|---|---:|---|
| G0 | DLC smoke job | none | Passed. |
| G1 | Probe-integrity audit on five splits | audit seed `123` | Passed for the required elastoplastic criterion. |
| Recovery | Recovered `m7_pta` policy checkpoint | policy seed `42` | Passed; final checkpoint reached about 500k timesteps. |
| G2 | Encoder-sensitivity audit on `m7_pta` / `ood_elastoplastic` | policy seed `42`; encoder seeds `11, 22, 33` | Failed. |
| G3 | Six-seed extension training for main methods and ablations | policy seeds `2, 3, 4` | Submitted; completion not verified. |
| G4 | Extra-seed OOD evaluation | policy seeds `2, 3, 4` | Not submitted; wait for checkpoint verification. |

## Six-Seed DLC Submissions - 2026-05-02

These single-GPU DLC jobs were submitted from `/cpfs/shared/simulation/zhuzihou/dev/probe-then-act` with `PYOPENGL_PLATFORM=egl`, `EGL_DEVICE_ID=0`, `DLC_GPU_COUNT=1`, and the CPFS Genesis virtualenv. Status here means submission/status-check evidence only.

### Superseded wrong-image batch

The following jobs used the launcher default image `dsw-registry-vpc.cn-beijing.cr.aliyuncs.com/pai-training-algorithm/pytorch:py311-cu126`. They were stopped with `dlc stop job --force --quiet` and verified as `Status=Stopped`, `ReasonCode=StoppedByUser`.

| Method | Seed | Job name | DLC JobId | Status |
|---|---:|---|---|---|
| `m1_reactive` | `2` | `pta_6seed_m1_s2_0_1` | `dlc1fnz66irdtt53` | Stopped; superseded |
| `m1_reactive` | `3` | `pta_6seed_m1_s3_0_1` | `dlc1grxjpme48f42` | Stopped; superseded |
| `m1_reactive` | `4` | `pta_6seed_m1_s4_0_1` | `dlc1hlwbuybmxzoy` | Stopped; superseded |
| `m7_pta` | `2` | `pta_6seed_m7_s2_0_1` | `dlc1ifv40ataq681` | Stopped; superseded |
| `m7_pta` | `3` | `pta_6seed_m7_s3_0_1` | `dlc1j9tw5ml95364` | Stopped; superseded |
| `m7_pta` | `4` | `pta_6seed_m7_s4_0_1` | `dlc1kds9oqpw3tlk` | Stopped; superseded |
| `m7_noprobe` | `2` | `pta_6seed_noprobe_s2_0_1` | `dlc1l7r1u28m76xd` | Stopped; superseded |
| `m7_noprobe` | `3` | `pta_6seed_noprobe_s3_0_1` | `dlc1m1ptzedr0xzf` | Stopped; superseded |
| `m7_noprobe` | `4` | `pta_6seed_noprobe_s4_0_1` | `dlc1mlp0qyumfoza` | Stopped; superseded |
| `m7_nobelief` | `2` | `pta_6seed_nobelief_s2_0_1` | `dlc1npnea2btsko8` | Stopped; superseded |
| `m7_nobelief` | `3` | `pta_6seed_nobelief_s3_0_1` | `dlc1ojm6fe71khii` | Stopped; superseded |
| `m7_nobelief` | `4` | `pta_6seed_nobelief_s4_0_1` | `dlc1pnkjyijrgg2e` | Stopped; superseded |

### Verified-image replacement batch

The following jobs use `pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang`. Fresh `dlc get job` checks verified this exact image for all 12 replacements.

| Method | Seed | Job name | DLC JobId | Latest checked status |
|---|---:|---|---|---|
| `m1_reactive` | `2` | `pta_6seed_m1_s2_verified_0_1` | `dlcinh3wqgkl9znn` | `Running` |
| `m1_reactive` | `3` | `pta_6seed_m1_s3_verified_0_1` | `dlcjrfhfuj9tnyua` | `Running` |
| `m1_reactive` | `4` | `pta_6seed_m1_s4_verified_0_1` | `dlckle9l6gvky1u2` | `Running` |
| `m7_pta` | `2` | `pta_6seed_m7_s2_verified_0_1` | `dlclfd1qi6ighksj` | `Running` |
| `m7_pta` | `3` | `pta_6seed_m7_s3_verified_0_1` | `dlcm9btvu85dbdnf` | `Running` |
| `m7_pta` | `4` | `pta_6seed_m7_s4_verified_0_1` | `dlcn3am166829kst` | `Running` |
| `m7_noprobe` | `2` | `pta_6seed_noprobe_s2_verified_0_1` | `dlcnx9e6iz405ixz` | `Running` |
| `m7_noprobe` | `3` | `pta_6seed_noprobe_s3_verified_0_1` | `dlcor86buummj2q3` | `Running` |
| `m7_noprobe` | `4` | `pta_6seed_noprobe_s4_verified_0_1` | `dlcpb7d3eoe4xt9a` | `Running` |
| `m7_nobelief` | `2` | `pta_6seed_nobelief_s2_verified_0_1` | `dlcq5658qlnkam7v` | `Running` |
| `m7_nobelief` | `3` | `pta_6seed_nobelief_s3_verified_0_1` | `dlcqz4xe2nb4zr57` | `Running` |
| `m7_nobelief` | `4` | `pta_6seed_nobelief_s4_verified_0_1` | `dlcrt3pje7za55l2` | `Running` |

### ETA snapshot

Status checked at `2026-05-02T03:04:20Z`: all 12 replacement jobs were `Running`, and every job still reported the verified Genesis/PTA image. Historical anchors:

- Full `m7_pta seed42` recovery job `dlc1hn82yye94ojd` ran from `2026-05-01T11:30:21Z` to `2026-05-01T21:09:15Z`, about 9h39m for 500k timesteps.
- Earlier 400k-to-500k ablation resume jobs took about 1h50m-2h08m for the final 100k, consistent with about 9h-11h for a full 500k run.

ETA should therefore be treated as a range, not a completion claim. For the 12 running jobs, expect completion roughly `2026-05-02T12:00Z`-`2026-05-02T14:00Z` if they continue at the historical rate.

Next evidence gate: do not mark any row complete until `results/dlc/runs/*.json` shows `exit_code=0` and the corresponding checkpoint loads with SB3.

## Confirmed Seed Sensitivity

### Legacy random-stress encoder sensitivity: confirmed for old `m7_pta` diagnostic only

The old `random_eval_encoder_stress` G2 diagnostic fixed the policy checkpoint at `m7_pta` policy seed `42` and varied only the freshly constructed evaluation encoder seed. The gate failed because transfer changed by about 66 percentage points. This confirms sensitivity to random evaluation encoders in the legacy diagnostic protocol; it is not matched M7 headline evidence.

| Policy seed | Encoder seed | Episodes | Mean transfer |
|---:|---:|---:|---:|
| `42` | `11` | 3 | `0.7027` |
| `42` | `22` | 3 | `0.3101` |
| `42` | `33` | 3 | `0.9701` |

Measured legacy random-stress G2 range: `65.9958` percentage points. This is the strongest confirmed diagnostic seed-sensitivity finding because the audit isolates evaluation encoder seed while holding policy seed, checkpoint, split, and method fixed, but it does not establish corrected matched-encoder M7 performance.

## M7 Matched-Encoder Protocol Update

The old G2 result is now classified as a `random_eval_encoder_stress` diagnostic. It deliberately paired a fixed `m7_pta seed42` policy checkpoint with freshly initialized evaluation encoders, so it is not claim-bearing matched M7 evidence.

The 12 verified-image six-seed replacement jobs submitted before encoder persistence are `policy-only legacy diagnostics`. They can support policy-seed variance analysis when the encoder protocol is stated, but they must not be mixed into matched-encoder M7 headline claims.

Future full `m7_pta` claim artifacts must preserve the matched policy-plus-encoder bundle:

- `best_model.zip`
- `best_model.json`
- `belief_encoder.pt`
- `belief_encoder_metadata.json`

### Policy-seed transfer variance: observed across learned methods

The existing 3-policy-seed OOD table also shows large `ood_elastoplastic` transfer ranges across policy seeds `0, 1, 42`. This is not the same as the G2 encoder audit, because training seed changes both the learned policy and training trajectory. It is still relevant to reviewer concerns about whether three seeds are sufficient evidence.

| Method | Policy seeds | `ood_elastoplastic` mean transfer values | Range |
|---|---:|---|---:|
| `m1_reactive` | `0, 1, 42` | `0.4052`, `0.9748`, `0.0000` | `97.48` pp |
| `m7_pta` | `0, 1, 42` | `0.0000`, `0.9774`, `0.8439` | `97.74` pp |
| `m7_noprobe` | `0, 1, 42` | `0.9931`, `0.0000`, `0.0012` | `99.31` pp |
| `m7_nobelief` | `0, 1, 42` | `0.9770`, `0.0000`, `0.4114` | `97.70` pp |

From the current evidence, the trainings/evaluations with clear policy-seed variance on `ood_elastoplastic` are therefore:

- `m1_reactive`
- `m7_pta`
- `m7_noprobe`
- `m7_nobelief`

The result should be described as high policy-seed variance, not as a proven mechanism-level failure. More seeds would estimate the variance better; they would not by themselves explain it.

## Not Yet Confirmed As Seed Sensitivity

- `m8_teacher`: only policy seed `42` is present, so there is no seed-sensitivity measurement.
- Extra policy seeds `2`, `3`, and `4` for `m1_reactive`, `m7_pta`, `m7_noprobe`, and `m7_nobelief`: submitted but not yet completed, loaded, or evaluated.
- G2-style encoder-seed sensitivity for `m1_reactive`, `m7_noprobe`, `m7_nobelief`, or `m8_teacher`: not measured by the current G2 audit.

## Implication For G3/G4

The immediate blocker is not DLC reliability. G0, G1, checkpoint recovery, artifact registration, and worker records indicate that the execution path is functioning.

The blocker is evidence quality: G2 found that `m7_pta` evaluation can swing strongly when only encoder seed changes, and the existing OOD table already shows large policy-seed transfer ranges on `ood_elastoplastic` across multiple learned methods.

The six-seed training jobs were intentionally submitted despite the G2 failure to collect candidate seeds for later analysis. Do not submit G4 evaluation or strengthen paper claims until the 12 worker records finish successfully and the resulting checkpoints load.
