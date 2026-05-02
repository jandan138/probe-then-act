# Six-Seed DLC Training Design

Date: 2026-05-02

## Purpose

Submit additional DLC training jobs so the learned methods used in the paper-facing comparisons can be extended from the current policy seeds `0, 1, 42` to six policy seeds `0, 1, 2, 3, 4, 42`.

## Scope

Submit new training jobs for seeds `2`, `3`, and `4` for these methods:

- `m1_reactive`
- `m7_pta`
- `m7_noprobe`
- `m7_nobelief`

This is 12 new single-GPU DLC jobs. The existing seeds `0`, `1`, and `42` are not resubmitted.

## Execution Path

Use the existing CPFS DLC runtime tree:

```text
/cpfs/shared/simulation/zhuzihou/dev/probe-then-act
```

Submit through:

```text
pta/scripts/dlc/launch_job.sh
```

Run each job as `custom env PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 ...` to preserve the successful EGL worker path used by G0/G1/G2 and recovery jobs.

Export the verified image explicitly before submission:

```text
DLC_IMAGE=pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang
```

Do not rely on the `launch_job.sh` default image for these training jobs.

2026-05-02 correction: `launch_job.sh` has also been updated so its default image is the verified Genesis/PTA image. Explicit export is still recommended because it makes the submission environment auditable in shell history and dry-run output.

## Job Matrix

| Method | Script | Extra seeds | Jobs |
|---|---|---:|---:|
| `m1_reactive` | `pta/scripts/train_baselines.py --method m1` | `2, 3, 4` | 3 |
| `m7_pta` | `pta/scripts/train_m7.py` | `2, 3, 4` | 3 |
| `m7_noprobe` | `pta/scripts/train_m7.py --ablation no_probe` | `2, 3, 4` | 3 |
| `m7_nobelief` | `pta/scripts/train_m7.py --ablation no_belief` | `2, 3, 4` | 3 |

Each job trains for `500000` timesteps with `--residual-scale 0.05`, `--horizon 500`, and `--eval-freq 50000`. M7 variants also use `--latent-dim 16 --n-probes 3`.

## Records

After submission, update `docs/30_records/SEED_SENSITIVITY_STATUS.md` with the submitted job names and DLC job IDs. If any jobs were submitted with the wrong image, mark them as stopped/superseded and record the verified-image replacements. Do not claim training completion until worker records and checkpoints prove completion.

## Verification

Before submission, run a dry-run or command inspection to confirm all 12 intended jobs are present and no existing seeds are duplicated. After submission, verify that the DLC CLI output includes one job ID per intended job and that `dlc get job` reports the verified `mahaoxiang` image for every replacement job.
