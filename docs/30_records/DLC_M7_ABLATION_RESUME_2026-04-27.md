# DLC M7 Ablation Resume Record - 2026-04-27

## Context

Five M7 ablation DLC jobs were submitted on 2026-04-26 with the verified GPU image:

`pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang`

The jobs used `PYOPENGL_PLATFORM=egl` and `EGL_DEVICE_ID=0` after the earlier OSMesa failure was diagnosed.

## Failed Jobs

| Variant | Seed | DLC JobId | Last logged timestep | Durable checkpoint |
|---|---:|---|---:|---:|
| `no_probe` | `0` | `dlc1rdewp6njx537` | `428544` | `400000` |
| `no_probe` | `1` | `dlc1rxe3gq1g32gl` | `448512` | `400000` |
| `no_belief` | `42` | `dlc1s7douizhclbl` | `426496` | `400000` |
| `no_belief` | `0` | `dlc1shda8a1otiry` | `421376` | `400000` |
| `no_belief` | `1` | `dlc1t1cgzu86d8m4` | `419840` | `400000` |

## Root Cause

All five jobs failed with:

```text
OSError: [Errno 28] No space left on device
```

The failure occurred while TensorBoard or eval callback output was writing under the shared project directory. The failed jobs were otherwise training normally and had passed the previous OpenGL initialization issue.

## Resume State

The stable resume point is the latest checkpoint written before the storage failure:

```text
checkpoints/m7_pta_noprobe_seed0/m7_pta_400000_steps.zip
checkpoints/m7_pta_noprobe_seed1/m7_pta_400000_steps.zip
checkpoints/m7_pta_nobelief_seed42/m7_pta_400000_steps.zip
checkpoints/m7_pta_nobelief_seed0/m7_pta_400000_steps.zip
checkpoints/m7_pta_nobelief_seed1/m7_pta_400000_steps.zip
```

Each file was verified with `PPO.load(..., device="cpu")`, and each loaded model reported `num_timesteps == 400000`.

The extra progress after 400k that appears in logs is not reliably resumable because no later SB3 checkpoint was written.

## Resume Strategy

Do not resubmit `run_task.sh train_ablation ...` directly. That path recreates PPO from scratch through `train_m7.py`.

Resume with `pta/scripts/resume_m7.py`, which:

- loads the 400k SB3 checkpoint,
- computes the remaining target as `500000 - model.num_timesteps`,
- calls `model.learn(total_timesteps=remaining, reset_num_timesteps=False)`,
- disables TensorBoard writes,
- keeps `PYOPENGL_PLATFORM=egl`.

Expected remaining work per job is about 100k timesteps.

## Local Resume Verification

Implemented resume entrypoint:

```text
pta/scripts/resume_m7.py
```

Test coverage:

```text
/shared/smartbot/zhuzihou/envs/pta-genesis-py310-cu128/bin/python -m pytest tests/test_resume_m7.py -q
5 passed in 0.12s
```

Dry-run loading was performed for all five 400k checkpoints. Each reported:

```text
Loaded checkpoint steps: 400,000
Remaining train steps:   100,000
```

Storage state before resubmission:

```text
/shared/smartbot: 600T size, 522T used, 79T available, 87% used
inodes: 4.3G total, 1.8G used, 2.6G free, 41% used
```

Relevant project usage before resubmission:

```text
17M checkpoints/m7_pta_noprobe_seed0
17M checkpoints/m7_pta_noprobe_seed1
17M checkpoints/m7_pta_nobelief_seed42
17M checkpoints/m7_pta_nobelief_seed0
17M checkpoints/m7_pta_nobelief_seed1
2.6M logs
2.1M results/dlc
```

## Resume Jobs Submitted

Submitted at `2026-04-27T12:56:51Z` with:

- image: `pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang`
- workspace: `270969`
- resource: `quota1r947pmazvk`
- data sources: `d-mzps5b7joy2axmqpa8,d-d49o5g0h2818sw8j1g,d-8wz4emfs21s5ajs9oz`
- resources: `1 GPU`, `14 CPU`, `100Gi` memory, `100Gi` shared memory
- env: `PTA_CODE_ROOT=/shared/smartbot/zhuzihou/dev/probe-then-act`, `GENESIS_ROOT=/shared/smartbot/zhuzihou/dev/Genesis`, `GENESIS_VENV=/shared/smartbot/zhuzihou/envs/pta-genesis-py310-cu128`, `PYOPENGL_PLATFORM=egl`, `EGL_DEVICE_ID=0`
- resume command: `bash .../run_task.sh custom .../python -u pta/scripts/resume_m7.py ... --target-timesteps 500000 --save-freq 50000 --eval-freq 0`

| Variant | Seed | Resume checkpoint | New DLC JobId | Initial status |
|---|---:|---|---|---|
| `no_probe` | `0` | `checkpoints/m7_pta_noprobe_seed0/m7_pta_400000_steps.zip` | `dlc14uard6mq7vsw` | `Queuing` / `JobEnqueued` |
| `no_probe` | `1` | `checkpoints/m7_pta_noprobe_seed1/m7_pta_400000_steps.zip` | `dlc15e9y4qc12v0j` | `Queuing` / `JobEnqueued` |
| `no_belief` | `42` | `checkpoints/m7_pta_nobelief_seed42/m7_pta_400000_steps.zip` | `dlc15o9jiielquzg` | `Queuing` / `JobEnqueued` |
| `no_belief` | `0` | `checkpoints/m7_pta_nobelief_seed0/m7_pta_400000_steps.zip` | `dlc15y94wa65t7c8` | `Queuing` / `JobEnqueued` |
| `no_belief` | `1` | `checkpoints/m7_pta_nobelief_seed1/m7_pta_400000_steps.zip` | `dlc16i8bnuxurvgb` | `Queuing` / `JobEnqueued` |

Use this DLC CLI form for status checks:

```bash
/cpfs/shared/simulation/zhuzihou/dev/usd-scene-physics-prep/dlc get job <JOB_ID> \
  --endpoint=pai-dlc.cn-beijing.aliyuncs.com \
  --region=cn-beijing
```
