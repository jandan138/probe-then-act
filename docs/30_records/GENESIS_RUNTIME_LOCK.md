# Genesis Runtime Lock

Date: 2026-04-26

The experiment should use the personal Genesis fork, not a random pip package or the upstream branch tip.

## Locked Runtime

- Remote: `git@github.com:jandan138/Genesis.git`
- Branch: `main`
- Commit: `0f82aa73f8c51a246cabdd4e5f3919224a62956b`
- Commit message: `feat: add WSLg and headless runtime support`
- Upstream remote retained locally as: `https://github.com/Genesis-Embodied-AI/Genesis.git`
- Upstream base visible in this history: `0e095af`

## Why This Fork Is Required

This fork contains the local runtime changes used by the probe experiments:

- WSLg interactive viewer launch wrapper and probe/fallback logic.
- Linux headless PyOpenGL platform defaulting for GPU and CPU cases.
- Offscreen rasterizer fallback from failed EGL to OSMesa where applicable.
- Panda bowl MJCF asset changes, including the high-wall bowl variant.
- Camera/recording and offscreen-rendering fixes that are covered by focused tests.

## Reproduce Checkout

```bash
git clone git@github.com:jandan138/Genesis.git
cd Genesis
git checkout 0f82aa73f8c51a246cabdd4e5f3919224a62956b
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For probe repo commands, export this Genesis path before running training/eval:

```bash
export GENESIS_ROOT=/cpfs/shared/simulation/zhuzihou/dev/Genesis
export GENESIS_VENV=/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv
export PYTHONPATH="${GENESIS_ROOT}:${PYTHONPATH:-}"
```

## Verification Evidence

Focused runtime tests run locally before pushing:

```bash
source /home/zhuzihou/dev/Genesis/.venv/bin/activate
pytest tests/test_interactive_viewer_env.py \
  tests/test_wslg_viewer_wrapper.py \
  tests/test_render.py::test_offscreen_rasterizer_falls_back_to_next_platform \
  tests/test_recorders.py::test_camera_recording_captures_one_frame_per_scene_step \
  -q
```

Observed result: `7 passed in 25.66s`.

`git diff --check` also completed with no output before the Genesis commit.

The broader render/recorder command `pytest tests/test_interactive_viewer_env.py tests/test_wslg_viewer_wrapper.py tests/test_render.py tests/test_recorders.py -q` was intentionally stopped after it began producing multiple failures/errors in heavier render cases. Treat that as an environment-sensitive broad suite, not as a passed gate.

## Artifact Boundary

Genesis source is in Git. Probe checkpoints are separate artifacts because checkpoints are not stored in normal Git.
