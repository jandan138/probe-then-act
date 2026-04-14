# Probe-Then-Act Viewer Skill Design

## Overview

This design defines a new reusable skill for launching interactive Genesis viewer scenes from the `probe-then-act` repository on this WSL machine.

The skill is not tied to a single scene. It standardizes the safe launch workflow for any repo scene script that should run under WSLg with the validated local Genesis path.

## Goal

Provide one skill that future sessions can use to:

- launch the `probe-then-act` scoop/carry viewer scene reliably;
- launch other `probe-then-act` interactive Genesis scenes through the same stable path;
- avoid the known unstable viewer path caused by non-local Genesis installs or default WSLg hardware OpenGL.

## Non-Goals

- Do not replace the existing `local-genesis-wslg-viewer` skill.
- Do not cover headless video rendering.
- Do not guarantee that arbitrary external Genesis scripts work.
- Do not encode scene-specific robotics logic into the skill body.

## Proposed Skill

### Name

`local-probe-then-act-viewer`

### Purpose

This skill will be the repository-specific companion to `local-genesis-wslg-viewer`.

`local-genesis-wslg-viewer` solves the machine-level WSLg launch problem.

`local-probe-then-act-viewer` will solve the repository-level problem:

- which scene scripts in this repo are valid interactive viewer entrypoints;
- how to launch them through the safe local Genesis wrapper;
- what success evidence to check after launch.

## Core Workflow

The skill will instruct future sessions to follow this order:

1. Identify the intended scene entry script in `probe-then-act`.
2. Prefer a validated repo entrypoint if one already exists.
3. Launch it through `/home/zhuzihou/dev/Genesis/scripts/run_wslg_genesis_viewer.sh`.
4. Ensure execution resolves `genesis.__file__` to `/home/zhuzihou/dev/Genesis/genesis/__init__.py`.
5. Verify success with process evidence plus `Viewer created.` and an X window entry.

The skill should explicitly forbid launching repo viewer scenes through the previously failing `isaacsim-agent` Genesis installation when an interactive WSLg window is required.

## Scene Script Contract

To be considered compatible with this skill, a repo scene script must satisfy all of the following:

1. It is a standalone Python entrypoint that can be executed directly.
2. It can run with `PYTHONPATH` including both `/home/zhuzihou/dev/Genesis` and the `probe-then-act` repo paths.
3. It is responsible for creating a `gs.Scene(..., show_viewer=True)` path or an equivalent monkey-patched viewer path.
4. It keeps the viewer alive long enough for manual inspection instead of immediately exiting.
5. It does not require the old unstable viewer chain through the wrong Genesis package.

The first documented compatible example will be the current scoop/carry viewer script:

- `/home/zhuzihou/dev/probe-then-act/pta/scripts/debug_viewers/interactive_bowl_viewer.py`

The skill may list additional validated scene entrypoints later, but examples should remain examples rather than hardcoded requirements.

## Launch Interface

The preferred launch pattern in the skill should be:

```bash
PYTHONPATH=/home/zhuzihou/dev/probe-then-act:/home/zhuzihou/dev/probe-then-act/pta/scripts \
  /home/zhuzihou/dev/Genesis/scripts/run_wslg_genesis_viewer.sh \
  /home/zhuzihou/dev/probe-then-act/<scene-script>.py
```

For the scoop/carry scene, the documented example will point to:

```bash
PYTHONPATH=/home/zhuzihou/dev/probe-then-act:/home/zhuzihou/dev/probe-then-act/pta/scripts \
  /home/zhuzihou/dev/Genesis/scripts/run_wslg_genesis_viewer.sh \
  /home/zhuzihou/dev/probe-then-act/pta/scripts/debug_viewers/interactive_bowl_viewer.py
```

## Verification Requirements

The skill should require fresh verification after every launch attempt.

Successful launch evidence should include:

- wrapper output showing local Genesis paths;
- `target_execution_genesis_file=/home/zhuzihou/dev/Genesis/genesis/__init__.py`;
- Genesis log line `Viewer created.`;
- a live Python process running from `/home/zhuzihou/dev/Genesis/.venv/bin/python`;
- an X window entry such as `Genesis 0.4.4` in `xwininfo -root -tree`.

Failure evidence should distinguish:

- wrapper rejection before launch;
- scene build stall before viewer creation;
- viewer creation failure;
- black/unstable window after creation.

## Relationship to Existing Skill

The new skill should build on `local-genesis-wslg-viewer` rather than duplicate it.

Expected structure:

- machine-level viewer workaround remains in `local-genesis-wslg-viewer`;
- repo-specific scene-launch guidance lives in `local-probe-then-act-viewer`;
- the new skill references the existing wrapper as the required launch path.

## Minimal Deliverables

Implementation should create only what is required for this design:

1. a new skill directory under `~/.codex/skills/local-probe-then-act-viewer/`;
2. a `SKILL.md` describing triggers, workflow, examples, and verification;
3. only supporting files if the skill truly needs a helper script beyond the existing Genesis wrapper.

The default assumption should be: no new helper script unless the skill cannot stay clear and reusable without one.

## Risks

- The scoop/carry script is currently under `results/scoop_debug/`, which is functional but not an ideal long-term home for a reusable entrypoint.
- Additional scenes may need small repo-side cleanup before they satisfy the scene-script contract.
- If local Genesis viewer behavior changes, the skill must continue to defer to the wrapper as the source of truth.

## Recommendation

Implement the skill as a thin repo-specific layer over the already validated local Genesis WSLg wrapper.

Keep the skill focused on launch discipline, valid scene entrypoints, and verification evidence. Do not turn it into a large catalog or a generalized Genesis tutorial.
