# Cron ARIS Orchestrator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cron-driven local coordinator that monitors the post-hotfix research pipeline, advances one stage at a time, and hands completed experiment evidence into the repo's automatic research workflow.

**Architecture:** Add one idempotent Python coordinator under `pta/scripts/`, one shell wrapper for cron execution, and one helper script that prints or installs the crontab entry. The coordinator reads facts from processes and artifacts, persists a small JSON state file, launches at most one next-stage command, and writes a dedicated orchestration log.

**Tech Stack:** Python 3.11, bash, cron, pytest, existing PTA training/eval scripts, JSON state files.

---

### Task 1: Add coordinator state model and filesystem/process probes

**Files:**
- Create: `pta/scripts/cron_aris_orchestrator.py`
- Test: `tests/test_cron_aris_orchestrator.py`

- [ ] **Step 1: Write the failing tests for state detection helpers**

```python
from pathlib import Path


def test_detect_run_completion_from_final_checkpoint(tmp_path):
    from pta.scripts.cron_aris_orchestrator import detect_run_completion

    run_dir = tmp_path / "checkpoints" / "m1_reactive_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "scoop_transfer_teacher_final.zip").write_text("ok")

    result = detect_run_completion(
        checkpoint_dir=run_dir,
        final_name="scoop_transfer_teacher_final.zip",
    )

    assert result.completed is True
    assert result.final_checkpoint == run_dir / "scoop_transfer_teacher_final.zip"


def test_detect_active_process_from_ps_output():
    from pta.scripts.cron_aris_orchestrator import parse_ps_output

    output = "21934 40730 python pta/scripts/train_baselines.py --method m8 --seed 42"
    processes = parse_ps_output(output)

    assert processes[0]["pid"] == 21934
    assert "train_baselines.py" in processes[0]["cmd"]


def test_choose_latest_resume_checkpoint(tmp_path):
    from pta.scripts.cron_aris_orchestrator import choose_latest_resume_checkpoint

    ckpt_dir = tmp_path / "checkpoints" / "m8_teacher_seed42"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "scoop_transfer_teacher_50000_steps.zip").write_text("a")
    (ckpt_dir / "scoop_transfer_teacher_final.zip").write_text("b")

    result = choose_latest_resume_checkpoint(ckpt_dir)

    assert result.name == "scoop_transfer_teacher_final.zip"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cron_aris_orchestrator.py -v`
Expected: FAIL with `ModuleNotFoundError` or missing symbol errors for the new coordinator helpers.

- [ ] **Step 3: Write the minimal coordinator module skeleton and helpers**

```python
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CompletionStatus:
    completed: bool
    final_checkpoint: Path | None


def parse_ps_output(output: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_str, elapsed_str, cmd = line.split(maxsplit=2)
        rows.append({"pid": int(pid_str), "elapsed": int(elapsed_str), "cmd": cmd})
    return rows


def detect_run_completion(checkpoint_dir: Path, final_name: str) -> CompletionStatus:
    final_path = checkpoint_dir / final_name
    return CompletionStatus(completed=final_path.exists(), final_checkpoint=final_path if final_path.exists() else None)


def choose_latest_resume_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("*.zip"))
    if not candidates:
        return None
    finals = [p for p in candidates if p.name.endswith("_final.zip")]
    return finals[-1] if finals else candidates[-1]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cron_aris_orchestrator.py -v`
Expected: PASS for the three helper tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cron_aris_orchestrator.py pta/scripts/cron_aris_orchestrator.py
git commit -m "feat(automation): add coordinator state probes"
```

### Task 2: Add stage graph and next-step decision logic

**Files:**
- Modify: `pta/scripts/cron_aris_orchestrator.py`
- Modify: `tests/test_cron_aris_orchestrator.py`

- [ ] **Step 1: Write the failing tests for stage transitions**

```python
def test_decide_next_step_returns_running_when_process_active():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": True, "completed": False},
        "m1": {"running": False, "completed_seeds": []},
        "m7": {"running": False, "completed_seeds": []},
        "ood_eval": {"completed": False},
    }

    decision = decide_next_step(state)

    assert decision["action"] == "wait"
    assert decision["stage"] == "m8"


def test_decide_next_step_launches_first_missing_m1_seed():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": False, "completed": True},
        "m1": {"running": False, "completed_seeds": [42]},
        "m7": {"running": False, "completed_seeds": []},
        "ood_eval": {"completed": False},
    }

    decision = decide_next_step(state)

    assert decision == {"action": "launch_m1", "seed": 0}


def test_decide_next_step_hands_off_after_ood_eval():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": False, "completed": True},
        "m1": {"running": False, "completed_seeds": [42, 0, 1]},
        "m7": {"running": False, "completed_seeds": [42, 0, 1]},
        "ood_eval": {"completed": True},
    }

    decision = decide_next_step(state)

    assert decision == {"action": "handoff_aris"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cron_aris_orchestrator.py -v`
Expected: FAIL because `decide_next_step` does not exist yet.

- [ ] **Step 3: Implement minimal stage decision logic**

```python
M1_SEEDS = [42, 0, 1]
M7_SEEDS = [42, 0, 1]


def _first_missing_seed(done: list[int], expected: list[int]) -> int | None:
    for seed in expected:
        if seed not in done:
            return seed
    return None


def decide_next_step(state: dict) -> dict[str, object]:
    if state["m8"]["running"]:
        return {"action": "wait", "stage": "m8"}
    if not state["m8"]["completed"]:
        return {"action": "launch_m8_resume"}

    if state["m1"]["running"]:
        return {"action": "wait", "stage": "m1"}
    missing_m1 = _first_missing_seed(state["m1"]["completed_seeds"], M1_SEEDS)
    if missing_m1 is not None:
        return {"action": "launch_m1", "seed": missing_m1}

    if state["m7"]["running"]:
        return {"action": "wait", "stage": "m7"}
    missing_m7 = _first_missing_seed(state["m7"]["completed_seeds"], M7_SEEDS)
    if missing_m7 is not None:
        return {"action": "launch_m7", "seed": missing_m7}

    if not state["ood_eval"]["completed"]:
        return {"action": "run_ood_eval"}

    return {"action": "handoff_aris"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cron_aris_orchestrator.py -v`
Expected: PASS for stage transition tests and prior helper tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cron_aris_orchestrator.py pta/scripts/cron_aris_orchestrator.py
git commit -m "feat(automation): add sequential stage decisions"
```

### Task 3: Add command builders and launch execution helpers

**Files:**
- Modify: `pta/scripts/cron_aris_orchestrator.py`
- Modify: `tests/test_cron_aris_orchestrator.py`

- [ ] **Step 1: Write the failing tests for launch commands**

```python
def test_build_m1_command_uses_hotfix_scale():
    from pta.scripts.cron_aris_orchestrator import build_command

    command = build_command({"action": "launch_m1", "seed": 42})

    assert "train_baselines.py --method m1 --seed 42" in command
    assert "--residual-scale 0.05" in command


def test_build_m7_command_uses_hotfix_scale():
    from pta.scripts.cron_aris_orchestrator import build_command

    command = build_command({"action": "launch_m7", "seed": 0})

    assert "train_m7.py --seed 0" in command
    assert "--residual-scale 0.05" in command


def test_build_ood_eval_command_uses_corrected_script_defaults():
    from pta.scripts.cron_aris_orchestrator import build_command

    command = build_command({"action": "run_ood_eval"})

    assert "run_ood_eval_v2.py" in command
    assert "--residual-scale 0.05" in command
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cron_aris_orchestrator.py -v`
Expected: FAIL because `build_command` does not exist yet.

- [ ] **Step 3: Implement command builders and detached launcher**

```python
import subprocess


def build_command(decision: dict[str, object]) -> str:
    action = decision["action"]
    if action == "launch_m8_resume":
        return (
            "python pta/scripts/train_baselines.py --method m8 --seed 42 "
            "--total-timesteps 500000 --residual-scale 0.05 "
            "--resume-from checkpoints/m8_teacher_seed42/scoop_transfer_teacher_final.zip"
        )
    if action == "launch_m1":
        seed = decision["seed"]
        return (
            f"python pta/scripts/train_baselines.py --method m1 --seed {seed} "
            "--total-timesteps 500000 --residual-scale 0.05"
        )
    if action == "launch_m7":
        seed = decision["seed"]
        return (
            f"python pta/scripts/train_m7.py --seed {seed} "
            "--total-timesteps 500000 --residual-scale 0.05"
        )
    if action == "run_ood_eval":
        return "python pta/scripts/run_ood_eval_v2.py --residual-scale 0.05"
    raise ValueError(f"Unsupported action: {action}")


def launch_detached(command: str, log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        proc = subprocess.Popen(
            ["bash", "-lc", f"setsid -f {command} >> '{log_path}' 2>&1 < /dev/null"],
            cwd=cwd,
            stdout=handle,
            stderr=handle,
        )
    return proc.pid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cron_aris_orchestrator.py -v`
Expected: PASS for command-builder tests plus previous tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cron_aris_orchestrator.py pta/scripts/cron_aris_orchestrator.py
git commit -m "feat(automation): add orchestrator launch commands"
```

### Task 4: Add state persistence, coordinator log, and top-level CLI

**Files:**
- Modify: `pta/scripts/cron_aris_orchestrator.py`
- Modify: `tests/test_cron_aris_orchestrator.py`

- [ ] **Step 1: Write the failing tests for state persistence and idempotency**

```python
def test_save_and_load_state_roundtrip(tmp_path):
    from pta.scripts.cron_aris_orchestrator import load_state, save_state

    state_path = tmp_path / "aris_state.json"
    state = {"stage": "m1", "m1": {"completed_seeds": [42]}}
    save_state(state_path, state)

    loaded = load_state(state_path)

    assert loaded == state


def test_load_state_defaults_when_missing(tmp_path):
    from pta.scripts.cron_aris_orchestrator import load_state

    loaded = load_state(tmp_path / "missing.json")

    assert loaded["stage"] == "bootstrap"
    assert loaded["m1"]["completed_seeds"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cron_aris_orchestrator.py -v`
Expected: FAIL because `load_state` and `save_state` do not exist yet.

- [ ] **Step 3: Implement state file handling and CLI main**

```python
DEFAULT_STATE = {
    "stage": "bootstrap",
    "m8": {"running": False, "completed": False},
    "m1": {"running": False, "completed_seeds": []},
    "m7": {"running": False, "completed_seeds": []},
    "ood_eval": {"completed": False},
    "aris": {"ready": False, "blocked": False},
}


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return json.loads(json.dumps(DEFAULT_STATE))
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    state_path = project_root / "results" / "orchestration" / "aris_state.json"
    log_path = project_root / "logs" / "orchestration" / "cron_aris_orchestrator.log"
    state = load_state(state_path)
    append_log(log_path, "coordinator tick")
    save_state(state_path, state)
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cron_aris_orchestrator.py -v`
Expected: PASS for state roundtrip/default tests plus previous tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cron_aris_orchestrator.py pta/scripts/cron_aris_orchestrator.py
git commit -m "feat(automation): add orchestrator state persistence"
```

### Task 5: Wire real artifact discovery into coordinator state reconciliation

**Files:**
- Modify: `pta/scripts/cron_aris_orchestrator.py`
- Modify: `tests/test_cron_aris_orchestrator.py`

- [ ] **Step 1: Write the failing tests for artifact-based reconciliation**

```python
def test_reconcile_state_marks_m1_seed_complete_from_final_zip(tmp_path):
    from pta.scripts.cron_aris_orchestrator import reconcile_state

    project_root = tmp_path
    run_dir = project_root / "checkpoints" / "m1_reactive_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "scoop_transfer_teacher_final.zip").write_text("ok")

    state = reconcile_state(project_root=project_root, ps_output="")

    assert 42 in state["m1"]["completed_seeds"]


def test_reconcile_state_marks_ood_eval_complete_from_csv(tmp_path):
    from pta.scripts.cron_aris_orchestrator import reconcile_state

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "main_results.csv").write_text("method,split\n")

    state = reconcile_state(project_root=tmp_path, ps_output="")

    assert state["ood_eval"]["completed"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cron_aris_orchestrator.py -v`
Expected: FAIL because `reconcile_state` does not exist yet.

- [ ] **Step 3: Implement reconciliation from project artifacts and ps output**

```python
def reconcile_state(project_root: Path, ps_output: str) -> dict:
    state = json.loads(json.dumps(DEFAULT_STATE))
    processes = parse_ps_output(ps_output)
    cmds = [row["cmd"] for row in processes]

    state["m8"]["running"] = any("train_baselines.py --method m8 --seed 42" in cmd for cmd in cmds)
    state["m1"]["running"] = any("train_baselines.py --method m1" in cmd for cmd in cmds)
    state["m7"]["running"] = any("train_m7.py" in cmd for cmd in cmds)

    m8_dir = project_root / "checkpoints" / "m8_teacher_seed42"
    state["m8"]["completed"] = detect_run_completion(m8_dir, "scoop_transfer_teacher_final.zip").completed

    for seed in [42, 0, 1]:
        m1_dir = project_root / "checkpoints" / f"m1_reactive_seed{seed}"
        if detect_run_completion(m1_dir, "scoop_transfer_teacher_final.zip").completed:
            state["m1"]["completed_seeds"].append(seed)

        m7_dir = project_root / "checkpoints" / f"m7_pta_seed{seed}"
        if detect_run_completion(m7_dir, "m7_pta_final.zip").completed or detect_run_completion(m7_dir, "m7_pta_final").completed:
            state["m7"]["completed_seeds"].append(seed)

    state["ood_eval"]["completed"] = (project_root / "results" / "main_results.csv").exists()
    return state
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cron_aris_orchestrator.py -v`
Expected: PASS for reconciliation tests plus previous tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cron_aris_orchestrator.py pta/scripts/cron_aris_orchestrator.py
git commit -m "feat(automation): reconcile orchestrator state from artifacts"
```

### Task 6: Add ARIS handoff files and handoff decision execution

**Files:**
- Modify: `pta/scripts/cron_aris_orchestrator.py`
- Modify: `tests/test_cron_aris_orchestrator.py`

- [ ] **Step 1: Write the failing tests for ARIS handoff readiness**

```python
def test_write_handoff_files_creates_machine_and_human_records(tmp_path):
    from pta.scripts.cron_aris_orchestrator import write_handoff_files

    state = {
        "m8": {"completed": True},
        "m1": {"completed_seeds": [42, 0, 1]},
        "m7": {"completed_seeds": [42, 0, 1]},
        "ood_eval": {"completed": True},
    }

    records = write_handoff_files(tmp_path, state)

    assert records["json"].exists()
    assert records["summary"].exists()


def test_decide_next_step_stays_blocked_when_handoff_failed():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": False, "completed": True},
        "m1": {"running": False, "completed_seeds": [42, 0, 1]},
        "m7": {"running": False, "completed_seeds": [42, 0, 1]},
        "ood_eval": {"completed": True},
        "aris": {"blocked": True},
    }

    decision = decide_next_step(state)

    assert decision == {"action": "blocked", "stage": "aris"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cron_aris_orchestrator.py -v`
Expected: FAIL because `write_handoff_files` and blocked handling do not exist yet.

- [ ] **Step 3: Implement ARIS handoff file generation and blocked behavior**

```python
def write_handoff_files(project_root: Path, state: dict) -> dict[str, Path]:
    out_dir = project_root / "results" / "orchestration"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "aris_handoff_ready.json"
    summary_path = out_dir / "aris_handoff_summary.md"
    json_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    summary_path.write_text(
        "# ARIS Handoff Ready\n\n"
        f"- M8 complete: {state['m8']['completed']}\n"
        f"- M1 seeds: {state['m1']['completed_seeds']}\n"
        f"- M7 seeds: {state['m7']['completed_seeds']}\n"
        f"- OOD eval complete: {state['ood_eval']['completed']}\n",
        encoding="utf-8",
    )
    return {"json": json_path, "summary": summary_path}


def decide_next_step(state: dict) -> dict[str, object]:
    if state.get("aris", {}).get("blocked"):
        return {"action": "blocked", "stage": "aris"}
    # keep the earlier decision logic below this guard
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cron_aris_orchestrator.py -v`
Expected: PASS for ARIS handoff tests plus previous tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cron_aris_orchestrator.py pta/scripts/cron_aris_orchestrator.py
git commit -m "feat(automation): add ARIS handoff records"
```

### Task 7: Add cron wrapper and install helper

**Files:**
- Create: `pta/scripts/run_cron_aris_orchestrator.sh`
- Create: `pta/scripts/install_cron_aris_orchestrator.sh`
- Test: `tests/test_cron_shell_contract.py`

- [ ] **Step 1: Write the failing tests for wrapper contents**

```python
from pathlib import Path


def test_cron_wrapper_exports_required_env_vars():
    wrapper = Path("pta/scripts/run_cron_aris_orchestrator.sh").read_text(encoding="utf-8")
    assert "source /home/zhuzihou/dev/Genesis/.venv/bin/activate" in wrapper
    assert "export PYOPENGL_PLATFORM=osmesa" in wrapper
    assert "python3 pta/scripts/cron_aris_orchestrator.py" in wrapper


def test_install_script_prints_90_minute_schedule():
    install_script = Path("pta/scripts/install_cron_aris_orchestrator.sh").read_text(encoding="utf-8")
    assert "*/90" not in install_script
    assert "0 */3 * * *" in install_script or "30 1-23/3 * * *" in install_script
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cron_shell_contract.py -v`
Expected: FAIL because the new shell scripts do not exist yet.

- [ ] **Step 3: Create the cron wrapper and install helper**

```bash
#!/bin/bash
set -euo pipefail

source /home/zhuzihou/dev/Genesis/.venv/bin/activate
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d:${PYTHONPATH:-}

cd /home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d
python3 pta/scripts/cron_aris_orchestrator.py >> logs/orchestration/cron_aris_orchestrator.log 2>&1
```

```bash
#!/bin/bash
set -euo pipefail

CRON_A="0 */3 * * * /home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d/pta/scripts/run_cron_aris_orchestrator.sh"
CRON_B="30 1-22/3 * * * /home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d/pta/scripts/run_cron_aris_orchestrator.sh"

printf '%s\n%s\n' "$CRON_A" "$CRON_B"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cron_shell_contract.py -v`
Expected: PASS for shell contract tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cron_shell_contract.py pta/scripts/run_cron_aris_orchestrator.sh pta/scripts/install_cron_aris_orchestrator.sh
git commit -m "feat(automation): add cron wrapper scripts"
```

### Task 8: Wire coordinator main flow end-to-end

**Files:**
- Modify: `pta/scripts/cron_aris_orchestrator.py`
- Modify: `tests/test_cron_aris_orchestrator.py`

- [ ] **Step 1: Write the failing integration-style test for one-step advancement**

```python
def test_main_launches_next_stage_once_when_m8_done(tmp_path, monkeypatch):
    from pta.scripts import cron_aris_orchestrator as mod

    commands = []

    def fake_launch(command, log_path, cwd):
        commands.append(command)
        return 99999

    monkeypatch.setattr(mod, "launch_detached", fake_launch)
    monkeypatch.setattr(mod, "read_ps_output", lambda: "")

    m8_dir = tmp_path / "checkpoints" / "m8_teacher_seed42"
    m8_dir.mkdir(parents=True)
    (m8_dir / "scoop_transfer_teacher_final.zip").write_text("ok")

    exit_code = mod.run_coordinator(project_root=tmp_path)

    assert exit_code == 0
    assert len(commands) == 1
    assert "--method m1 --seed 42" in commands[0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cron_aris_orchestrator.py -v`
Expected: FAIL because `run_coordinator` and `read_ps_output` do not exist yet.

- [ ] **Step 3: Implement the end-to-end coordinator tick**

```python
def read_ps_output() -> str:
    proc = subprocess.run(
        ["bash", "-lc", "ps -eo pid,etimes,cmd | grep -E 'train_baselines.py|train_m7.py|run_ood_eval_v2.py' | grep -v grep || true"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout


def run_coordinator(project_root: Path) -> int:
    ps_output = read_ps_output()
    state = reconcile_state(project_root=project_root, ps_output=ps_output)
    decision = decide_next_step(state)
    state["stage"] = decision["action"]

    if decision["action"] in {"launch_m8_resume", "launch_m1", "launch_m7", "run_ood_eval"}:
        command = build_command(decision)
        log_name = f"{decision['action']}.log"
        pid = launch_detached(command, project_root / "logs" / "orchestration" / log_name, project_root)
        state["last_launch"] = {"action": decision["action"], "pid": pid, "command": command}
    elif decision["action"] == "handoff_aris":
        write_handoff_files(project_root, state)

    save_state(project_root / "results" / "orchestration" / "aris_state.json", state)
    append_log(project_root / "logs" / "orchestration" / "cron_aris_orchestrator.log", json.dumps({"decision": decision}, sort_keys=True))
    return 0


def main() -> int:
    return run_coordinator(Path(__file__).resolve().parents[2])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cron_aris_orchestrator.py tests/test_cron_shell_contract.py -v`
Expected: PASS for the full orchestration test suite.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cron_aris_orchestrator.py tests/test_cron_shell_contract.py pta/scripts/cron_aris_orchestrator.py
git commit -m "feat(automation): wire cron coordinator end-to-end"
```

### Task 9: Verify cron wrapper behavior and document operator usage

**Files:**
- Create: `docs/30_records/CRON_ARIS_ORCHESTRATOR_RUNBOOK.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Write the failing documentation contract test**

```python
from pathlib import Path


def test_runbook_mentions_cron_install_and_recovery_steps():
    text = Path("docs/30_records/CRON_ARIS_ORCHESTRATOR_RUNBOOK.md").read_text(encoding="utf-8")
    assert "crontab -e" in text
    assert "results/orchestration/aris_state.json" in text
    assert "logs/orchestration/cron_aris_orchestrator.log" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cron_shell_contract.py -v`
Expected: FAIL because the runbook does not exist yet.

- [ ] **Step 3: Add runbook and CLAUDE.md note**

```markdown
# Cron ARIS Orchestrator Runbook

## Install

Print the two-line 90-minute schedule:

`bash pta/scripts/install_cron_aris_orchestrator.sh`

Open the crontab editor:

`crontab -e`

Paste both lines so the job runs every 90 minutes.

## State and Logs

- state: `results/orchestration/aris_state.json`
- coordinator log: `logs/orchestration/cron_aris_orchestrator.log`

## Recovery

After reboot, wait for the next cron tick or run manually:

`bash pta/scripts/run_cron_aris_orchestrator.sh`
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cron_shell_contract.py -v`
Expected: PASS for the runbook contract test and shell contract tests.

- [ ] **Step 5: Commit**

```bash
git add tests/test_cron_shell_contract.py docs/30_records/CRON_ARIS_ORCHESTRATOR_RUNBOOK.md CLAUDE.md
git commit -m "docs(automation): add cron orchestrator runbook"
```

### Task 10: Final verification

**Files:**
- Verify only

- [ ] **Step 1: Run unit and orchestration tests**

Run: `pytest tests/test_cron_aris_orchestrator.py tests/test_cron_shell_contract.py -v`
Expected: PASS with 0 failures.

- [ ] **Step 2: Run the coordinator once manually in the worktree**

Run: `python3 pta/scripts/cron_aris_orchestrator.py`
Expected: exit code `0`, updated `results/orchestration/aris_state.json`, appended coordinator log entry, and either a `wait` decision or a single valid launch decision.

- [ ] **Step 3: Inspect generated state and log files**

Run: `read results/orchestration/aris_state.json` and `read logs/orchestration/cron_aris_orchestrator.log`
Expected: latest decision matches current pipeline reality and no duplicate launches occurred.

- [ ] **Step 4: Print the cron lines for operator install**

Run: `bash pta/scripts/install_cron_aris_orchestrator.sh`
Expected: exactly two cron lines, offset to achieve 90-minute cadence.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(automation): add cron-driven ARIS orchestrator"
```
