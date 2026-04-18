from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import re
import shlex
import subprocess


@dataclass
class CompletionStatus:
    completed: bool
    final_checkpoint: Path | None


M1_SEEDS = [42, 0, 1]
M7_SEEDS = [42, 0, 1]
DEFAULT_STATE = {
    "stage": "bootstrap",
    "m8": {"running": False, "completed": False},
    "m1": {"running": False, "completed_seeds": []},
    "m7": {"running": False, "completed_seeds": []},
    "ood_eval": {"completed": False},
    "aris": {"ready": False, "blocked": False},
}


def parse_ps_output(output: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=2)
        if len(parts) != 3:
            continue
        pid_str, elapsed_str, cmd = parts
        if not pid_str.isdigit() or not elapsed_str.isdigit():
            continue
        rows.append({"pid": int(pid_str), "elapsed": int(elapsed_str), "cmd": cmd})
    return rows


def detect_run_completion(checkpoint_dir: Path, final_name: str) -> CompletionStatus:
    final_path = checkpoint_dir / final_name
    return CompletionStatus(
        completed=final_path.exists(),
        final_checkpoint=final_path if final_path.exists() else None,
    )


def choose_latest_resume_checkpoint(checkpoint_dir: Path) -> Path | None:
    candidates = sorted(checkpoint_dir.glob("*.zip"))
    if not candidates:
        return None
    finals = [path for path in candidates if path.name.endswith("_final.zip")]
    if finals:
        return finals[-1]
    return max(candidates, key=_checkpoint_sort_key)


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_(\d+)_steps\.zip$", path.name)
    if match:
        return (int(match.group(1)), path.name)
    return (-1, path.name)


def _first_missing_seed(done: list[int], expected: list[int]) -> int | None:
    for seed in expected:
        if seed not in done:
            return seed
    return None


def decide_next_step(state: dict) -> dict[str, object]:
    if state.get("aris", {}).get("blocked"):
        return {"action": "blocked", "stage": "aris"}

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


def build_command(decision: dict[str, object]) -> str:
    action = decision["action"]
    if action == "launch_m8_resume":
        return (
            "python pta/scripts/train_baselines.py --method m8 --seed 42 "
            "--total-timesteps 500000 --residual-scale 0.05 "
            "--resume-from checkpoints/m8_teacher_seed42/scoop_transfer_teacher_final.zip"
        )
    if action == "launch_m1":
        return (
            f"python pta/scripts/train_baselines.py --method m1 --seed {decision['seed']} "
            "--total-timesteps 500000 --residual-scale 0.05"
        )
    if action == "launch_m7":
        return (
            f"python pta/scripts/train_m7.py --seed {decision['seed']} "
            "--total-timesteps 500000 --residual-scale 0.05"
        )
    if action == "run_ood_eval":
        return "python pta/scripts/run_ood_eval_v2.py --residual-scale 0.05"
    raise ValueError(f"Unsupported action: {action}")


def launch_detached(command: str, log_path: Path, cwd: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            shlex.split(command),
            cwd=cwd,
            stdout=handle,
            stderr=handle,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    return process.pid


def load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return json.loads(json.dumps(DEFAULT_STATE))
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def write_handoff_files(project_root: Path, state: dict) -> dict[str, Path]:
    out_dir = project_root / "results" / "orchestration"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "aris_handoff_ready.json"
    summary_path = out_dir / "aris_handoff_summary.md"
    handoff_state = json.loads(json.dumps(state))
    handoff_state.setdefault("aris", {})["ready"] = True
    json_path.write_text(
        json.dumps(handoff_state, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary_path.write_text(
        "# ARIS Handoff Ready\n\n"
        f"- M8 complete: {state['m8']['completed']}\n"
        f"- M1 seeds: {state['m1']['completed_seeds']}\n"
        f"- M7 seeds: {state['m7']['completed_seeds']}\n"
        f"- OOD eval complete: {state['ood_eval']['completed']}\n"
        "- Main results: results/main_results.csv\n"
        "- OOD per-seed results: results/ood_eval_per_seed.csv\n"
        "- Handoff record: results/orchestration/aris_handoff_ready.json\n",
        encoding="utf-8",
    )
    return {"json": json_path, "summary": summary_path}


def execute_decision(
    project_root: Path, state: dict, decision: dict[str, object]
) -> dict[str, object]:
    if decision["action"] == "handoff_aris":
        return {
            "action": "handoff_aris",
            "records": write_handoff_files(project_root, state),
        }
    raise ValueError(f"Unsupported action: {decision['action']}")


def reconcile_state(project_root: Path, ps_output: str) -> dict:
    state = json.loads(json.dumps(DEFAULT_STATE))
    processes = parse_ps_output(ps_output)
    cmds = [row["cmd"] for row in processes]

    state["m8"]["running"] = any(
        "train_baselines.py --method m8 --seed 42" in cmd for cmd in cmds
    )
    state["m1"]["running"] = any(
        "train_baselines.py --method m1" in cmd for cmd in cmds
    )
    state["m7"]["running"] = any("train_m7.py" in cmd for cmd in cmds)

    m8_dir = project_root / "checkpoints" / "m8_teacher_seed42"
    state["m8"]["completed"] = detect_run_completion(
        m8_dir, "scoop_transfer_teacher_final.zip"
    ).completed

    for seed in M1_SEEDS:
        m1_dir = project_root / "checkpoints" / f"m1_reactive_seed{seed}"
        if detect_run_completion(m1_dir, "scoop_transfer_teacher_final.zip").completed:
            state["m1"]["completed_seeds"].append(seed)

    for seed in M7_SEEDS:
        m7_dir = project_root / "checkpoints" / f"m7_pta_seed{seed}"
        if (
            detect_run_completion(m7_dir, "m7_pta_final.zip").completed
            or detect_run_completion(m7_dir, "m7_pta_final").completed
        ):
            state["m7"]["completed_seeds"].append(seed)

    results_dir = project_root / "results"
    state["ood_eval"]["completed"] = all(
        (results_dir / name).exists()
        for name in ["main_results.csv", "ood_eval_per_seed.csv"]
    )
    return state


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def read_ps_output() -> str:
    process = subprocess.run(
        [
            "bash",
            "-lc",
            "ps -eo pid,etimes,cmd | grep -E 'train_baselines.py|train_m7.py|run_ood_eval_v2.py' | grep -v grep || true",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return process.stdout


def run_coordinator(project_root: Path) -> int:
    state_path = project_root / "results" / "orchestration" / "aris_state.json"
    log_dir = project_root / "logs" / "orchestration"
    persisted_state = load_state(state_path)
    state = reconcile_state(project_root=project_root, ps_output=read_ps_output())
    state["aris"] = {
        "ready": persisted_state.get("aris", {}).get("ready", False),
        "blocked": persisted_state.get("aris", {}).get("blocked", False),
    }
    decision = decide_next_step(state)
    state["stage"] = decision["action"]

    if decision["action"] in {
        "launch_m8_resume",
        "launch_m1",
        "launch_m7",
        "run_ood_eval",
    }:
        command = build_command(decision)
        pid = launch_detached(
            command,
            log_dir / f"{decision['action']}.log",
            project_root,
        )
        state["last_launch"] = {
            "action": decision["action"],
            "pid": pid,
            "command": command,
        }
    elif decision["action"] == "handoff_aris":
        execute_decision(project_root, state, decision)
        state["aris"]["ready"] = True

    save_state(state_path, state)
    append_log(
        log_dir / "cron_aris_orchestrator.log",
        json.dumps({"decision": decision}, sort_keys=True),
    )
    return 0


def main(project_root: Path | None = None) -> int:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    return run_coordinator(project_root)


if __name__ == "__main__":
    raise SystemExit(main())
