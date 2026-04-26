from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
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
OOD_SPLITS = [
    "id_sand",
    "ood_snow",
    "ood_elastoplastic",
    "ood_sand_soft",
    "ood_sand_hard",
]
OPTIONAL_OOD_CHECKPOINTS = {
    "m8_teacher": {
        "seeds": [0, 1],
        "ckpt_pattern": "checkpoints/m8_teacher_seed{seed}/best/best_model",
    },
    "m7_noprobe": {
        "seeds": [42, 0, 1],
        "ckpt_pattern": "checkpoints/m7_pta_noprobe_seed{seed}/best/best_model",
    },
    "m7_nobelief": {
        "seeds": [42, 0, 1],
        "ckpt_pattern": "checkpoints/m7_pta_nobelief_seed{seed}/best/best_model",
    },
}
OOD_RESULT_FIELDNAMES = [
    "method",
    "seed",
    "split",
    "mean_reward",
    "std_reward",
    "mean_transfer",
    "std_transfer",
    "mean_spill",
    "std_spill",
    "success_rate",
    "n_failed_episodes",
]
OOD_RESULT_FLOAT_FIELDS = [
    "mean_reward",
    "std_reward",
    "mean_transfer",
    "std_transfer",
    "mean_spill",
    "std_spill",
    "success_rate",
]
OOD_AGGREGATE_FIELDNAMES = [
    "method",
    "split",
    "n_seeds",
    "mean_reward_mean",
    "mean_reward_std",
    "mean_transfer_mean",
    "mean_transfer_std",
    "mean_spill_mean",
    "mean_spill_std",
    "success_rate_mean",
    "success_rate_std",
    "n_failed_episodes_mean",
    "n_failed_episodes_std",
    "n_failed_episodes_sum",
]
OOD_AGGREGATE_FLOAT_FIELDS = [
    "mean_reward_mean",
    "mean_reward_std",
    "mean_transfer_mean",
    "mean_transfer_std",
    "mean_spill_mean",
    "mean_spill_std",
    "success_rate_mean",
    "success_rate_std",
    "n_failed_episodes_mean",
    "n_failed_episodes_std",
]
DEFAULT_STATE = {
    "stage": "bootstrap",
    "m8": {"running": False, "completed": False},
    "m1": {"running": False, "completed_seeds": []},
    "m7": {"running": False, "completed_seeds": []},
    "ood_eval": {"running": False, "completed": False, "resume_allowed": True},
    "aris": {"ready": False, "blocked": False, "failure_reason": None},
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


def outputs_newer_than_dependencies(
    output_paths: list[Path], dependency_paths: list[Path]
) -> bool:
    if not dependency_paths:
        return False
    if any(not path.exists() for path in output_paths):
        return False
    newest_dependency = max(path.stat().st_mtime_ns for path in dependency_paths)
    oldest_output = min(path.stat().st_mtime_ns for path in output_paths)
    return oldest_output > newest_dependency


def read_ood_result_keys(per_seed_path: Path) -> set[tuple[str, int, str]]:
    if not per_seed_path.exists():
        return set()

    keys = set()
    with per_seed_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if any(row.get(field) in (None, "") for field in OOD_RESULT_FIELDNAMES):
                    return set()
                for field in OOD_RESULT_FLOAT_FIELDS:
                    if not math.isfinite(float(row[field])):
                        raise ValueError(f"non-finite {field}")
                _coerce_nonnegative_int(row["n_failed_episodes"], "n_failed_episodes")
                key = (
                    row["method"],
                    _coerce_nonnegative_int(row["seed"], "seed"),
                    row["split"],
                )
                if key in keys:
                    return set()
                keys.add(key)
            except (KeyError, OverflowError, TypeError, ValueError):
                return set()
    return keys


def _coerce_nonnegative_int(value, field: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError(f"invalid {field}")
    return int(number)


def read_ood_aggregate_counts(main_results_path: Path) -> dict[tuple[str, str], int]:
    if not main_results_path.exists():
        return {}

    counts = {}
    with main_results_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                if any(row.get(field) in (None, "") for field in OOD_AGGREGATE_FIELDNAMES):
                    return {}
                for field in OOD_AGGREGATE_FLOAT_FIELDS:
                    if not math.isfinite(float(row[field])):
                        raise ValueError(f"non-finite {field}")
                n_seeds = _coerce_nonnegative_int(row["n_seeds"], "n_seeds")
                _coerce_nonnegative_int(
                    row["n_failed_episodes_sum"], "n_failed_episodes_sum"
                )
                key = (row["method"], row["split"])
                if key in counts:
                    return {}
                counts[key] = n_seeds
            except (KeyError, OverflowError, TypeError, ValueError):
                return {}
    return counts


def expected_ood_aggregate_counts(
    expected_keys: set[tuple[str, int, str]],
) -> dict[tuple[str, str], int]:
    seeds_by_group: dict[tuple[str, str], set[int]] = {}
    for method, seed, split in expected_keys:
        seeds_by_group.setdefault((method, split), set()).add(seed)
    return {key: len(seeds) for key, seeds in seeds_by_group.items()}


def _checkpoint_if_exists(project_root: Path, pattern: str, seed: int) -> Path | None:
    ckpt_path = project_root / pattern.format(seed=seed)
    if ckpt_path.exists():
        return ckpt_path
    zip_path = ckpt_path.with_suffix(".zip")
    if zip_path.exists():
        return zip_path
    return None


def optional_ood_checkpoint_paths(project_root: Path) -> list[Path]:
    paths = []
    for method_cfg in OPTIONAL_OOD_CHECKPOINTS.values():
        for seed in method_cfg["seeds"]:
            path = _checkpoint_if_exists(project_root, method_cfg["ckpt_pattern"], seed)
            if path is not None:
                paths.append(path)
    return paths


def expected_ood_result_keys(
    project_root: Path,
    state: dict,
) -> set[tuple[str, int, str]]:
    keys = set()
    for seed in state["m1"]["completed_seeds"]:
        keys.update(("m1_reactive", seed, split) for split in OOD_SPLITS)
    if state["m8"]["completed"]:
        keys.update(("m8_teacher", 42, split) for split in OOD_SPLITS)
    for seed in state["m7"]["completed_seeds"]:
        keys.update(("m7_pta", seed, split) for split in OOD_SPLITS)
    for method, method_cfg in OPTIONAL_OOD_CHECKPOINTS.items():
        for seed in method_cfg["seeds"]:
            if _checkpoint_if_exists(project_root, method_cfg["ckpt_pattern"], seed):
                keys.update((method, seed, split) for split in OOD_SPLITS)
    return keys


def ood_outputs_complete(
    project_root: Path,
    results_dir: Path,
    dependency_paths: list[Path],
    state: dict,
) -> bool:
    ood_outputs = [
        results_dir / "main_results.csv",
        results_dir / "ood_eval_per_seed.csv",
    ]
    dependencies = [*dependency_paths, *optional_ood_checkpoint_paths(project_root)]
    if not outputs_newer_than_dependencies(ood_outputs, dependencies):
        return False
    main_results_path = results_dir / "main_results.csv"
    per_seed_path = results_dir / "ood_eval_per_seed.csv"
    if main_results_path.stat().st_mtime_ns < per_seed_path.stat().st_mtime_ns:
        return False

    expected_keys = expected_ood_result_keys(project_root, state)
    if not expected_keys:
        return False
    actual_keys = read_ood_result_keys(per_seed_path)
    if actual_keys != expected_keys:
        return False
    aggregate_counts = read_ood_aggregate_counts(main_results_path)
    expected_counts = expected_ood_aggregate_counts(expected_keys)
    return aggregate_counts == expected_counts


def ood_resume_allowed(results_dir: Path, dependency_paths: list[Path]) -> bool:
    output_paths = [
        results_dir / "main_results.csv",
        results_dir / "ood_eval_per_seed.csv",
    ]
    existing_outputs = [path for path in output_paths if path.exists()]
    if not existing_outputs or not dependency_paths:
        return True
    newest_dependency = max(path.stat().st_mtime_ns for path in dependency_paths)
    return all(path.stat().st_mtime_ns > newest_dependency for path in existing_outputs)


def decide_next_step(state: dict) -> dict[str, object]:
    if state.get("aris", {}).get("blocked"):
        return {"action": "blocked", "stage": "aris"}

    if state.get("aris", {}).get("ready"):
        return {"action": "ready", "stage": "aris"}

    if state["ood_eval"].get("running"):
        return {"action": "wait", "stage": "ood_eval"}

    if state["m7"]["running"]:
        return {"action": "wait", "stage": "m7"}

    if state["m1"]["running"]:
        return {"action": "wait", "stage": "m1"}

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
        if not state["ood_eval"].get("resume_allowed", True):
            return {"action": "run_ood_eval_no_resume"}
        return {"action": "run_ood_eval"}

    return {"action": "handoff_aris"}


def build_command(decision: dict[str, object], project_root: Path | None = None) -> str:
    action = decision["action"]
    if action == "launch_m8_resume":
        resume_path = Path(
            "checkpoints/m8_teacher_seed42/scoop_transfer_teacher_final.zip"
        )
        if project_root is not None:
            latest = choose_latest_resume_checkpoint(
                project_root / "checkpoints" / "m8_teacher_seed42"
            )
            if latest is not None:
                resume_path = latest.relative_to(project_root)
        return (
            "python pta/scripts/train_baselines.py --method m8 --seed 42 "
            "--total-timesteps 500000 --residual-scale 0.05 "
            f"--resume-from {resume_path}"
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
    if action == "run_ood_eval_no_resume":
        return "python pta/scripts/run_ood_eval_v2.py --residual-scale 0.05 --no-resume"
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


def write_handoff_files(
    project_root: Path, state: dict, *, ready: bool
) -> dict[str, Path]:
    out_dir = project_root / "results" / "orchestration"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "aris_handoff_ready.json"
    summary_path = out_dir / "aris_handoff_summary.md"
    handoff_state = json.loads(json.dumps(state))
    handoff_state.setdefault("aris", {})["ready"] = ready
    json_path.write_text(
        json.dumps(handoff_state, indent=2, sort_keys=True), encoding="utf-8"
    )
    heading = "# ARIS Handoff Ready\n\n" if ready else "# ARIS Handoff Pending\n\n"
    summary_path.write_text(
        (
            heading
            + f"- M8 complete: {state['m8']['completed']}\n"
            + f"- M1 seeds: {state['m1']['completed_seeds']}\n"
            + f"- M7 seeds: {state['m7']['completed_seeds']}\n"
            + f"- OOD eval complete: {state['ood_eval']['completed']}\n"
            + "- Main results: results/main_results.csv\n"
            + "- OOD per-seed results: results/ood_eval_per_seed.csv\n"
            + "- Handoff record: results/orchestration/aris_handoff_ready.json\n"
        ),
        encoding="utf-8",
    )
    return {"json": json_path, "summary": summary_path}


def run_handoff_command(command: str, log_path: Path, cwd: Path) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        completed = subprocess.run(
            shlex.split(command),
            cwd=cwd,
            stdout=handle,
            stderr=handle,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    return {"command": command, "returncode": completed.returncode}


def execute_decision(
    project_root: Path, state: dict, decision: dict[str, object]
) -> dict[str, object]:
    if decision["action"] == "handoff_aris":
        command = read_handoff_command(project_root)
        if command is None:
            return {
                "action": "handoff_aris",
                "records": write_handoff_files(project_root, state, ready=True),
                "returncode": 0,
            }

        result = {
            "action": "handoff_aris",
            **run_handoff_command(
                command,
                project_root / "logs" / "orchestration" / "handoff_aris.log",
                project_root,
            ),
        }
        if result["returncode"] == 0:
            result["records"] = write_handoff_files(project_root, state, ready=True)
        else:
            result["failure_reason"] = (
                f"handoff command failed with exit code {result['returncode']}"
            )
        return result
    raise ValueError(f"Unsupported action: {decision['action']}")


def reconcile_state(project_root: Path, ps_output: str) -> dict:
    state = json.loads(json.dumps(DEFAULT_STATE))
    processes = parse_ps_output(ps_output)
    cmds = [row["cmd"] for row in processes]
    completed_checkpoints: list[Path] = []

    state["m8"]["running"] = any(
        "train_baselines.py --method m8 --seed 42" in cmd for cmd in cmds
    )
    state["m1"]["running"] = any(
        "train_baselines.py --method m1" in cmd for cmd in cmds
    )
    state["m7"]["running"] = any("train_m7.py" in cmd for cmd in cmds)
    state["ood_eval"]["running"] = any("run_ood_eval_v2.py" in cmd for cmd in cmds)

    m8_dir = project_root / "checkpoints" / "m8_teacher_seed42"
    m8_status = detect_run_completion(m8_dir, "scoop_transfer_teacher_final.zip")
    state["m8"]["completed"] = m8_status.completed
    if m8_status.final_checkpoint is not None:
        completed_checkpoints.append(m8_status.final_checkpoint)

    for seed in M1_SEEDS:
        m1_dir = project_root / "checkpoints" / f"m1_reactive_seed{seed}"
        m1_status = detect_run_completion(m1_dir, "scoop_transfer_teacher_final.zip")
        if m1_status.completed:
            state["m1"]["completed_seeds"].append(seed)
            if m1_status.final_checkpoint is not None:
                completed_checkpoints.append(m1_status.final_checkpoint)

    for seed in M7_SEEDS:
        m7_dir = project_root / "checkpoints" / f"m7_pta_seed{seed}"
        m7_status = detect_run_completion(m7_dir, "m7_pta_final.zip")
        if not m7_status.completed:
            m7_status = detect_run_completion(m7_dir, "m7_pta_final")
        if m7_status.completed:
            state["m7"]["completed_seeds"].append(seed)
            if m7_status.final_checkpoint is not None:
                completed_checkpoints.append(m7_status.final_checkpoint)

    results_dir = project_root / "results"
    ood_dependencies = [
        *completed_checkpoints,
        *optional_ood_checkpoint_paths(project_root),
    ]
    state["ood_eval"]["resume_allowed"] = ood_resume_allowed(
        results_dir,
        ood_dependencies,
    )
    state["ood_eval"]["completed"] = ood_outputs_complete(
        project_root,
        results_dir,
        completed_checkpoints,
        state,
    )
    return state


def append_log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


def read_handoff_command(project_root: Path) -> str | None:
    command_path = (
        project_root / "results" / "orchestration" / "aris_handoff_command.txt"
    )
    if not command_path.exists():
        return None
    command = command_path.read_text(encoding="utf-8").strip()
    return command or None


def timestamp_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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
    check_time = timestamp_now()
    state["aris"] = {
        "ready": persisted_state.get("aris", {}).get("ready", False),
        "blocked": persisted_state.get("aris", {}).get("blocked", False),
        "failure_reason": persisted_state.get("aris", {}).get("failure_reason"),
    }
    if state["aris"]["ready"] and not state["ood_eval"]["completed"]:
        state["aris"]["ready"] = False
    decision = decide_next_step(state)
    state["stage"] = decision["action"]
    state["last_check_time"] = check_time
    log_entry: dict[str, object] = {"timestamp": check_time, "decision": decision}

    if decision["action"] in {
        "launch_m8_resume",
        "launch_m1",
        "launch_m7",
        "run_ood_eval",
        "run_ood_eval_no_resume",
    }:
        command = build_command(decision, project_root=project_root)
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
        log_entry.update({"command": command, "pid": pid})
    elif decision["action"] == "handoff_aris":
        result = execute_decision(project_root, state, decision)
        state["aris"]["ready"] = result.get("returncode", 0) == 0
        state["aris"]["blocked"] = result.get("returncode", 0) != 0
        state["aris"]["failure_reason"] = result.get("failure_reason")
        if "command" in result:
            state["last_handoff"] = {
                "action": decision["action"],
                "command": result["command"],
                "returncode": result["returncode"],
            }
            if "failure_reason" in result:
                state["last_handoff"]["failure_reason"] = result["failure_reason"]
            log_entry.update(
                {
                    "command": result["command"],
                    "returncode": result["returncode"],
                    "failure_reason": result.get("failure_reason"),
                }
            )

    save_state(state_path, state)
    append_log(
        log_dir / "cron_aris_orchestrator.log",
        json.dumps(log_entry, sort_keys=True),
    )
    return 0


def main(project_root: Path | None = None) -> int:
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    return run_coordinator(project_root)


if __name__ == "__main__":
    raise SystemExit(main())
