from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


@dataclass
class CompletionStatus:
    completed: bool
    final_checkpoint: Path | None


M1_SEEDS = [42, 0, 1]
M7_SEEDS = [42, 0, 1]


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
