from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


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
