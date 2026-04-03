"""Collect demonstration data from a trained teacher policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def collect_demonstrations(
    teacher: Any,
    env: Any,
    n_episodes: int,
    save_dir: Optional[Path] = None,
) -> list[dict[str, Any]]:
    """Roll out the teacher policy and store transitions.

    Parameters
    ----------
    teacher : Any
        A trained teacher policy (e.g. ``PrivilegedTeacher``).
    env : Any
        Gym-compatible environment instance.
    n_episodes : int
        Number of full episodes to collect.
    save_dir : Path, optional
        If provided, serialise the collected dataset to this directory.

    Returns
    -------
    list[dict[str, Any]]
        List of episode dictionaries, each containing ``obs``, ``actions``,
        ``rewards``, ``dones``, and ``infos`` arrays.
    """
    raise NotImplementedError
