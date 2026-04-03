"""State dump -- Serialise full environment state for debugging.

Writes a snapshot of particle positions, robot joint state, sensor
readings, and task metrics to disk so that a failed episode can be
inspected offline.
"""

from __future__ import annotations

from typing import Any, Dict


def dump_state(
    scene_components: Any,
    task_state: Dict[str, Any],
    step: int,
    output_dir: str,
) -> str:
    """Write a full state snapshot to *output_dir*.

    Parameters
    ----------
    scene_components:
        The :class:`SceneComponents` with handles to all entities.
    task_state:
        Current task state dict (metrics, counters, etc.).
    step:
        Current simulation step number.
    output_dir:
        Directory to write the snapshot files into.

    Returns
    -------
    str
        Path to the created snapshot file (NPZ or JSON).
    """
    raise NotImplementedError


def load_state(scene_components: Any, snapshot_path: str) -> None:
    """Restore environment state from a snapshot file.

    Parameters
    ----------
    scene_components:
        The :class:`SceneComponents` to restore into.
    snapshot_path:
        Path to the snapshot file created by :func:`dump_state`.
    """
    raise NotImplementedError
