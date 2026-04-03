"""Checkpoint save / load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def save_checkpoint(
    state: Dict[str, Any],
    path: Path,
    is_best: bool = False,
) -> None:
    """Persist a training checkpoint to disk.

    Parameters
    ----------
    state : dict[str, Any]
        Checkpoint payload — typically includes ``model_state_dict``,
        ``optimizer_state_dict``, ``epoch``, ``global_step``, and any
        scheduler / curriculum state.
    path : Path
        File path for the checkpoint (e.g. ``checkpoints/step_100000.pt``).
    is_best : bool
        If ``True``, also copy the checkpoint to ``best.pt`` in the same
        directory.
    """
    raise NotImplementedError


def load_checkpoint(
    path: Path,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint from disk.

    Parameters
    ----------
    path : Path
        Path to the checkpoint file.
    map_location : str, optional
        Device mapping string (e.g. ``"cpu"``).  Passed directly to
        ``torch.load``.

    Returns
    -------
    dict[str, Any]
        The checkpoint payload.
    """
    raise NotImplementedError
