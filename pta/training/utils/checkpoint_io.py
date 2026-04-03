"""Checkpoint save / load utilities.

Supports both SB3 model checkpoints (via model.save / model.load) and
custom PyTorch checkpoints with metadata (config, seed, step, metrics).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    state: Dict[str, Any],
    path: Path,
    is_best: bool = False,
) -> None:
    """Persist a training checkpoint to disk.

    Parameters
    ----------
    state : dict[str, Any]
        Checkpoint payload -- typically includes ``model_state_dict``,
        ``optimizer_state_dict``, ``epoch``, ``global_step``, and any
        scheduler / curriculum state.
    path : Path
        File path for the checkpoint (e.g. ``checkpoints/step_100000.pt``).
    is_best : bool
        If ``True``, also copy the checkpoint to ``best.pt`` in the same
        directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp if not present
    if "timestamp" not in state:
        state["timestamp"] = datetime.now(timezone.utc).isoformat()

    torch.save(state, path)

    if is_best:
        best_path = path.parent / "best.pt"
        shutil.copy2(path, best_path)


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

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location=map_location, weights_only=False)


def save_sb3_checkpoint(
    model: Any,
    path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save an SB3 model plus optional metadata sidecar.

    Parameters
    ----------
    model:
        A Stable-Baselines3 model (PPO, RecurrentPPO, etc.).
    path:
        Save path (without extension -- SB3 adds ``.zip``).
    metadata:
        Optional metadata dict (config, seed, step, metrics, etc.)
        saved as a JSON sidecar alongside the model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))

    if metadata is not None:
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)


def load_sb3_metadata(path: Path) -> Dict[str, Any]:
    """Load the JSON metadata sidecar for an SB3 checkpoint.

    Parameters
    ----------
    path:
        Path to the ``.zip`` model file (the sidecar is ``.json``).

    Returns
    -------
    dict
        Metadata dict.
    """
    meta_path = Path(path).with_suffix(".json")
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)
