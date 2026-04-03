"""OOD split — novel tools not seen during training."""

from __future__ import annotations

from typing import Any, Dict


def get_ood_tool_split() -> Dict[str, Any]:
    """Return the OOD tool evaluation split.

    The robot uses end-effector tools (e.g. different gripper shapes,
    spatula) that were not present in training.

    Returns
    -------
    dict[str, Any]
        Split configuration with ``name="ood_tool"`` and the
        corresponding tool parameter overrides.
    """
    raise NotImplementedError
