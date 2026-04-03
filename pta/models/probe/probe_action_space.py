"""Probe action space — primitive types and their parameter specifications."""

from __future__ import annotations

from enum import IntEnum
from typing import Any, Dict


class ProbeAction(IntEnum):
    """Enumeration of probe primitive types."""

    TAP = 0
    PRESS = 1
    DRAG = 2


# Per-primitive parameter specifications.
# Keys: ``"param_dim"`` (int), ``"description"`` (str).
PROBE_PRIMITIVES: Dict[ProbeAction, Dict[str, Any]] = {
    ProbeAction.TAP: {
        "param_dim": 6,
        "description": (
            "Quick tap at a target (x, y, z) with approach direction "
            "(dx, dy, dz). Returns impact force profile."
        ),
    },
    ProbeAction.PRESS: {
        "param_dim": 6,
        "description": (
            "Sustained press at (x, y, z) with target force (fx, fy, fz). "
            "Returns force-displacement curve."
        ),
    },
    ProbeAction.DRAG: {
        "param_dim": 6,
        "description": (
            "Lateral drag from (x0, y0, z0) to (x1, y1, z1). "
            "Returns friction / shear profile."
        ),
    },
}
