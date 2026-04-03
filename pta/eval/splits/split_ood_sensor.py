"""OOD split — novel sensor configurations not seen during training."""

from __future__ import annotations

from typing import Any, Dict


def get_ood_sensor_split() -> Dict[str, Any]:
    """Return the OOD sensor evaluation split.

    The agent operates with degraded or novel sensor modalities (e.g.
    missing depth channel, lower-resolution tactile) not encountered
    during training.

    Returns
    -------
    dict[str, Any]
        Split configuration with ``name="ood_sensor"`` and the
        corresponding sensor parameter overrides.
    """
    raise NotImplementedError
