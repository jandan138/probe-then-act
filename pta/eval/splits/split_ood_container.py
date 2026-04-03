"""OOD split — novel containers not seen during training."""

from __future__ import annotations

from typing import Any, Dict


def get_ood_container_split() -> Dict[str, Any]:
    """Return the OOD container evaluation split.

    Tasks involve target containers (bowls, boxes, trays) with
    geometries not seen during training.

    Returns
    -------
    dict[str, Any]
        Split configuration with ``name="ood_container"`` and the
        corresponding container parameter overrides.
    """
    raise NotImplementedError
