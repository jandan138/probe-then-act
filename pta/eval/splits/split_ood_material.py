"""OOD split — novel materials not seen during training."""

from __future__ import annotations

from typing import Any, Dict


def get_ood_material_split() -> Dict[str, Any]:
    """Return the OOD material evaluation split.

    Objects use materials (e.g. rubber, glass, ceramic) that were
    excluded from the training distribution.

    Returns
    -------
    dict[str, Any]
        Split configuration with ``name="ood_material"`` and the
        corresponding material parameter overrides.
    """
    raise NotImplementedError
