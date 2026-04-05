"""In-distribution evaluation split."""

from __future__ import annotations

from typing import Any, Dict, List


def get_id_split(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Return the in-distribution evaluation split configuration.

    Uses the same material families seen during training (sand, snow,
    elastoplastic) with default parameters.

    Parameters
    ----------
    config : dict, optional
        Base experiment configuration (unused for now).

    Returns
    -------
    dict[str, Any]
        Split configuration with ``name``, ``materials`` list, and
        scene overrides for each material.
    """
    return {
        "name": "id",
        "materials": [
            {
                "family": "sand",
                "params": {},  # default sand params
            },
            {
                "family": "snow",
                "params": {},
            },
            {
                "family": "elastoplastic",
                "params": {},
            },
        ],
    }
