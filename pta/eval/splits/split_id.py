"""In-distribution evaluation split."""

from __future__ import annotations

from typing import Any, Dict


def get_id_split(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return the in-distribution evaluation split configuration.

    The ID split uses the same object categories, materials, and
    containers seen during training, but with held-out random seeds.

    Parameters
    ----------
    config : dict
        Base experiment configuration.

    Returns
    -------
    dict[str, Any]
        Split configuration including ``name``, environment parameter
        overrides, and evaluation seeds.
    """
    raise NotImplementedError
