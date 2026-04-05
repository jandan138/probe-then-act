"""OOD split -- novel materials not seen during training.

OOD-Material split: trained on sand+elastoplastic, test on snow
(held-out family) plus extreme parameter ranges for seen families.
"""

from __future__ import annotations

from typing import Any, Dict


def get_ood_material_split() -> Dict[str, Any]:
    """Return the OOD material evaluation split.

    Tests generalization to:
    1. Held-out family: snow (if training excluded it)
    2. Extreme parameter ranges beyond training distribution

    Returns
    -------
    dict[str, Any]
        Split configuration with ``name="ood_material"`` and material
        configs for evaluation.
    """
    return {
        "name": "ood_material",
        "materials": [
            # Held-out family: snow
            {
                "family": "snow",
                "params": {},
            },
            # Extreme sand: high E, high friction
            {
                "family": "sand",
                "params": {"E": 7.5e4, "rho": 1800.0},
                "label": "sand_extreme",
            },
            # Extreme elastoplastic: high yield stress
            {
                "family": "elastoplastic",
                "params": {"E": 5e5, "nu": 0.4, "rho": 1400.0},
                "label": "elastoplastic_extreme",
            },
            # Liquid (never seen in training)
            {
                "family": "liquid",
                "params": {},
                "label": "liquid_novel",
            },
        ],
    }
