"""Tool geometry randomisation for domain randomisation and OOD testing.

Applies random perturbations (scale, aspect ratio, curvature offset)
to a tool mesh or its parametric representation so that the policy
learns to generalise across tool shapes.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def randomize_tool_geometry(
    tool_config: Dict[str, Any],
    rng: np.random.Generator | None = None,
    scale_range: tuple[float, float] = (0.8, 1.2),
    aspect_range: tuple[float, float] = (0.9, 1.1),
) -> Dict[str, Any]:
    """Return a modified copy of *tool_config* with randomised geometry.

    Parameters
    ----------
    tool_config:
        Base tool configuration from the tool registry.
    rng:
        Numpy random generator for reproducibility.
    scale_range:
        Min/max uniform scale factor.
    aspect_range:
        Min/max aspect-ratio perturbation.

    Returns
    -------
    dict
        New tool config with updated geometry parameters.
    """
    raise NotImplementedError
