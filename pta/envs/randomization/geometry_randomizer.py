"""GeometryRandomizer -- Randomise container and tool shapes.

Perturbs container dimensions (width, depth, wall angle) and tool
geometry to test policy robustness to shape variation.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class GeometryRandomizer:
    """Randomise geometric parameters of containers and tools.

    Usage::

        gr = GeometryRandomizer(config)
        new_container_cfg = gr.randomize_container(base_cfg, rng)
        new_tool_cfg = gr.randomize_tool(base_cfg, rng)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Store geometry randomisation ranges.

        Parameters
        ----------
        config:
            Sub-config with ``container`` and ``tool`` randomisation
            ranges (scale, aspect, depth offsets, etc.).
        """
        self.config = config

    def randomize_container(
        self,
        base_config: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Return a modified copy of *base_config* with randomised container geometry.

        Parameters
        ----------
        base_config:
            Default container config.
        rng:
            Numpy random generator.

        Returns
        -------
        dict
            New container config with perturbed dimensions.
        """
        raise NotImplementedError

    def randomize_tool(
        self,
        base_config: Dict[str, Any],
        rng: np.random.Generator,
    ) -> Dict[str, Any]:
        """Return a modified copy of *base_config* with randomised tool geometry.

        Parameters
        ----------
        base_config:
            Default tool config from the tool registry.
        rng:
            Numpy random generator.

        Returns
        -------
        dict
            New tool config with perturbed geometry.
        """
        raise NotImplementedError
