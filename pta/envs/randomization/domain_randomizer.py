"""DomainRandomizer -- Top-level domain randomisation controller.

Coordinates material sampling, tool geometry perturbation, container
variation, lighting changes, and observation noise so that each
episode sees a diverse but controlled set of conditions.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


class DomainRandomizer:
    """Orchestrates all per-episode randomisation.

    Usage::

        dr = DomainRandomizer(config)
        episode_params = dr.sample_episode(rng)
        dr.apply(scene_components, episode_params)
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialise with randomisation ranges from *config*.

        Parameters
        ----------
        config:
            Randomisation sub-config with keys ``material``, ``tool``,
            ``container``, ``lighting``, ``observation_noise``, etc.
        """
        self.config = config

    def sample_episode(self, rng: np.random.Generator) -> Dict[str, Any]:
        """Sample all randomisation parameters for a single episode.

        Returns
        -------
        dict
            Keys include ``material_params``, ``tool_params``,
            ``container_params``, ``lighting_params``, etc.
        """
        raise NotImplementedError

    def apply(self, scene_components: Any, episode_params: Dict[str, Any]) -> None:
        """Apply *episode_params* to the scene before ``scene.build()`` or reset.

        Parameters
        ----------
        scene_components:
            The :class:`SceneComponents` to mutate.
        episode_params:
            Output of :meth:`sample_episode`.
        """
        raise NotImplementedError
