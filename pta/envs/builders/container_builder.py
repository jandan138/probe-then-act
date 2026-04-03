"""ContainerBuilder -- Adds source and target containers to the scene.

Containers are rigid bodies (bowls, trays, bins) that hold the
granular / fluid material.  The source container is filled with MPM
particles at reset; the target container is the drop-off zone.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import genesis as gs


class ContainerBuilder:
    """Build source and target container entities."""

    def build_containers(
        self,
        scene: Any,
        config: Dict[str, Any],
    ) -> Tuple[Any, Any]:
        """Add source and target containers to *scene*.

        Parameters
        ----------
        scene:
            A ``gs.Scene`` instance (not yet built).
        config:
            Container sub-config with ``source`` and ``target``
            sections, each specifying mesh/primitive shape, position,
            and scale.

        Returns
        -------
        tuple[entity, entity]
            ``(source_container, target_container)`` Genesis entities.
        """
        raise NotImplementedError

    def _add_container(self, scene: Any, container_cfg: Dict[str, Any]) -> Any:
        """Add a single container entity to *scene*."""
        raise NotImplementedError
