"""SceneBuilder -- Assembles a complete Genesis scene.

Responsibilities:
  1. Initialise Genesis backend (``gs.init``).
  2. Create a ``gs.Scene``.
  3. Delegate to RobotBuilder, ContainerBuilder, ToolBuilder, and
     SensorBuilder to populate the scene.
  4. Call ``scene.build()`` and return the fully-constructed scene with
     references to all entities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import genesis as gs


@dataclass
class SceneComponents:
    """Container returned by :meth:`SceneBuilder.build_scene`."""

    scene: Any  # gs.Scene
    robot: Any  # gs.Entity (Franka)
    tool: Any  # gs.Entity (scoop tool)
    source_container: Any  # gs.Entity
    target_container: Any  # gs.Entity
    sensors: Dict[str, Any]


class SceneBuilder:
    """High-level builder that wires together all sub-builders.

    Usage::

        builder = SceneBuilder()
        components = builder.build_scene(config)
    """

    def build_scene(self, config: Dict[str, Any]) -> SceneComponents:
        """Construct the full Genesis scene from *config*.

        Parameters
        ----------
        config:
            Nested dict (typically loaded from a YAML env config) containing
            keys for ``robot``, ``tool``, ``containers``, ``sensors``,
            ``materials``, and ``scene`` sections.

        Returns
        -------
        SceneComponents
            Dataclass holding the built scene and all entity handles.
        """
        raise NotImplementedError

    def _init_genesis(self, config: Dict[str, Any]) -> None:
        """Call ``gs.init(backend=gs.gpu)`` with settings from *config*."""
        raise NotImplementedError

    def _create_scene(self, config: Dict[str, Any]) -> Any:
        """Create and return a bare ``gs.Scene``."""
        raise NotImplementedError
