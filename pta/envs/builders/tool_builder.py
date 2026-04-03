"""ToolBuilder -- Adds scoop/tool entities to the Genesis scene.

Tools are rigid bodies attached to the robot end-effector.  For v1,
the Franka gripper itself acts as the scoop tool, so this builder
is a thin pass-through.
"""

from __future__ import annotations

from typing import Any, Dict

import genesis as gs


class ToolBuilder:
    """Build and attach a tool entity to the robot in a Genesis scene."""

    def build_tool(self, scene: Any, config: Dict[str, Any] | None = None) -> Any:
        """Add a tool (e.g. scoop) to *scene*.

        For v1, the tool is the Franka gripper itself.  This method
        returns ``None`` and the robot entity is used as the tool handle
        in :class:`SceneComponents`.

        Parameters
        ----------
        scene:
            A ``gs.Scene`` instance (not yet built).
        config:
            Tool sub-config.  Ignored in v1.

        Returns
        -------
        entity or None
            The Genesis entity representing the tool, or ``None`` if
            the gripper is used directly.
        """
        # v1: gripper IS the tool -- no separate entity needed.
        return None

    def _load_mesh(self, mesh_path: str, scale: float = 1.0) -> Any:
        """Load a tool mesh and apply uniform scaling.

        Reserved for future use when custom scoop meshes are available
        under ``assets/tool_meshes/``.
        """
        return gs.morphs.Mesh(file=mesh_path, scale=scale)
