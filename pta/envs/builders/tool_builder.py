"""ToolBuilder -- Adds scoop/tool entities to the Genesis scene.

Tools are rigid bodies attached to the robot end-effector.  Geometry
comes from the tool library (mesh files under ``assets/tool_meshes/``)
and may be randomised at episode start for OOD evaluation.
"""

from __future__ import annotations

from typing import Any, Dict

import genesis as gs


class ToolBuilder:
    """Build and attach a tool entity to the robot in a Genesis scene."""

    def build_tool(self, scene: Any, config: Dict[str, Any]) -> Any:
        """Add a tool (e.g. scoop) to *scene*.

        Parameters
        ----------
        scene:
            A ``gs.Scene`` instance (not yet built).
        config:
            Tool sub-config with keys ``name``, ``mesh_path``,
            ``attach_link``, ``scale``, and optional randomisation
            parameters.

        Returns
        -------
        entity
            The Genesis entity representing the tool.
        """
        raise NotImplementedError

    def _load_mesh(self, mesh_path: str, scale: float) -> Any:
        """Load a tool mesh and apply uniform scaling."""
        raise NotImplementedError
