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
        config: Dict[str, Any] | None = None,
    ) -> Tuple[Any, Any]:
        """Add source and target containers to *scene*.

        Parameters
        ----------
        scene:
            A ``gs.Scene`` instance (not yet built).
        config:
            Container sub-config with ``source`` and ``target``
            sections, each specifying position, size, wall height,
            and wall thickness.

        Returns
        -------
        tuple[entity, entity]
            ``(source_container, target_container)`` Genesis entities
            (base plate entities).
        """
        cfg = config or {}
        source_cfg = cfg.get("source", {
            "pos": (0.5, 0.0, 0.05),
            "size": (0.15, 0.15, 0.005),
            "wall_thickness": 0.005,
            "wall_height": 0.08,
        })
        target_cfg = cfg.get("target", {
            "pos": (0.5, 0.35, 0.05),
            "size": (0.12, 0.12, 0.005),
            "wall_thickness": 0.005,
            "wall_height": 0.10,
        })

        source = self._add_container(scene, source_cfg)
        target = self._add_container(scene, target_cfg)
        return source, target

    def _add_container(self, scene: Any, container_cfg: Dict[str, Any]) -> Any:
        """Add a single open-top box container to *scene*.

        Creates a base plate + 4 walls, all fixed and MPM-coupled.
        Returns the base entity as the handle.
        """
        mat = gs.materials.Rigid(needs_coup=True, coup_friction=0.5)

        pos = container_cfg["pos"]
        size = container_cfg["size"]
        wt = container_cfg.get("wall_thickness", 0.005)
        wh = container_cfg.get("wall_height", 0.08)

        bx, by, bz = pos
        sx, sy, _ = size
        half_sx, half_sy = sx / 2, sy / 2

        # Base plate
        base = scene.add_entity(
            material=mat,
            morph=gs.morphs.Box(pos=pos, size=size, fixed=True),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(color=(0.6, 0.6, 0.6)),
            ),
        )

        # 4 walls
        walls = [
            ((bx + half_sx + wt / 2, by, bz + wh / 2), (wt, sy + 2 * wt, wh)),
            ((bx - half_sx - wt / 2, by, bz + wh / 2), (wt, sy + 2 * wt, wh)),
            ((bx, by + half_sy + wt / 2, bz + wh / 2), (sx, wt, wh)),
            ((bx, by - half_sy - wt / 2, bz + wh / 2), (sx, wt, wh)),
        ]
        for w_pos, w_size in walls:
            scene.add_entity(
                material=mat,
                morph=gs.morphs.Box(pos=w_pos, size=w_size, fixed=True),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(color=(0.5, 0.5, 0.5)),
                ),
            )

        return base
