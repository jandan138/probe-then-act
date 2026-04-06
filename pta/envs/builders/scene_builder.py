"""SceneBuilder -- Assembles a complete Genesis scene for scoop-and-transfer.

Responsibilities:
  1. Initialise Genesis backend (``gs.init``).
  2. Create a ``gs.Scene`` with MPM + Rigid coupling.
  3. Add ground plane, containers, Franka robot, MPM particles, and camera.
  4. Call ``scene.build()`` and configure robot PD gains.
  5. Return a :class:`SceneComponents` dataclass with all entity handles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch

import genesis as gs

from pta.envs.builders.material_builder import MaterialBuilder


@dataclass
class SceneComponents:
    """Container returned by :meth:`SceneBuilder.build_scene`."""

    scene: Any  # gs.Scene
    robot: Any  # gs.Entity (Franka)
    tool: Any  # gs.Entity (scoop tool -- for v1 the gripper IS the tool)
    source_container: Any  # gs.Entity
    target_container: Any  # gs.Entity
    particles: Any  # gs.Entity (MPM particles)
    camera: Any  # camera handle
    sensors: Dict[str, Any] = field(default_factory=dict)

    # Useful cached references
    ee_link: Any = None
    left_finger_link: Any = None
    right_finger_link: Any = None
    arm_dof_idx: Any = None
    finger_dof_idx: Any = None

    # Geometry metadata for metric computation
    source_pos: tuple = (0.5, 0.0, 0.0)
    source_size: tuple = (0.15, 0.15, 0.10)
    target_pos: tuple = (0.5, 0.35, 0.0)
    target_size: tuple = (0.12, 0.12, 0.12)


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "ctrl_dt": 2e-3,
    "substeps": 25,
    # MPM domain -- must enclose source/target containers + workspace
    "mpm_lower_bound": (-0.1, -0.5, -0.05),
    "mpm_upper_bound": (1.0, 0.8, 0.8),
    "mpm_grid_density": 128,  # 128 needed for scoop-particle coupling
    # Task layout: "flat" (original) or "edge_push" (elevated platform)
    "task_layout": "edge_push",
    # Containers (flat layout)
    "source_pos": (0.5, 0.0, 0.05),
    "source_size": (0.15, 0.15, 0.005),  # thin base plate
    "source_wall_thickness": 0.005,
    "source_wall_height": 0.08,
    "target_pos": (0.5, 0.35, 0.05),
    "target_size": (0.12, 0.12, 0.005),  # thin base plate
    "target_wall_thickness": 0.005,
    "target_wall_height": 0.10,
    # Particles
    "particle_material": "sand",
    "particle_params": {},
    "particle_pos": (0.55, 0.03, 0.20),   # near platform edge for edge_push
    "particle_size": (0.12, 0.06, 0.03),
    # Camera
    "camera_res": (128, 128),
    "camera_pos": (1.2, -0.3, 0.8),
    "camera_lookat": (0.5, 0.15, 0.1),
    "camera_fov": 60,
    # Robot
    "robot_pos": (0.0, 0.0, 0.0),
    # Tool
    "tool_type": "gripper",  # "gripper" or "scoop"
    # Build
    "n_envs": 0,  # 0 = single env
}


class SceneBuilder:
    """High-level builder that wires together all sub-builders.

    Usage::

        builder = SceneBuilder()
        components = builder.build_scene(config)
    """

    def __init__(self) -> None:
        self._material_builder = MaterialBuilder()

    def build_scene(self, config: Dict[str, Any] | None = None) -> SceneComponents:
        """Construct the full Genesis scene from *config*.

        Parameters
        ----------
        config:
            Nested dict (typically loaded from a YAML env config) containing
            keys for ``robot``, ``tool``, ``containers``, ``sensors``,
            ``materials``, and ``scene`` sections.  Missing keys fall back
            to ``_DEFAULT_CONFIG``.

        Returns
        -------
        SceneComponents
            Dataclass holding the built scene and all entity handles.
        """
        cfg = {**_DEFAULT_CONFIG, **(config or {})}

        # 1. Initialise Genesis
        self._init_genesis(cfg)

        # 2. Create the scene
        scene = self._create_scene(cfg)

        # 3. Add ground plane (coupled with MPM)
        self._add_ground(scene)

        # 4-5. Add containers (depends on layout)
        task_layout = cfg.get("task_layout", "flat")
        if task_layout == "edge_push":
            source_container, target_container = self._add_edge_push_layout(scene, cfg)
        else:
            # 4. Add source container
            source_container = self._add_box_container(
                scene,
                name="source",
                base_pos=cfg["source_pos"],
                base_size=cfg["source_size"],
                wall_thickness=cfg["source_wall_thickness"],
                wall_height=cfg["source_wall_height"],
            )

            # 5. Add target container
            target_container = self._add_box_container(
                scene,
                name="target",
                base_pos=cfg["target_pos"],
                base_size=cfg["target_size"],
                wall_thickness=cfg["target_wall_thickness"],
                wall_height=cfg["target_wall_height"],
            )

        # 6. Add MPM particles in the source container
        particles = self._add_particles(scene, cfg)

        # 7. Add Franka robot (with coup_friction for MPM interaction)
        robot = self._add_robot(scene, cfg)

        # 8. Add camera
        camera = self._add_camera(scene, cfg)

        # 9. Build the scene
        n_envs = cfg["n_envs"]
        if n_envs <= 1:
            scene.build(n_envs=0)
        else:
            scene.build(n_envs=n_envs)

        # 10. Configure robot PD gains (must be after build)
        tool_type = cfg.get("tool_type", "gripper")
        self._configure_robot_gains(robot, tool_type)

        # 11. Get link references (depends on tool type)
        if tool_type == "scoop":
            ee_link = robot.get_link("scoop")
            left_finger_link = None
            right_finger_link = None
            # Scoop: 7 arm DOFs, no finger DOFs
            arm_dof_idx = torch.arange(robot.n_dofs, device=gs.device)
            finger_dof_idx = torch.tensor([], dtype=torch.long, device=gs.device)
        else:
            ee_link = robot.get_link("hand")
            left_finger_link = robot.get_link("left_finger")
            right_finger_link = robot.get_link("right_finger")
            n_arm_dof = robot.n_dofs - 2
            arm_dof_idx = torch.arange(n_arm_dof, device=gs.device)
            finger_dof_idx = torch.arange(n_arm_dof, n_arm_dof + 2, device=gs.device)

        components = SceneComponents(
            scene=scene,
            robot=robot,
            tool=robot,  # v1: gripper IS the tool
            source_container=source_container,
            target_container=target_container,
            particles=particles,
            camera=camera,
            sensors={"overhead_cam": camera},
            ee_link=ee_link,
            left_finger_link=left_finger_link,
            right_finger_link=right_finger_link,
            arm_dof_idx=arm_dof_idx,
            finger_dof_idx=finger_dof_idx,
        )

        # Set source/target positions based on layout
        if task_layout == "edge_push":
            pp = cfg.get("platform_pos", (0.55, 0.0, 0.15))
            ps = cfg.get("platform_size", (0.20, 0.15, 0.01))
            pwh = cfg.get("platform_wall_height", 0.04)
            etp = cfg.get("edge_target_pos", (0.55, 0.15, 0.01))
            ets = cfg.get("edge_target_size", (0.30, 0.20, 0.005))
            etwh = cfg.get("edge_target_wall_height", 0.04)
            components.source_pos = pp
            components.source_size = (ps[0], ps[1], pwh)
            components.target_pos = etp
            components.target_size = (ets[0], ets[1], etwh)
        else:
            components.source_pos = cfg["source_pos"]
            components.source_size = (cfg["source_size"][0], cfg["source_size"][1], cfg["source_wall_height"])
            components.target_pos = cfg["target_pos"]
            components.target_size = (cfg["target_size"][0], cfg["target_size"][1], cfg["target_wall_height"])

        return components

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_genesis(self, config: Dict[str, Any]) -> None:
        """Call ``gs.init(backend=gs.gpu)`` with settings from *config*."""
        if gs._initialized:
            return
        gs.init(
            backend=gs.gpu,
            precision="32",
            logging_level="warning",
            performance_mode=True,
        )

    def _create_scene(self, config: Dict[str, Any]) -> Any:
        """Create and return a bare ``gs.Scene``."""
        ctrl_dt = config["ctrl_dt"]
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=ctrl_dt,
                substeps=config["substeps"],
            ),
            mpm_options=gs.options.MPMOptions(
                lower_bound=config["mpm_lower_bound"],
                upper_bound=config["mpm_upper_bound"],
                grid_density=config["mpm_grid_density"],
            ),
            rigid_options=gs.options.RigidOptions(
                dt=ctrl_dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=False,
        )
        return scene

    def _add_ground(self, scene: Any) -> Any:
        """Add a ground plane with rigid-MPM coupling."""
        return scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True,
                coup_friction=0.5,
            ),
            morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        )

    def _add_edge_push_layout(
        self, scene: Any, config: Dict[str, Any]
    ) -> tuple:
        """Add edge-push layout: elevated platform + target below edge.

        The source is an elevated platform with walls on 3 sides (not +y).
        Particles are pushed off the +y edge and fall into a target
        container at ground level.
        """
        mat = gs.materials.Rigid(needs_coup=True, coup_friction=0.5)

        pp = config.get("platform_pos", (0.55, 0.0, 0.15))
        ps = config.get("platform_size", (0.20, 0.15, 0.01))
        pwh = config.get("platform_wall_height", 0.04)
        px, py, pz = pp
        psx, psy, psz = ps
        half_x, half_y = psx / 2, psy / 2

        # Platform base
        platform = scene.add_entity(
            material=mat, morph=gs.morphs.Box(pos=pp, size=ps, fixed=True),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.6, 0.6, 0.6))),
        )

        # Walls on 3 sides (back -y, left -x, right +x). NO wall on +y (the edge)
        scene.add_entity(material=mat, morph=gs.morphs.Box(
            pos=(px, py - half_y, pz + pwh / 2), size=(psx, 0.005, pwh), fixed=True))
        scene.add_entity(material=mat, morph=gs.morphs.Box(
            pos=(px - half_x, py, pz + pwh / 2), size=(0.005, psy, pwh), fixed=True))
        scene.add_entity(material=mat, morph=gs.morphs.Box(
            pos=(px + half_x, py, pz + pwh / 2), size=(0.005, psy, pwh), fixed=True))

        # Target container below the platform edge
        etp = config.get("edge_target_pos", (0.55, 0.15, 0.01))
        ets = config.get("edge_target_size", (0.30, 0.20, 0.005))
        etwh = config.get("edge_target_wall_height", 0.04)
        etx, ety, etz = etp
        etsx, etsy, etsz = ets
        et_hx, et_hy = etsx / 2, etsy / 2

        # Target base
        target = scene.add_entity(
            material=mat, morph=gs.morphs.Box(pos=etp, size=ets, fixed=True),
            surface=gs.surfaces.Rough(diffuse_texture=gs.textures.ColorTexture(color=(0.5, 0.5, 0.5))),
        )

        # Target walls (3 sides, open on the side facing platform)
        scene.add_entity(material=mat, morph=gs.morphs.Box(
            pos=(etx, ety + et_hy, etz + etwh / 2), size=(etsx, 0.005, etwh), fixed=True))
        scene.add_entity(material=mat, morph=gs.morphs.Box(
            pos=(etx - et_hx, ety, etz + etwh / 2), size=(0.005, etsy, etwh), fixed=True))
        scene.add_entity(material=mat, morph=gs.morphs.Box(
            pos=(etx + et_hx, ety, etz + etwh / 2), size=(0.005, etsy, etwh), fixed=True))

        return platform, target

    def _add_box_container(
        self,
        scene: Any,
        name: str,
        base_pos: tuple,
        base_size: tuple,
        wall_thickness: float,
        wall_height: float,
    ) -> Any:
        """Add a simple open-top box container made of rigid box entities.

        Creates a base plate + 4 walls. All pieces are fixed and coupled
        to MPM.  Returns the base entity as the container handle (the
        walls are added to the scene but not tracked individually).
        """
        mat = gs.materials.Rigid(needs_coup=True, coup_friction=0.5)
        bx, by, bz = base_pos
        sx, sy, _ = base_size
        half_sx, half_sy = sx / 2, sy / 2

        # Base plate
        base_entity = scene.add_entity(
            material=mat,
            morph=gs.morphs.Box(
                pos=base_pos,
                size=base_size,
                fixed=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(0.6, 0.6, 0.6),
                ),
            ),
        )

        # 4 walls: front (+x), back (-x), left (+y), right (-y)
        wall_configs = [
            # pos (center of wall), size
            ((bx + half_sx + wall_thickness / 2, by, bz + wall_height / 2),
             (wall_thickness, sy + 2 * wall_thickness, wall_height)),
            ((bx - half_sx - wall_thickness / 2, by, bz + wall_height / 2),
             (wall_thickness, sy + 2 * wall_thickness, wall_height)),
            ((bx, by + half_sy + wall_thickness / 2, bz + wall_height / 2),
             (sx, wall_thickness, wall_height)),
            ((bx, by - half_sy - wall_thickness / 2, bz + wall_height / 2),
             (sx, wall_thickness, wall_height)),
        ]
        for w_pos, w_size in wall_configs:
            scene.add_entity(
                material=mat,
                morph=gs.morphs.Box(pos=w_pos, size=w_size, fixed=True),
                surface=gs.surfaces.Rough(
                    diffuse_texture=gs.textures.ColorTexture(
                        color=(0.5, 0.5, 0.5),
                    ),
                ),
            )

        return base_entity

    def _add_particles(self, scene: Any, config: Dict[str, Any]) -> Any:
        """Add MPM particles as a block inside the source container."""
        family = config.get("particle_material", "sand")
        params = config.get("particle_params", {})
        material = self._material_builder.create_material(family, params)

        particles = scene.add_entity(
            material=material,
            morph=gs.morphs.Box(
                pos=config["particle_pos"],
                size=config["particle_size"],
            ),
            surface=gs.surfaces.Default(
                color=(0.9, 0.8, 0.5, 1.0),
                vis_mode="particle",
            ),
        )
        return particles

    def _add_robot(self, scene: Any, config: Dict[str, Any]) -> Any:
        """Add a Franka Panda robot with MPM coupling."""
        robot_pos = config.get("robot_pos", (0.0, 0.0, 0.0))
        tool_type = config.get("tool_type", "gripper")

        if tool_type == "scoop":
            mjcf_file = "xml/franka_emika_panda/panda_scoop.xml"
        else:
            mjcf_file = "xml/franka_emika_panda/panda.xml"

        robot = scene.add_entity(
            material=gs.materials.Rigid(
                needs_coup=True,
                coup_friction=3.0,  # high friction for MPM particle interaction
            ),
            morph=gs.morphs.MJCF(
                file=mjcf_file,
                pos=robot_pos,
            ),
        )
        return robot

    def _add_camera(self, scene: Any, config: Dict[str, Any]) -> Any:
        """Add an overhead observation camera."""
        cam = scene.add_camera(
            res=config.get("camera_res", (128, 128)),
            pos=config.get("camera_pos", (1.2, -0.3, 0.8)),
            lookat=config.get("camera_lookat", (0.5, 0.15, 0.1)),
            fov=config.get("camera_fov", 60),
            GUI=False,
        )
        return cam

    def _configure_robot_gains(self, robot: Any, tool_type: str = "gripper") -> None:
        """Set PD gains and force ranges for the Franka. Must be called
        after scene.build()."""
        if tool_type == "scoop":
            # 7 arm DOFs only (no finger DOFs)
            robot.set_dofs_kp(
                np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]),
            )
            robot.set_dofs_kv(
                np.array([450, 450, 350, 350, 200, 200, 200]),
            )
            robot.set_dofs_force_range(
                np.array([-87, -87, -87, -87, -12, -12, -12]),
                np.array([87, 87, 87, 87, 12, 12, 12]),
            )
        else:
            robot.set_dofs_kp(
                np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
            )
            robot.set_dofs_kv(
                np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
            )
            robot.set_dofs_force_range(
                np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
            )
