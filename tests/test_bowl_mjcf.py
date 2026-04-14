import os
import sys
import xml.etree.ElementTree as ET

os.environ["PYOPENGL_PLATFORM"] = "osmesa"

sys.path.insert(0, "/home/zhuzihou/dev/probe-then-act")

import genesis as gs
import numpy as np
import torch


def main() -> None:
    gs.init(backend=gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=2e-3, substeps=25),
        mpm_options=gs.options.MPMOptions(
            lower_bound=(-0.1, -0.5, -0.05),
            upper_bound=(1.0, 0.8, 0.8),
            grid_density=64,
        ),
        rigid_options=gs.options.RigidOptions(
            dt=2e-3,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        show_viewer=False,
    )

    scene.add_entity(
        material=gs.materials.Rigid(needs_coup=True, coup_friction=0.5),
        morph=gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
    )

    robot = scene.add_entity(
        material=gs.materials.Rigid(needs_coup=True, coup_friction=1.0),
        morph=gs.morphs.MJCF(
            file="xml/franka_emika_panda/panda_bowl.xml",
            pos=(0.0, 0.0, 0.0),
        ),
    )

    sand = scene.add_entity(
        material=gs.materials.MPM.Sand(),
        morph=gs.morphs.Box(pos=(0.5, 0.0, 0.10), size=(0.10, 0.10, 0.04)),
        surface=gs.surfaces.Default(color=(0.9, 0.8, 0.5, 1.0), vis_mode="particle"),
    )

    scene.build(n_envs=0)

    assert robot.n_dofs == 7, f"Expected 7 DOFs, got {robot.n_dofs}"

    link_names = [link.name for link in robot.links]
    assert "scoop" in link_names, f"'scoop' link not found in {link_names}"

    scoop_link = robot.get_link("scoop")
    robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]))
    robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200]))

    home_qpos = torch.tensor([0, 0, 0, -1.57079, 0, 1.57079, -0.7853], device=gs.device)
    robot.set_qpos(home_qpos)

    for _ in range(50):
        scene.step()

    scoop_pos = scoop_link.get_pos()
    assert not torch.isnan(scoop_pos).any(), "NaN detected in bowl position"

    xml_path = "/home/zhuzihou/dev/Genesis/genesis/assets/xml/franka_emika_panda/panda_bowl.xml"
    geom_names = [
        elem.attrib.get("name", "") for elem in ET.parse(xml_path).iterfind(".//geom")
    ]
    assert "scoop_front" in geom_names, f"Front wall missing from {geom_names}"
    assert "scoop_hidden_front_lip" in geom_names, (
        f"Hidden front lip missing from {geom_names}"
    )
    assert "scoop_hidden_left_lip" in geom_names, (
        f"Hidden left lip missing from {geom_names}"
    )
    assert "scoop_hidden_right_lip" in geom_names, (
        f"Hidden right lip missing from {geom_names}"
    )
    assert "scoop_visual_front" in geom_names
    assert "scoop_visual_left" in geom_names
    assert "scoop_visual_right" in geom_names

    highwall_xml_path = "/home/zhuzihou/dev/Genesis/genesis/assets/xml/franka_emika_panda/panda_bowl_highwall.xml"
    highwall_geom_names = [
        elem.attrib.get("name", "")
        for elem in ET.parse(highwall_xml_path).iterfind(".//geom")
    ]
    assert "scoop_highwall_front" in highwall_geom_names
    assert "scoop_highwall_left" in highwall_geom_names
    assert "scoop_highwall_right" in highwall_geom_names
    assert "scoop_visual_highwall_front" in highwall_geom_names
    assert "scoop_visual_highwall_left" in highwall_geom_names
    assert "scoop_visual_highwall_right" in highwall_geom_names

    print("=== panda_bowl.xml test PASSED ===")
    print(f"DOFs: {robot.n_dofs}")
    print(f"Particles: {sand._n_particles}")


if __name__ == "__main__":
    main()
