import time

import genesis as gs


def main() -> None:
    if not gs._initialized:
        gs.init(backend=gs.cpu, precision="32", logging_level="info")

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(substeps=4),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            gravity=(0, 0, -9.8),
            box_box_detection=True,
            constraint_timeconst=0.01,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.5, 0.0, 0.7),
            camera_lookat=(0.2, 0.0, 0.1),
            camera_fov=50,
            max_FPS=60,
        ),
        show_viewer=True,
    )

    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(
        material=gs.materials.Rigid(rho=300),
        morph=gs.morphs.Box(
            pos=(0.5, 0.0, 0.07),
            size=(0.04, 0.04, 0.04),
        ),
        surface=gs.surfaces.Default(color=(1.0, 0.2, 0.2, 1.0)),
    )

    scene.build()
    print("MINIMAL_VIEWER_BUILT", flush=True)

    while scene.viewer.is_alive():
        scene.step()
        time.sleep(0.01)

    print("MINIMAL_VIEWER_EXITED", flush=True)


if __name__ == "__main__":
    main()
