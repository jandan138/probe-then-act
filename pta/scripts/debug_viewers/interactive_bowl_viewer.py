import torch
import genesis as gs

from pta.envs.builders.scene_builder import (
    SceneBuilder,
    _resolve_mpm_options_kwargs,
    _resolve_scene_substeps,
)
from pta.scripts.render_bowl_carry_onset_videos import (
    VIEWER_INITIAL_SETTLE_STEPS,
    build_bowl_carry_plan,
    build_bowl_viewer_motion_plan,
    build_bowl_viewer_waypoints,
    build_env,
    set_transport_phase,
)
from pta.scripts.run_scripted_baseline import EXTEND_FWD_S, HOME_S


orig_create_scene = SceneBuilder._create_scene


def patched_create_scene(self, config):
    ctrl_dt = config["ctrl_dt"]
    return gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=ctrl_dt, substeps=_resolve_scene_substeps(config)
        ),
        mpm_options=gs.options.MPMOptions(**_resolve_mpm_options_kwargs(config)),
        rigid_options=gs.options.RigidOptions(
            dt=ctrl_dt,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, -2.5, 2.0),
            camera_lookat=(1.0, 0.0, 0.0),
            camera_fov=35,
            max_FPS=60,
        ),
        show_viewer=True,
    )


SceneBuilder._create_scene = patched_create_scene


def scene_step(env, n=1):
    for _ in range(n):
        if not env.scene.viewer.is_alive():
            return False
        env.scene.step()
        if hasattr(env, "post_physics_update"):
            env.post_physics_update()
    return True


def interp(env, start_qpos, end_qpos, n_steps, settle_per_step):
    start = torch.tensor(start_qpos, dtype=torch.float32, device="cuda")
    end = torch.tensor(end_qpos, dtype=torch.float32, device="cuda")
    for i in range(n_steps):
        if not env.scene.viewer.is_alive():
            return False
        alpha = (i + 1) / n_steps
        qpos = start * (1 - alpha) + end * alpha
        env.robot.set_qpos(qpos)
        if not scene_step(env, settle_per_step):
            return False
    return True


def interp_pd(env, start_qpos, end_qpos, n_steps, settle_per_step):
    start = torch.tensor(start_qpos, dtype=torch.float32, device="cuda")
    end = torch.tensor(end_qpos, dtype=torch.float32, device="cuda")
    for i in range(n_steps):
        if not env.scene.viewer.is_alive():
            return False
        alpha = (i + 1) / n_steps
        qpos = start * (1 - alpha) + end * alpha
        env.robot.control_dofs_position(qpos)
        if not scene_step(env, settle_per_step):
            return False
    return True


def settle(env, n_steps):
    return scene_step(env, n_steps)


def play_once(env):
    env.reset()
    set_transport_phase(env, "off")
    if not settle(env, VIEWER_INITIAL_SETTLE_STEPS):
        return False
    for segment in build_bowl_viewer_motion_plan(
        first_half_steps=15, second_half_steps=15
    )[:4]:
        runner = interp_pd if segment["mode"] == "pd" else interp
        if not runner(
            env,
            segment["start"],
            segment["end"],
            segment["steps"],
            segment["settle_per_step"],
        ):
            return False
    if not settle(env, 10):
        return False
    motion_plan = build_bowl_viewer_motion_plan(
        first_half_steps=15, second_half_steps=15
    )
    lift_segment = motion_plan[4]
    if not interp_pd(
        env,
        lift_segment["start"],
        lift_segment["end"],
        lift_segment["steps"],
        lift_segment["settle_per_step"],
    ):
        return False
    if not settle(env, 10):
        return False
    set_transport_phase(env, "carry")
    for segment in build_bowl_carry_plan(first_half_steps=15, second_half_steps=15):
        if not interp(env, segment["start"], segment["end"], segment["steps"], 2):
            return False
        if not settle(env, segment["settle_after"]):
            return False
    set_transport_phase(env, "off")
    return True


try:
    env = build_env(tool_type="bowl_highwall", seed=102)
    print(
        "Genesis viewer launched. Use mouse to move camera; press i in the viewer for shortcuts.",
        flush=True,
    )
    while env.scene.viewer.is_alive():
        if not play_once(env):
            break
except KeyboardInterrupt:
    pass
finally:
    SceneBuilder._create_scene = orig_create_scene
