from pta.scripts import render_bowl_carry_onset_videos as render_script
from pta.scripts.run_scripted_baseline import (
    BOWL_APPROACH_S,
    BOWL_CAPTURE_S,
    BOWL_INSERT_S,
    BOWL_LIFT_S,
    BOWL_TRAVERSE_MID_S,
)


def test_viewer_scene_geometry_is_large_enough_for_bowl_loading() -> None:
    scene_cfg = render_script.build_viewer_scene_config("bowl_highwall")

    assert scene_cfg["task_layout"] == "flat"
    assert scene_cfg["source_size"][0] >= 0.24
    assert scene_cfg["source_size"][1] >= 0.20
    assert scene_cfg["source_wall_height"] >= 0.12
    assert scene_cfg["particle_size"][0] >= 0.15
    assert scene_cfg["particle_size"][1] >= 0.13
    assert scene_cfg["particle_size"][2] >= 0.04
    assert scene_cfg["particle_pos"][2] <= 0.12


def test_viewer_uses_shared_tightened_carry_plan() -> None:
    carry_plan = render_script.build_bowl_carry_plan(
        first_half_steps=15, second_half_steps=15
    )
    waypoints = render_script.build_bowl_viewer_waypoints()

    assert len(carry_plan) == 4

    entry_segment = carry_plan[0]
    assert entry_segment["start"] == waypoints["lift"]
    assert entry_segment["end"][0] < BOWL_TRAVERSE_MID_S[0]
    assert entry_segment["end"][1] <= BOWL_LIFT_S[1]
    assert entry_segment["settle_after"] >= 12


def test_viewer_uses_more_conservative_bowl_waypoints_than_baseline() -> None:
    waypoints = render_script.build_bowl_viewer_waypoints()

    assert waypoints["approach"][0] <= BOWL_APPROACH_S[0]
    assert waypoints["insert"][1] <= BOWL_INSERT_S[1]
    assert waypoints["capture"][1] <= BOWL_CAPTURE_S[1]
    assert waypoints["lift"][1] <= BOWL_LIFT_S[1]


def test_viewer_initial_settle_is_long_enough_for_material_to_relax() -> None:
    assert render_script.VIEWER_INITIAL_SETTLE_STEPS >= 40


def test_viewer_motion_plan_uses_pd_for_contact_critical_segments() -> None:
    plan = render_script.build_bowl_viewer_motion_plan(
        first_half_steps=15, second_half_steps=15
    )

    assert plan[0]["mode"] == "qpos"
    assert plan[1]["mode"] == "pd"
    assert plan[2]["mode"] == "pd"
    assert plan[3]["mode"] == "pd"
    assert plan[4]["mode"] == "pd"
    assert plan[5]["mode"] == "qpos"
