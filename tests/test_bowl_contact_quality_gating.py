from __future__ import annotations

from pta.envs.builders.scene_builder import (
    _DEFAULT_CONFIG,
    _bowl_contact_quality_active,
    _resolve_mpm_options_kwargs,
    _resolve_robot_material_kwargs,
    _resolve_scene_substeps,
)


def test_bowl_contact_quality_stays_off_for_default_edge_config() -> None:
    cfg = dict(_DEFAULT_CONFIG)
    assert not _bowl_contact_quality_active(cfg)
    assert _resolve_scene_substeps(cfg) == cfg["substeps"]
    assert _resolve_mpm_options_kwargs(cfg) == {
        "lower_bound": cfg["mpm_lower_bound"],
        "upper_bound": cfg["mpm_upper_bound"],
        "grid_density": cfg["mpm_grid_density"],
    }
    assert _resolve_robot_material_kwargs(cfg) == {
        "needs_coup": True,
        "coup_friction": 3.0,
    }


def test_bowl_contact_quality_requires_explicit_flat_bowl_gate() -> None:
    cfg = dict(_DEFAULT_CONFIG)
    cfg.update(
        {
            "tool_type": "bowl",
            "task_layout": "flat",
            "bowl_contact_quality_enabled": True,
            "bowl_enable_cpic": True,
            "bowl_substeps_override": 40,
            "bowl_robot_coup_friction": 6.0,
            "bowl_robot_coup_softness": 0.0005,
            "bowl_robot_sdf_cell_size": 0.002,
            "bowl_robot_sdf_min_res": 64,
            "bowl_robot_sdf_max_res": 256,
        }
    )

    assert _bowl_contact_quality_active(cfg)
    assert _resolve_scene_substeps(cfg) == 40
    assert _resolve_mpm_options_kwargs(cfg)["enable_CPIC"] is True
    assert _resolve_robot_material_kwargs(cfg) == {
        "needs_coup": True,
        "coup_friction": 6.0,
        "coup_softness": 0.0005,
        "sdf_cell_size": 0.002,
        "sdf_min_res": 64,
        "sdf_max_res": 256,
    }


def test_bowl_contact_quality_does_not_leak_into_edge_layout() -> None:
    cfg = dict(_DEFAULT_CONFIG)
    cfg.update(
        {
            "tool_type": "bowl",
            "task_layout": "edge_push",
            "bowl_contact_quality_enabled": True,
            "bowl_enable_cpic": True,
            "bowl_substeps_override": 40,
            "bowl_robot_coup_friction": 9.0,
        }
    )

    assert not _bowl_contact_quality_active(cfg)
    assert _resolve_scene_substeps(cfg) == cfg["substeps"]
    assert "enable_CPIC" not in _resolve_mpm_options_kwargs(cfg)
    assert _resolve_robot_material_kwargs(cfg) == {
        "needs_coup": True,
        "coup_friction": 3.0,
    }
