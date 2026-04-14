from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "pta" / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from bowl_transport_diagnosis import (
    FLAT_BOWL_SCENE,
    BASELINE_BOWL_SCENE,
    effective_scene_runtime_config,
    effective_task_runtime_config,
)


def test_effective_runtime_marks_contact_quality_inactive_for_non_flat_bowl() -> None:
    scene_cfg = {
        **BASELINE_BOWL_SCENE,
        "bowl_contact_quality_enabled": True,
        "bowl_enable_cpic": True,
        "bowl_substeps_override": 40,
        "bowl_robot_coup_friction": 6.0,
    }

    effective = effective_scene_runtime_config(scene_cfg)

    assert effective["bowl_contact_quality_active"] is False
    assert effective["effective_substeps"] == 25
    assert effective["effective_mpm_options"] == {
        "lower_bound": (-0.1, -0.5, -0.05),
        "upper_bound": (1.0, 0.8, 0.8),
        "grid_density": 128,
    }
    assert effective["effective_robot_material"] == {
        "needs_coup": True,
        "coup_friction": 3.0,
    }


def test_effective_runtime_marks_flat_bowl_overrides_as_active() -> None:
    scene_cfg = {
        **FLAT_BOWL_SCENE,
        "bowl_contact_quality_enabled": True,
        "bowl_enable_cpic": True,
        "bowl_substeps_override": 40,
        "bowl_robot_coup_friction": 6.0,
    }
    task_cfg = {"bowl_sticky_fallback_enabled": True}

    effective_scene = effective_scene_runtime_config(scene_cfg)
    effective_task = effective_task_runtime_config(scene_cfg, task_cfg)

    assert effective_scene["bowl_contact_quality_active"] is True
    assert effective_scene["effective_substeps"] == 40
    assert effective_scene["effective_mpm_options"]["enable_CPIC"] is True
    assert effective_scene["effective_robot_material"]["coup_friction"] == 6.0
    assert effective_task["bowl_sticky_fallback_available"] is True
    assert effective_task["bowl_sticky_runtime_activation_requires_phase"] == "carry"
