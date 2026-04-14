from __future__ import annotations

import torch

from pta.envs.tasks.scoop_transfer import (
    _should_apply_bowl_constraint_fallback,
    _project_points_into_local_box,
    _should_apply_bowl_sticky_fallback,
)


def test_sticky_fallback_gate_requires_explicit_bowl_flat_carry() -> None:
    assert not _should_apply_bowl_sticky_fallback(
        enabled=False,
        tool_type="bowl",
        task_layout="flat",
        phase="carry",
    )
    assert not _should_apply_bowl_sticky_fallback(
        enabled=True,
        tool_type="scoop",
        task_layout="flat",
        phase="carry",
    )
    assert not _should_apply_bowl_sticky_fallback(
        enabled=True,
        tool_type="bowl",
        task_layout="edge_push",
        phase="carry",
    )
    assert not _should_apply_bowl_sticky_fallback(
        enabled=True,
        tool_type="bowl",
        task_layout="flat",
        phase="off",
    )
    assert _should_apply_bowl_sticky_fallback(
        enabled=True,
        tool_type="bowl",
        task_layout="flat",
        phase="carry",
    )


def test_project_points_into_local_box_limits_snap_distance() -> None:
    points = torch.tensor([[0.06, 0.09, 0.14]], dtype=torch.float32)
    region_min = torch.tensor([-0.034, 0.013, 0.022], dtype=torch.float32)
    region_max = torch.tensor([0.034, 0.067, 0.12], dtype=torch.float32)

    projected = _project_points_into_local_box(
        points, region_min, region_max, max_snap=0.01
    )

    expected = torch.tensor([[0.05, 0.08, 0.13]], dtype=torch.float32)
    assert torch.allclose(projected, expected)


def test_constraint_fallback_gate_requires_explicit_bowl_flat_carry() -> None:
    assert not _should_apply_bowl_constraint_fallback(
        enabled=False,
        tool_type="bowl",
        task_layout="flat",
        phase="carry",
    )
    assert not _should_apply_bowl_constraint_fallback(
        enabled=True,
        tool_type="gripper",
        task_layout="flat",
        phase="carry",
    )
    assert not _should_apply_bowl_constraint_fallback(
        enabled=True,
        tool_type="bowl",
        task_layout="edge_push",
        phase="carry",
    )
    assert not _should_apply_bowl_constraint_fallback(
        enabled=True,
        tool_type="bowl",
        task_layout="flat",
        phase="off",
    )
    assert _should_apply_bowl_constraint_fallback(
        enabled=True,
        tool_type="bowl",
        task_layout="flat",
        phase="carry",
    )
