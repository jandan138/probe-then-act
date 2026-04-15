"""Fix 1: Test that edge-push trajectory has a settle segment at the end.

The settle segment is critical: after the 3-pass push, particles need time
to settle into the target AABB. Without it, the last ~90 steps of a 500-step
horizon are dead (robot frozen, accumulating penalties).
"""

import numpy as np
import pytest


def test_edge_push_trajectory_has_settle():
    """Trajectory末尾必须有静止settle段，让粒子有时间落入target AABB。"""
    from pta.envs.wrappers.joint_residual_wrapper import build_edge_push_trajectory

    traj = build_edge_push_trajectory()

    # 1) 总长度 >= 490 (原 410 + 80 settle)
    assert traj.shape[0] >= 490, f"Trajectory too short: {traj.shape[0]}, expected >= 490"

    # 2) shape 是 (T, 7) — 7 joints
    assert traj.shape[1] == 7, f"Expected 7 joints, got {traj.shape[1]}"


def test_settle_frames_are_static():
    """settle段的所有帧必须完全相同（静止）。"""
    from pta.envs.wrappers.joint_residual_wrapper import build_edge_push_trajectory

    traj = build_edge_push_trajectory()

    # 最后 80 帧全部相同
    settle_segment = traj[-80:]
    for i in range(1, len(settle_segment)):
        np.testing.assert_array_equal(
            settle_segment[i],
            settle_segment[0],
            err_msg=f"Settle frame {i} differs from frame 0",
        )


def test_settle_matches_push_endpoint():
    """settle帧的关节配置必须等于push终点帧。"""
    from pta.envs.wrappers.joint_residual_wrapper import build_edge_push_trajectory

    traj = build_edge_push_trajectory()

    # settle 前最后一帧（即 push 终点）
    push_end_frame = traj[-(80 + 1)]
    settle_first_frame = traj[-80]

    np.testing.assert_array_equal(
        settle_first_frame,
        push_end_frame,
        err_msg="Settle start doesn't match push endpoint",
    )


def test_original_push_structure_preserved():
    """原始的 approach + 3-pass push 结构不应被改变。"""
    from pta.envs.wrappers.joint_residual_wrapper import build_edge_push_trajectory

    traj = build_edge_push_trajectory()

    # 原始部分: 20(approach) + 30(extend→behind) + 3×100(push) + 2×30(return) = 410
    # settle 部分: 80
    # 总计: 490
    assert traj.shape[0] == 490, f"Expected 490 steps (410 push + 80 settle), got {traj.shape[0]}"
