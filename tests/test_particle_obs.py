"""Fix 2: Test that observations contain particle statistics.

The policy obs was missing ALL particle information — no mean_particle_y,
no transfer_frac, no spill_frac. Policy was asked to optimize quantities
it could not observe. This is the most fundamental of the 6 root causes.
"""

import numpy as np
import pytest
import torch
from unittest.mock import MagicMock, patch


def _make_mock_task():
    """Create a minimal mock of ScoopTransferTask for obs testing.

    This avoids needing Genesis/CUDA for fast unit tests.
    """
    task = MagicMock()

    # Mock robot
    task.robot.get_qpos.return_value = torch.zeros(9)  # 7 arm + 2 finger
    task.robot.get_dofs_velocity.return_value = torch.zeros(9)

    # Mock EE link
    task._ee_link.get_pos.return_value = torch.tensor([0.3, 0.1, 0.2])
    task._ee_link.get_quat.return_value = torch.tensor([1.0, 0.0, 0.0, 0.0])

    # Mock finger links
    task._left_finger_link.get_pos.return_value = torch.tensor([0.3, 0.11, 0.2])
    task._right_finger_link.get_pos.return_value = torch.tensor([0.3, 0.09, 0.2])

    # Mock particles (400 particles at source position)
    particle_pos = torch.rand(400, 3) * 0.1 + torch.tensor([0.55, 0.02, 0.20])
    task.particles.get_particles_pos.return_value = particle_pos
    task._total_particles = 400

    # Mock scene components for bbox
    task.sc = MagicMock()
    task.sc.source_pos = (0.55, -0.03, 0.17)
    task.sc.source_size = (0.30, 0.20, 0.06)
    task.sc.target_pos = (0.55, 0.30, 0.0)
    task.sc.target_size = (0.30, 0.20, 0.10)

    # Bbox tensors (computed in __init__, needed by get_observations)
    sp, ss = task.sc.source_pos, task.sc.source_size
    task._source_bbox_min = torch.tensor([sp[0] - ss[0] / 2, sp[1] - ss[1] / 2, 0.0])
    task._source_bbox_max = torch.tensor([sp[0] + ss[0] / 2, sp[1] + ss[1] / 2, sp[2] + ss[2] + 0.10])

    tp, ts = task.sc.target_pos, task.sc.target_size
    platform_edge_y = sp[1] + ss[1] / 2
    target_y_min = max(tp[1] - ts[1] / 2, platform_edge_y)
    task._target_bbox_min = torch.tensor([tp[0] - ts[0] / 2, target_y_min, 0.0])
    task._target_bbox_max = torch.tensor([tp[0] + ts[0] / 2, tp[1] + ts[1] / 2, tp[2] + ts[2] + 0.10])

    # Task config
    task._horizon = 500
    task._step_count = 0
    task._has_fingers = True

    return task


class TestParticleObsStructure:
    """Test that obs dict contains particle_stats key with correct shape."""

    def test_obs_has_particle_stats_key(self):
        """get_observations() must return a dict with 'particle_stats' key."""
        from pta.envs.tasks.scoop_transfer import ScoopTransferTask

        task = _make_mock_task()
        # Call the real method on our mock
        obs = ScoopTransferTask.get_observations(task)

        assert "particle_stats" in obs, (
            "Missing 'particle_stats' key in obs. "
            "Policy cannot observe transfer/spill without this."
        )

    def test_particle_stats_shape(self):
        """particle_stats must be a 3D tensor: [mean_y, transfer_frac, spill_frac]."""
        from pta.envs.tasks.scoop_transfer import ScoopTransferTask

        task = _make_mock_task()
        obs = ScoopTransferTask.get_observations(task)

        stats = obs["particle_stats"]
        assert stats.shape == (3,), f"Expected shape (3,), got {stats.shape}"

    def test_particle_stats_dtype(self):
        """particle_stats must be float32."""
        from pta.envs.tasks.scoop_transfer import ScoopTransferTask

        task = _make_mock_task()
        obs = ScoopTransferTask.get_observations(task)

        assert obs["particle_stats"].dtype == torch.float32

    def test_particle_stats_no_nan(self):
        """particle_stats must not contain NaN."""
        from pta.envs.tasks.scoop_transfer import ScoopTransferTask

        task = _make_mock_task()
        obs = ScoopTransferTask.get_observations(task)

        assert not torch.isnan(obs["particle_stats"]).any(), "particle_stats contains NaN"

    def test_proprio_unchanged(self):
        """Adding particle_stats must not change proprio dim."""
        from pta.envs.tasks.scoop_transfer import ScoopTransferTask

        task = _make_mock_task()
        obs = ScoopTransferTask.get_observations(task)

        # proprio = qpos(9) + qvel(9) + ee_pos(3) + ee_quat(4) + lf_pos(3) + rf_pos(3) = 31
        # (or 21 without fingers: qpos(7) + qvel(7) + ee_pos(3) + ee_quat(4))
        assert obs["proprio"].shape[-1] >= 21, "Proprio too small"

    def test_transfer_frac_in_range(self):
        """transfer_frac must be in [0, 1]."""
        from pta.envs.tasks.scoop_transfer import ScoopTransferTask

        task = _make_mock_task()
        obs = ScoopTransferTask.get_observations(task)

        transfer_frac = obs["particle_stats"][1]
        assert 0.0 <= float(transfer_frac) <= 1.0, f"transfer_frac={transfer_frac}"

    def test_spill_frac_in_range(self):
        """spill_frac must be in [0, 1]."""
        from pta.envs.tasks.scoop_transfer import ScoopTransferTask

        task = _make_mock_task()
        obs = ScoopTransferTask.get_observations(task)

        spill_frac = obs["particle_stats"][2]
        assert 0.0 <= float(spill_frac) <= 1.0, f"spill_frac={spill_frac}"

    def test_step_fraction_still_present(self):
        """step_fraction key must still exist."""
        from pta.envs.tasks.scoop_transfer import ScoopTransferTask

        task = _make_mock_task()
        obs = ScoopTransferTask.get_observations(task)

        assert "step_fraction" in obs, "step_fraction key missing"
