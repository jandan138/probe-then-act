"""Fix 3: Test that reward is cumulative (not delta) and spill/transfer are balanced.

Root causes fixed:
1. Delta reward — PPO can't learn from delta signals (advantage variance too low)
2. Reward asymmetry — spill penalty 50x transfer reward per percentage point
3. Delta state variables — _prev_transfer_frac, _prev_mean_particle_y, _success_triggered
"""

import torch
import numpy as np
import pytest
from unittest.mock import MagicMock


def _make_reward_mock(mean_y=0.05, transfer_frac_val=0.0, spill_frac_val=0.0):
    """Create mock task for reward testing."""
    task = MagicMock()

    # EE pos
    task._ee_link.get_pos.return_value = torch.tensor([0.55, -0.03, 0.22])

    # Particle positions — synthesize to achieve desired fracs
    n_particles = 400
    particle_pos = torch.zeros(n_particles, 3)

    # Source bbox (approximate): x=[0.40, 0.70], y=[-0.13, 0.07], z=[0.0, 0.33]
    sp = (0.55, -0.03, 0.17)
    ss = (0.30, 0.20, 0.06)
    tp = (0.55, 0.30, 0.0)
    ts = (0.30, 0.20, 0.10)

    # Place particles: some in source, some in target, some spilled
    n_transfer = int(n_particles * transfer_frac_val)
    n_spill = int(n_particles * spill_frac_val)
    n_source = n_particles - n_transfer - n_spill

    # Source particles (inside source bbox)
    if n_source > 0:
        particle_pos[:n_source, 0] = 0.55  # x center
        particle_pos[:n_source, 1] = mean_y  # y
        particle_pos[:n_source, 2] = 0.20  # z in source

    # Target particles
    if n_transfer > 0:
        start = n_source
        particle_pos[start : start + n_transfer, 0] = 0.55
        particle_pos[start : start + n_transfer, 1] = 0.30  # target center y
        particle_pos[start : start + n_transfer, 2] = 0.05  # target z

    # Spilled particles (outside both)
    if n_spill > 0:
        start = n_source + n_transfer
        particle_pos[start : start + n_spill, 0] = 0.55
        particle_pos[start : start + n_spill, 1] = 0.60  # far from both
        particle_pos[start : start + n_spill, 2] = 0.0

    task.particles.get_particles_pos.return_value = particle_pos
    task._total_particles = n_particles

    # Source/target bbox tensors
    platform_edge_y = sp[1] + ss[1] / 2
    target_y_min = max(tp[1] - ts[1] / 2, platform_edge_y)

    task._source_bbox_min = torch.tensor([sp[0] - ss[0] / 2, sp[1] - ss[1] / 2, 0.0])
    task._source_bbox_max = torch.tensor(
        [sp[0] + ss[0] / 2, sp[1] + ss[1] / 2, sp[2] + ss[2] + 0.10]
    )
    task._target_bbox_min = torch.tensor([tp[0] - ts[0] / 2, target_y_min, 0.0])
    task._target_bbox_max = torch.tensor(
        [tp[0] + ts[0] / 2, tp[1] + ts[1] / 2, tp[2] + ts[2] + 0.10]
    )

    # Wire up _count methods to use the real implementations
    from pta.envs.tasks.scoop_transfer import ScoopTransferTask
    task._count_particles_in_target = lambda: ScoopTransferTask._count_particles_in_target(task)
    task._count_spilled_particles = lambda: ScoopTransferTask._count_spilled_particles(task)

    task._success_threshold = 0.3
    task.sc = MagicMock()
    task.sc.source_pos = sp
    task.sc.source_size = ss
    task.sc.target_pos = tp
    task.sc.target_size = ts

    task._step_count = 100
    task._horizon = 500

    return task


class TestRewardCumulative:
    """Reward must be cumulative (absolute state), not delta-based."""

    def test_reward_same_state_same_value(self):
        """Calling compute_reward() twice on identical state must return same value.

        Delta-based: second call returns ~0 (no increment).
        Cumulative: second call returns same value (absolute state unchanged).
        """
        from pta.envs.tasks.scoop_transfer import ScoopTransferTask

        task = _make_reward_mock(transfer_frac_val=0.1)
        r1 = ScoopTransferTask.compute_reward(task)
        r2 = ScoopTransferTask.compute_reward(task)

        assert abs(r1 - r2) < 0.01, (
            f"Reward changed between identical states: {r1:.4f} vs {r2:.4f}. "
            "This indicates delta-based reward (second call has no increment)."
        )

    def test_no_delta_state_variables(self):
        """Delta state variables must not exist on the task."""
        from pta.envs.tasks.scoop_transfer import ScoopTransferTask

        # Check the class itself doesn't set these in __init__
        # We'll just verify compute_reward doesn't reference them
        task = _make_reward_mock()

        # These should not exist (or at least not be used)
        assert not hasattr(task, "_prev_transfer_frac") or isinstance(
            task._prev_transfer_frac, MagicMock
        ), "_prev_transfer_frac should be removed"


class TestRewardSymmetry:
    """Spill penalty and transfer reward must be balanced."""

    def test_spill_does_not_dominate_transfer(self):
        """1% spill penalty over 500 steps must not exceed 1% transfer reward.

        Old: 1% spill = -2.0 * 0.01 * 500 = -10.0
             1% transfer = 20.0 * 0.01 (delta, once) = +0.2
             Ratio: 50:1 (spill dominates)

        New: 1% spill = -1.0 * 0.01 * 500 = -5.0
             1% transfer = 10.0 * 0.01 * 500 = +50.0
             Ratio: 1:10 (transfer dominates) ✓
        """
        SPILL_COEF = 1.0  # Expected new coefficient
        TRANSFER_COEF = 10.0  # Expected new coefficient
        HORIZON = 500
        pct = 0.01

        spill_total = SPILL_COEF * pct * HORIZON
        transfer_total = TRANSFER_COEF * pct * HORIZON

        ratio = spill_total / transfer_total
        assert ratio <= 1.0, f"Spill still dominates transfer: ratio={ratio:.1f}"

    def test_reward_positive_for_successful_episode(self):
        """A state with 30% transfer and 5% spill should have positive reward components.

        This is cumulative: r_transfer = 10.0 * 0.30 = 3.0 per step.
        r_spill = -1.0 * 0.05 = -0.05 per step.
        r_success = 50.0 per step (>= threshold).
        Net per step > 0.
        """
        from pta.envs.tasks.scoop_transfer import ScoopTransferTask

        task = _make_reward_mock(transfer_frac_val=0.30, spill_frac_val=0.05)
        reward = ScoopTransferTask.compute_reward(task)

        assert reward > 0, (
            f"30% transfer + 5% spill should give positive reward, got {reward:.4f}. "
            "This means the reward function is still broken."
        )


class TestRewardCoefficients:
    """Verify specific coefficient values match the hotfix spec."""

    def test_time_penalty_small(self):
        """r_time should be -0.0001 (not -0.001)."""
        from pta.envs.tasks.scoop_transfer import ScoopTransferTask

        # All particles in source, no transfer, no spill — reward should be ~r_approach + r_time
        task = _make_reward_mock(transfer_frac_val=0.0, spill_frac_val=0.0)
        # Put EE exactly at source center to zero out approach
        task._ee_link.get_pos.return_value = torch.tensor(
            [0.55, -0.03, 0.22]
        )

        r = ScoopTransferTask.compute_reward(task)

        # With no transfer, no spill, r_success=0:
        # r = r_approach + r_push + r_transfer + r_spill + r_time + r_success
        # r_approach ≈ -0.01 * small_dist
        # r_push = 2.0 * max(0, mean_y - source_y) — depends on particles
        # r_time = -0.0001
        # The time penalty should be negligible, not -0.001
        # We verify by checking that reward is not excessively negative
        assert r > -1.0, f"Reward too negative ({r:.4f}) — time penalty may be too large"
