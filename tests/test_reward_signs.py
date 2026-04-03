"""Test that rewards are bounded and have the correct sign."""

from __future__ import annotations

from typing import Any

import pytest


class TestRewardSigns:
    """Verify reward signal properties."""

    def test_reward_is_finite(self, dummy_env: Any) -> None:
        """Rewards should never be NaN or Inf."""
        raise NotImplementedError

    def test_reward_bounded(self, dummy_env: Any) -> None:
        """Rewards should stay within a reasonable range."""
        raise NotImplementedError

    def test_success_gives_positive_reward(self, dummy_env: Any) -> None:
        """A successful episode should yield positive cumulative reward."""
        raise NotImplementedError

    def test_failure_does_not_give_positive_reward(self, dummy_env: Any) -> None:
        """A failed episode should not yield a large positive reward."""
        raise NotImplementedError
