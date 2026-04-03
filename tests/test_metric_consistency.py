"""Test that computed metrics are consistent with raw rollout data."""

from __future__ import annotations

from typing import Any, Dict

import pytest


class TestMetricConsistency:
    """Verify correctness of evaluation metrics."""

    def test_success_rate_in_zero_one(self, sample_rollout_data: Dict[str, Any]) -> None:
        """Success rate should be between 0 and 1."""
        raise NotImplementedError

    def test_mean_return_matches_manual_sum(self, sample_rollout_data: Dict[str, Any]) -> None:
        """Mean return should equal the manual average of episode returns."""
        raise NotImplementedError

    def test_episode_length_positive(self, sample_rollout_data: Dict[str, Any]) -> None:
        """Mean episode length should be a positive integer."""
        raise NotImplementedError

    def test_metric_keys_present(self, sample_rollout_data: Dict[str, Any]) -> None:
        """The metrics dict should contain all expected keys."""
        raise NotImplementedError
