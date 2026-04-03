"""Test that observation shapes match the configured specification."""

from __future__ import annotations

from typing import Any

import pytest


class TestObservationShapes:
    """Verify observation tensor shapes against configuration."""

    def test_obs_shape_after_reset(self, dummy_env: Any, default_config: dict[str, Any]) -> None:
        """Observation returned by reset() should match the configured shape."""
        raise NotImplementedError

    def test_obs_shape_after_step(self, dummy_env: Any) -> None:
        """Observation returned by step() should match the reset() shape."""
        raise NotImplementedError

    def test_obs_contains_expected_keys(self, dummy_env: Any) -> None:
        """Dict observations should contain all keys declared in the config."""
        raise NotImplementedError

    def test_obs_dtype_is_float32(self, dummy_env: Any) -> None:
        """Observation tensors should have dtype float32."""
        raise NotImplementedError
