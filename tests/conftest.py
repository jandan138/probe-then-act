"""Shared pytest fixtures for the PTA test suite."""

from __future__ import annotations

from typing import Any, Dict

import pytest


@pytest.fixture
def default_config() -> Dict[str, Any]:
    """Return a minimal default configuration for testing.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary with sensible defaults for env,
        policy, and training sections.
    """
    raise NotImplementedError


@pytest.fixture
def dummy_env() -> Any:
    """Create a lightweight dummy environment for unit tests.

    Returns
    -------
    Any
        A Gym-compatible environment with a small observation and
        action space suitable for fast testing.
    """
    raise NotImplementedError


@pytest.fixture
def dummy_policy() -> Any:
    """Create a randomly-initialised dummy policy for testing.

    Returns
    -------
    Any
        A policy that accepts observations from ``dummy_env`` and
        produces valid actions.
    """
    raise NotImplementedError


@pytest.fixture
def sample_rollout_data() -> Dict[str, Any]:
    """Generate a small synthetic rollout dataset.

    Returns
    -------
    dict[str, Any]
        Dictionary with keys ``obs``, ``actions``, ``rewards``,
        ``dones`` containing small tensor arrays.
    """
    raise NotImplementedError
