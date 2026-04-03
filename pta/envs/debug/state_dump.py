"""Dump full simulator state for debugging."""

from pathlib import Path


def dump_state(scene, path: Path) -> None:
    """Save full Genesis scene state (particles, rigid bodies, joints) to disk."""
    raise NotImplementedError


def load_state(scene, path: Path) -> None:
    """Restore Genesis scene state from disk."""
    raise NotImplementedError
