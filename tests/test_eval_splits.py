"""Test that OOD evaluation splits exclude training-set IDs."""

from __future__ import annotations

import pytest


class TestEvalSplits:
    """Verify OOD split correctness."""

    def test_ood_material_excludes_training_materials(self) -> None:
        """OOD material split should contain no training-set material IDs."""
        raise NotImplementedError

    def test_ood_tool_excludes_training_tools(self) -> None:
        """OOD tool split should contain no training-set tool IDs."""
        raise NotImplementedError

    def test_ood_container_excludes_training_containers(self) -> None:
        """OOD container split should contain no training-set container IDs."""
        raise NotImplementedError

    def test_ood_sensor_excludes_training_sensors(self) -> None:
        """OOD sensor split should contain no training-set sensor configs."""
        raise NotImplementedError

    def test_id_split_uses_only_training_ids(self) -> None:
        """ID split should only reference IDs present in the training set."""
        raise NotImplementedError

    def test_splits_are_non_empty(self) -> None:
        """Every evaluation split should contain at least one configuration."""
        raise NotImplementedError
