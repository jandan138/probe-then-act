"""Spill metrics -- Fraction of material lost outside containers."""

from __future__ import annotations


def compute_spill_ratio(
    spilled_particles: int,
    total_particles: int,
) -> float:
    """Fraction of particles that left both containers (spilled).

    Parameters
    ----------
    spilled_particles:
        Number of MPM particles outside both source and target
        container AABBs.
    total_particles:
        Total number of MPM particles in the scene.

    Returns
    -------
    float
        Spill ratio in [0, 1].  Lower is better.
    """
    raise NotImplementedError
