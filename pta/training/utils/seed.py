"""Deterministic seeding for reproducible experiments."""

from __future__ import annotations


def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Also configures CuDNN for deterministic behaviour when a CUDA device
    is available.

    Parameters
    ----------
    seed : int
        Integer seed value.
    """
    raise NotImplementedError
