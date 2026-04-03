"""Experiment logger — unified interface for TensorBoard / W&B / CSV."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentLogger:
    """Unified experiment logger that dispatches to multiple backends.

    Supports TensorBoard, Weights-and-Biases, and a simple CSV fallback.

    Parameters
    ----------
    log_dir : Path
        Root directory for log artefacts.
    project : str
        Project name (used by W&B).
    run_name : str, optional
        Human-readable run identifier.
    backends : list[str]
        Logging backends to enable.  Accepted values:
        ``"tensorboard"``, ``"wandb"``, ``"csv"``.
    """

    def __init__(
        self,
        log_dir: Path,
        project: str = "pta",
        run_name: Optional[str] = None,
        backends: Optional[list[str]] = None,
    ) -> None:
        self.log_dir = log_dir
        self.project = project
        self.run_name = run_name
        self.backends = backends or ["tensorboard", "csv"]
        raise NotImplementedError

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
    ) -> None:
        """Log a single scalar value.

        Parameters
        ----------
        tag : str
            Metric name (e.g. ``"train/reward_mean"``).
        value : float
            Scalar value.
        step : int
            Global training step.
        """
        raise NotImplementedError

    def log_scalars(
        self,
        tag_value_map: Dict[str, float],
        step: int,
    ) -> None:
        """Log multiple scalar values at once.

        Parameters
        ----------
        tag_value_map : dict[str, float]
            Mapping from metric names to values.
        step : int
            Global training step.
        """
        raise NotImplementedError

    def log_config(self, config: Dict[str, Any]) -> None:
        """Persist the full experiment configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary.
        """
        raise NotImplementedError

    def close(self) -> None:
        """Flush buffers and close all logging backends."""
        raise NotImplementedError
