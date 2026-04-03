"""Experiment logger -- unified interface for TensorBoard / CSV / stdout.

Provides a simple, consistent API for logging scalars and configs
during RL training.  TensorBoard is the primary backend; CSV is a
lightweight fallback that always runs.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentLogger:
    """Unified experiment logger that dispatches to multiple backends.

    Supports TensorBoard and a simple CSV fallback.

    Parameters
    ----------
    log_dir : Path
        Root directory for log artefacts.
    project : str
        Project name (used as a label / prefix).
    run_name : str, optional
        Human-readable run identifier.
    backends : list[str]
        Logging backends to enable.  Accepted values:
        ``"tensorboard"``, ``"csv"``, ``"stdout"``.
    """

    def __init__(
        self,
        log_dir: Path,
        project: str = "pta",
        run_name: Optional[str] = None,
        backends: Optional[List[str]] = None,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.project = project
        self.run_name = run_name or "run"
        self.backends = backends or ["tensorboard", "csv"]

        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self._tb_writer = None
        if "tensorboard" in self.backends:
            try:
                from torch.utils.tensorboard import SummaryWriter
                tb_dir = self.log_dir / "tensorboard" / self.run_name
                tb_dir.mkdir(parents=True, exist_ok=True)
                self._tb_writer = SummaryWriter(log_dir=str(tb_dir))
            except ImportError:
                print(
                    "WARNING: tensorboard not installed, disabling TB logging",
                    file=sys.stderr,
                )

        # CSV writer
        self._csv_path: Optional[Path] = None
        self._csv_file = None
        self._csv_writer = None
        self._csv_fields: List[str] = []
        if "csv" in self.backends:
            self._csv_path = self.log_dir / f"{self.run_name}_metrics.csv"

        # Stdout
        self._stdout = "stdout" in self.backends

        # Track logged tags for CSV header management
        self._seen_tags: set = set()

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
        if self._tb_writer is not None:
            self._tb_writer.add_scalar(tag, value, step)

        if self._csv_path is not None:
            self._write_csv_row({tag: value}, step)

        if self._stdout:
            print(f"[step {step:>8d}] {tag} = {value:.6f}")

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
        if self._tb_writer is not None:
            for tag, value in tag_value_map.items():
                self._tb_writer.add_scalar(tag, value, step)

        if self._csv_path is not None:
            self._write_csv_row(tag_value_map, step)

        if self._stdout:
            parts = [f"{tag}={value:.4f}" for tag, value in tag_value_map.items()]
            print(f"[step {step:>8d}] {' | '.join(parts)}")

    def log_config(self, config: Dict[str, Any]) -> None:
        """Persist the full experiment configuration.

        Parameters
        ----------
        config : dict[str, Any]
            Configuration dictionary.
        """
        config_path = self.log_dir / f"{self.run_name}_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)

        if self._tb_writer is not None:
            # Log config as text to TensorBoard
            config_str = json.dumps(config, indent=2, default=str)
            self._tb_writer.add_text("config", f"```json\n{config_str}\n```", 0)

    def close(self) -> None:
        """Flush buffers and close all logging backends."""
        if self._tb_writer is not None:
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write_csv_row(self, tag_value_map: Dict[str, float], step: int) -> None:
        """Append a row to the CSV log, creating the file if needed."""
        assert self._csv_path is not None

        new_tags = set(tag_value_map.keys()) - self._seen_tags
        needs_new_header = len(new_tags) > 0

        if needs_new_header:
            self._seen_tags.update(new_tags)
            self._csv_fields = ["step"] + sorted(self._seen_tags)

            # If file is open, close it so we can rewrite the header
            if self._csv_file is not None:
                self._csv_file.close()
                self._csv_file = None
                self._csv_writer = None

        if self._csv_writer is None:
            mode = "a" if self._csv_path.exists() and not needs_new_header else "w"
            self._csv_file = open(self._csv_path, mode, newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self._csv_fields,
                extrasaction="ignore",
            )
            if mode == "w" or not self._csv_path.exists():
                self._csv_writer.writeheader()

        row = {"step": step}
        row.update(tag_value_map)
        self._csv_writer.writerow(row)
        self._csv_file.flush()
