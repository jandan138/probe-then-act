"""Canonical project paths."""

from __future__ import annotations

from pathlib import Path

#: Root directory of the PTA repository.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

#: Default directory for model checkpoints.
CHECKPOINT_DIR: Path = PROJECT_ROOT / "checkpoints"

#: Default directory for evaluation results.
RESULTS_DIR: Path = PROJECT_ROOT / "results"

#: Default directory for training logs.
LOG_DIR: Path = PROJECT_ROOT / "logs"

#: Default directory for configuration files.
CONFIG_DIR: Path = PROJECT_ROOT / "configs"

#: Default directory for exported figures.
FIGURES_DIR: Path = RESULTS_DIR / "figures"

#: Default directory for exported tables.
TABLES_DIR: Path = RESULTS_DIR / "tables"

#: Default directory for exported videos.
VIDEOS_DIR: Path = RESULTS_DIR / "videos"
