"""Create a video montage from multiple rollout recordings."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def create_montage(
    video_paths: list[Path],
    output_path: Path,
    grid: Optional[tuple[int, int]] = None,
    fps: int = 30,
) -> Path:
    """Tile multiple videos into a single montage grid.

    Parameters
    ----------
    video_paths : list[Path]
        Paths to input video files.
    output_path : Path
        Destination file path for the montage video.
    grid : tuple[int, int], optional
        (rows, cols) layout.  Inferred automatically if not specified.
    fps : int
        Frames per second for the output montage.

    Returns
    -------
    Path
        Path to the saved montage video file.
    """
    raise NotImplementedError
