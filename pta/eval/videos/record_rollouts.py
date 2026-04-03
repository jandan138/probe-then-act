"""Record evaluation rollouts as video files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


def record_rollout_video(
    policy: Any,
    env: Any,
    output_path: Path,
    n_episodes: int = 1,
    fps: int = 30,
    resolution: Optional[tuple[int, int]] = None,
) -> Path:
    """Record a policy rollout as an MP4 video.

    Parameters
    ----------
    policy : Any
        Trained task policy.
    env : Any
        Gym-compatible environment with a render method.
    output_path : Path
        Destination file path for the video.
    n_episodes : int
        Number of episodes to record.
    fps : int
        Frames per second for the output video.
    resolution : tuple[int, int], optional
        (width, height) resolution.  Uses the environment default if
        not specified.

    Returns
    -------
    Path
        Path to the saved video file.
    """
    raise NotImplementedError
