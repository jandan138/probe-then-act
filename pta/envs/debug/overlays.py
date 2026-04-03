"""Metric overlays -- Render text/gauge overlays onto debug frames.

Used during development to visually inspect reward components,
particle counts, and task progress on rendered video frames.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def render_metric_overlay(
    frame: np.ndarray,
    metrics: Dict[str, float],
    position: tuple[int, int] = (10, 30),
    font_scale: float = 0.5,
) -> np.ndarray:
    """Draw metric text onto an RGB *frame* and return the annotated image.

    Parameters
    ----------
    frame:
        RGB image array, shape ``(H, W, 3)``, dtype ``uint8``.
    metrics:
        Metric name -> value dict to render.
    position:
        ``(x, y)`` pixel position for the first line of text.
    font_scale:
        Font size scaling factor.

    Returns
    -------
    np.ndarray
        Annotated copy of *frame*.
    """
    raise NotImplementedError
