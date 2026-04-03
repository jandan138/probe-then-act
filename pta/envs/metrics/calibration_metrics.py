"""Calibration metrics -- Expected Calibration Error (ECE).

Measures how well the belief encoder's uncertainty estimates
correlate with actual prediction errors.  A well-calibrated model
should have low ECE.
"""

from __future__ import annotations

import numpy as np


def compute_expected_calibration_error(
    predicted_means: np.ndarray,
    predicted_stds: np.ndarray,
    ground_truth: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error for regression.

    Bins predictions by predicted confidence (1/std), then measures
    the gap between predicted and observed error within each bin.

    Parameters
    ----------
    predicted_means:
        Model predictions, shape ``(N, D)``.
    predicted_stds:
        Predicted standard deviations, shape ``(N, D)``.
    ground_truth:
        True values, shape ``(N, D)``.
    n_bins:
        Number of calibration bins.

    Returns
    -------
    float
        ECE value (lower is better, 0 = perfectly calibrated).
    """
    raise NotImplementedError
