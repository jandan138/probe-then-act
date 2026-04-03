"""Tactile encoder — maps raw tactile traces to a feature vector."""

from __future__ import annotations

import torch
import torch.nn as nn


class TactileEncoder(nn.Module):
    """Encode a tactile-sensor trace into a compact feature vector.

    A *trace* is a temporal sequence of tactile readings collected during a
    single probe interaction (tap, press, or drag).

    Parameters
    ----------
    input_dim : int
        Dimensionality of a single tactile reading (e.g. taxel count).
    feature_dim : int
        Dimensionality of the output feature vector.
    seq_model : str
        Temporal model type: ``"lstm"`` | ``"transformer"`` | ``"1d_conv"``.
    """

    def __init__(
        self,
        input_dim: int = 64,
        feature_dim: int = 128,
        seq_model: str = "lstm",
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.seq_model = seq_model
        raise NotImplementedError

    def forward(self, tactile_trace: torch.Tensor) -> torch.Tensor:
        """Encode a tactile trace.

        Parameters
        ----------
        tactile_trace : torch.Tensor
            Tensor of shape ``(B, T, input_dim)`` where *T* is the number of
            time-steps in the trace.

        Returns
        -------
        torch.Tensor
            Feature vector of shape ``(B, feature_dim)``.
        """
        raise NotImplementedError
