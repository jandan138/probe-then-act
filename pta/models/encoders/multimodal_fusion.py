"""Multimodal fusion — combine vision, tactile, and proprioception features."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MultimodalFusion(nn.Module):
    """Fuse feature vectors from vision, tactile, and proprioception encoders.

    Supports several fusion strategies (concatenation, gated, attention-based).

    Parameters
    ----------
    vision_dim : int
        Dimensionality of the vision feature vector.
    tactile_dim : int
        Dimensionality of the tactile feature vector.
    proprio_dim : int
        Dimensionality of the proprioception feature vector.
    fused_dim : int
        Dimensionality of the output fused representation.
    fusion_type : str
        Fusion strategy: ``"concat"`` | ``"gated"`` | ``"attention"``.
    """

    def __init__(
        self,
        vision_dim: int = 256,
        tactile_dim: int = 128,
        proprio_dim: int = 64,
        fused_dim: int = 256,
        fusion_type: str = "concat",
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.tactile_dim = tactile_dim
        self.proprio_dim = proprio_dim
        self.fused_dim = fused_dim
        self.fusion_type = fusion_type
        raise NotImplementedError

    def forward(
        self,
        vision_feat: torch.Tensor,
        tactile_feat: Optional[torch.Tensor] = None,
        proprio_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Fuse multimodal features.

        Parameters
        ----------
        vision_feat : torch.Tensor
            Vision features of shape ``(B, vision_dim)``.
        tactile_feat : torch.Tensor, optional
            Tactile features of shape ``(B, tactile_dim)``.
        proprio_feat : torch.Tensor, optional
            Proprioception features of shape ``(B, proprio_dim)``.

        Returns
        -------
        torch.Tensor
            Fused feature vector of shape ``(B, fused_dim)``.
        """
        raise NotImplementedError
