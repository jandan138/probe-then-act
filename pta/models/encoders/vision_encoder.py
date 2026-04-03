"""Vision encoder — maps RGB / depth images to a compact feature vector."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Encode RGB and (optionally) depth images into a feature vector.

    Parameters
    ----------
    in_channels : int
        Number of input channels (3 for RGB, 4 for RGB-D).
    feature_dim : int
        Dimensionality of the output feature vector.
    backbone : str
        Name of the CNN / ViT backbone to use (e.g. ``"resnet18"``).
    pretrained : bool
        Whether to initialise the backbone with pre-trained weights.
    """

    def __init__(
        self,
        in_channels: int = 3,
        feature_dim: int = 256,
        backbone: str = "resnet18",
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.feature_dim = feature_dim
        self.backbone_name = backbone
        self.pretrained = pretrained
        raise NotImplementedError

    def forward(
        self,
        rgb: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Produce a feature vector from visual input.

        Parameters
        ----------
        rgb : torch.Tensor
            RGB image tensor of shape ``(B, 3, H, W)``.
        depth : torch.Tensor, optional
            Depth image tensor of shape ``(B, 1, H, W)``.

        Returns
        -------
        torch.Tensor
            Feature vector of shape ``(B, feature_dim)``.
        """
        raise NotImplementedError
