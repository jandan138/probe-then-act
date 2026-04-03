"""Shared type aliases used across the PTA codebase."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple, Union

import torch

#: A nested configuration dictionary (e.g. parsed from YAML).
Config = Dict[str, Any]

#: A single observation — either a tensor or a dict of named tensors.
Observation = Union[torch.Tensor, Dict[str, torch.Tensor]]

#: An action vector.
Action = torch.Tensor

#: Shape specification for observation spaces.
ObsShape = Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]

#: Evaluation metrics dictionary.
Metrics = Dict[str, float]

#: A batch of transitions used for training.
Batch = Dict[str, torch.Tensor]

#: Device specification accepted by PyTorch.
DeviceLike = Union[str, torch.device]
