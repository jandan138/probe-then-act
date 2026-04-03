"""Distillation losses for teacher-to-student knowledge transfer."""

from __future__ import annotations

import torch


def distillation_loss(
    student_action: torch.Tensor,
    teacher_action: torch.Tensor,
    student_z: torch.Tensor,
    teacher_z: torch.Tensor,
    action_weight: float = 1.0,
    representation_weight: float = 0.1,
) -> torch.Tensor:
    """Compute the combined distillation loss.

    Matches both the output actions and the internal latent
    representations between teacher and student.

    Parameters
    ----------
    student_action : torch.Tensor
        Student action predictions, shape ``(B, action_dim)``.
    teacher_action : torch.Tensor
        Teacher action targets, shape ``(B, action_dim)``.
    student_z : torch.Tensor
        Student latent belief, shape ``(B, latent_dim)``.
    teacher_z : torch.Tensor
        Teacher latent representation, shape ``(B, latent_dim)``.
    action_weight : float
        Weight for the action-matching term.
    representation_weight : float
        Weight for the representation-matching term.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    raise NotImplementedError


def behavior_cloning_loss(
    predicted_action: torch.Tensor,
    target_action: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Standard behaviour-cloning (BC) loss.

    Parameters
    ----------
    predicted_action : torch.Tensor
        Model action predictions, shape ``(B, action_dim)``.
    target_action : torch.Tensor
        Demonstration action targets, shape ``(B, action_dim)``.
    reduction : str
        Reduction mode: ``"mean"`` | ``"sum"`` | ``"none"``.

    Returns
    -------
    torch.Tensor
        BC loss (scalar when reduction is ``"mean"`` or ``"sum"``).
    """
    raise NotImplementedError
