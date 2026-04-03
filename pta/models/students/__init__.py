"""Student models — student policy and distillation losses."""

from pta.models.students.distillation_losses import (
    behavior_cloning_loss,
    distillation_loss,
)
from pta.models.students.student_policy import StudentPolicy

__all__ = [
    "StudentPolicy",
    "distillation_loss",
    "behavior_cloning_loss",
]
