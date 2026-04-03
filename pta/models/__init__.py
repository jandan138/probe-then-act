"""PTA Models — Probe-Then-Act architecture components.

Submodules:
    encoders  – vision, tactile, proprioception, and multimodal fusion
    probe     – probe policy and action-space definitions
    belief    – latent belief encoder, uncertainty, and auxiliary heads
    policy    – task policy, action head, risk head
    teachers  – privileged teacher (sim-only, full state access)
    students  – student policy wrapper and distillation losses
"""

from pta.models.encoders import (
    MultimodalFusion,
    ProprioceptionEncoder,
    TactileEncoder,
    VisionEncoder,
)
from pta.models.probe import ProbeAction, ProbePolicy
from pta.models.belief import (
    LatentBeliefEncoder,
    UncertaintyHead,
    MaterialPredictionHead,
    DynamicsPredictionHead,
)
from pta.models.policy import ActionHead, RiskHead, TaskPolicy
from pta.models.teachers import PrivilegedTeacher
from pta.models.students import StudentPolicy

__all__ = [
    # encoders
    "VisionEncoder",
    "TactileEncoder",
    "ProprioceptionEncoder",
    "MultimodalFusion",
    # probe
    "ProbePolicy",
    "ProbeAction",
    # belief
    "LatentBeliefEncoder",
    "UncertaintyHead",
    "MaterialPredictionHead",
    "DynamicsPredictionHead",
    # policy
    "TaskPolicy",
    "ActionHead",
    "RiskHead",
    # teachers
    "PrivilegedTeacher",
    # students
    "StudentPolicy",
]
