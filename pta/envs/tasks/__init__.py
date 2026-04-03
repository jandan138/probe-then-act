"""pta.envs.tasks -- Task definitions (reset, step, reward, done)."""

from pta.envs.tasks.base_task import BaseTask
from pta.envs.tasks.scoop_transfer import ScoopTransferTask
from pta.envs.tasks.level_fill import LevelFillTask

__all__ = [
    "BaseTask",
    "ScoopTransferTask",
    "LevelFillTask",
]
