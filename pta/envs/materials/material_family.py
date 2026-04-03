"""MaterialFamily enum and MaterialParams dataclass.

These define the canonical material taxonomy used throughout the
project.  Stable integer IDs are assigned so that train/test splits
remain reproducible across code changes.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Dict


class MaterialFamily(enum.IntEnum):
    """Canonical material families backed by Genesis MPM solvers.

    Integer values are stable IDs used in train/test split definitions.
    Do **not** re-number existing entries.
    """

    SAND = 0
    SNOW = 1
    ELASTOPLASTIC = 2
    LIQUID = 3


@dataclass
class MaterialParams:
    """Physical parameters for an MPM material instance.

    Attributes
    ----------
    family:
        Which :class:`MaterialFamily` this belongs to.
    E:
        Young's modulus (Pa).
    nu:
        Poisson's ratio.
    rho:
        Density (kg/m^3).
    friction_angle:
        Internal friction angle in degrees (granular materials).
    cohesion:
        Cohesion strength (Pa).
    viscosity:
        Dynamic viscosity (Pa*s, liquids only).
    extra:
        Any additional solver-specific parameters.
    """

    family: MaterialFamily
    E: float = 1e4
    nu: float = 0.2
    rho: float = 1500.0
    friction_angle: float = 25.0
    cohesion: float = 0.0
    viscosity: float = 0.0
    extra: Dict[str, float] = field(default_factory=dict)
