"""Factory for Genesis MPM material objects.

Maps a :class:`MaterialFamily` and :class:`MaterialParams` to the
appropriate ``gs.materials.MPM.*`` constructor call.
"""

from __future__ import annotations

from typing import Any

import genesis as gs

from pta.envs.materials.material_family import MaterialFamily, MaterialParams


def create_mpm_material(family: MaterialFamily, params: MaterialParams) -> Any:
    """Instantiate a Genesis MPM material.

    Parameters
    ----------
    family:
        The material family enum (Sand, Snow, ElastoPlastic, Liquid).
    params:
        Physical parameters forwarded to the constructor.

    Returns
    -------
    material
        A Genesis MPM material object (e.g. ``gs.materials.MPM.Sand(...)``).

    Raises
    ------
    ValueError
        If *family* is not a recognised MPM material family.
    """
    raise NotImplementedError
