"""MaterialBuilder -- Creates Genesis MPM / Rigid materials.

Wraps the Genesis material API so that the rest of the codebase only needs
to specify a ``(family, params)`` pair instead of constructing low-level
material objects directly.

Supported families: Sand, Snow, ElastoPlastic, Liquid, Rigid.
"""

from __future__ import annotations

from typing import Any, Dict

import genesis as gs


class MaterialBuilder:
    """Factory for Genesis material objects."""

    #: Mapping from family name to ``gs.materials`` constructor.
    _FAMILY_MAP: Dict[str, Any] = {
        "sand": gs.materials.MPM.Sand,
        "snow": gs.materials.MPM.Snow,
        "elastoplastic": gs.materials.MPM.ElastoPlastic,
        "liquid": gs.materials.MPM.Liquid,
        "rigid": gs.materials.Rigid,
    }

    def create_material(self, family: str, params: Dict[str, Any]) -> Any:
        """Instantiate a Genesis material for *family* with *params*.

        Parameters
        ----------
        family:
            One of ``"sand"``, ``"snow"``, ``"elastoplastic"``,
            ``"liquid"``, ``"rigid"``.
        params:
            Keyword arguments forwarded to the Genesis material
            constructor (e.g. ``E``, ``nu``, ``rho``).

        Returns
        -------
        material
            A Genesis material object ready to pass to
            ``scene.add_entity(material=...)``.

        Raises
        ------
        ValueError
            If *family* is not in :attr:`_FAMILY_MAP`.
        """
        raise NotImplementedError
