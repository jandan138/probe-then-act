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

    def create_material(self, family: str, params: Dict[str, Any] | None = None) -> Any:
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
        family_lower = family.lower()
        if family_lower not in self._FAMILY_MAP:
            raise ValueError(
                f"Unknown material family '{family}'. "
                f"Supported: {list(self._FAMILY_MAP.keys())}"
            )
        constructor = self._FAMILY_MAP[family_lower]
        if params is None:
            params = {}
        return constructor(**params)

    def create_coupled_rigid(
        self,
        coup_friction: float = 0.5,
        needs_coup: bool = True,
        fixed: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Create a Rigid material configured for MPM coupling.

        Parameters
        ----------
        coup_friction:
            Friction coefficient for rigid-MPM coupling.
        needs_coup:
            Whether this rigid body participates in MPM coupling.
        fixed:
            Whether the rigid body is fixed in place (unused here,
            fixedness is set on the morph).
        **kwargs:
            Additional keyword arguments for ``gs.materials.Rigid``.
        """
        return gs.materials.Rigid(
            needs_coup=needs_coup,
            coup_friction=coup_friction,
            **kwargs,
        )
