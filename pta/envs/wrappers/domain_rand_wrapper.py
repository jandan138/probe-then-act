"""Domain randomization wrapper for the Genesis Scoop-and-Transfer env.

Randomises material parameters on each ``reset()`` call by rebuilding
the environment with a freshly sampled material.  This is used for the
Domain-Randomisation PPO baseline (M3).

Since Genesis requires scene.build() to change materials, this wrapper
creates an entirely new GenesisGymWrapper on each reset with new
scene_config.  This is expensive but necessary for material changes.

For a lighter-weight approach that does not change materials, consider
randomising only dynamics parameters post-build (not supported by
Genesis MPM in v1).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import gymnasium
import numpy as np

from pta.envs.materials.material_family import MaterialFamily


# ---------------------------------------------------------------------------
# Training ranges for each material family
# ---------------------------------------------------------------------------

_MATERIAL_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "sand": {
        "E": (5e3, 5e4),       # Young's modulus
        "nu": (0.15, 0.35),    # Poisson's ratio
        "rho": (1200.0, 2000.0),  # Density
    },
    "snow": {
        "E": (1e3, 1e5),
        "nu": (0.1, 0.3),
        "rho": (100.0, 500.0),
    },
    "elastoplastic": {
        "E": (1e4, 1e6),
        "nu": (0.2, 0.45),
        "rho": (800.0, 2500.0),
    },
}

# Which families to include in domain randomization
_DEFAULT_FAMILIES: List[str] = ["sand", "snow", "elastoplastic"]


class DomainRandWrapper(gymnasium.Wrapper):
    """Gymnasium wrapper that randomises material on each reset.

    Wraps a GenesisGymWrapper and, on each ``reset()``, selects a
    random material family and samples continuous parameters from the
    training ranges.

    Parameters
    ----------
    env:
        The base GenesisGymWrapper to wrap.
    families:
        List of material family names to sample from.
        Default: ``["sand", "snow", "elastoplastic"]``.
    task_config:
        Task-level config passed to the env on rebuild.
    scene_config_base:
        Base scene config (material keys will be overridden).
    rebuild_on_reset:
        If True, rebuild the Genesis scene on each reset (required for
        material changes).  If False, only log which material *would*
        have been selected (useful for fast iteration / debugging).
    seed:
        Base seed for the parameter RNG.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        families: Optional[List[str]] = None,
        task_config: Optional[Dict[str, Any]] = None,
        scene_config_base: Optional[Dict[str, Any]] = None,
        rebuild_on_reset: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__(env)
        self.families = families or _DEFAULT_FAMILIES
        self.task_config = task_config or {}
        self.scene_config_base = scene_config_base or {}
        self.rebuild_on_reset = rebuild_on_reset
        self._rng = np.random.default_rng(seed)
        self._current_material: Optional[str] = None
        self._current_params: Dict[str, float] = {}
        self._reset_count = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset with a randomly sampled material.

        Parameters
        ----------
        seed:
            If provided, re-seeds the parameter RNG.
        options:
            Optional reset options.

        Returns
        -------
        tuple[ndarray, dict]
            ``(observation, info)`` with info containing
            ``"material_family"`` and ``"material_params"``.
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample material
        family = self._rng.choice(self.families)
        params = self._sample_params(family)
        self._current_material = family
        self._current_params = params
        self._reset_count += 1

        if self.rebuild_on_reset:
            # Rebuild the environment with the new material
            self._rebuild_env(family, params)

        obs, info = self.env.reset(seed=seed, options=options)

        # Inject domain-rand metadata into info
        info["material_family"] = family
        info["material_params"] = params
        info["domain_rand_reset_count"] = self._reset_count

        return obs, info

    def _sample_params(self, family: str) -> Dict[str, float]:
        """Sample continuous parameters for *family* from training ranges."""
        ranges = _MATERIAL_RANGES.get(family, {})
        params = {}
        for key, (lo, hi) in ranges.items():
            params[key] = float(self._rng.uniform(lo, hi))
        return params

    def _rebuild_env(self, family: str, params: Dict[str, float]) -> None:
        """Rebuild the Genesis env with new material parameters.

        This is the expensive path -- creates a new ScoopTransferTask
        and replaces the inner env's task.
        """
        from pta.envs.wrappers.gym_wrapper import GenesisGymWrapper

        # Build new scene config
        scene_config = {
            **self.scene_config_base,
            "particle_material": family,
            "particle_params": params,
        }

        # Close old env
        try:
            self.env.close()
        except Exception:
            pass

        # Create new env
        new_env = GenesisGymWrapper(
            task_config=self.task_config,
            scene_config=scene_config,
        )

        # Replace the wrapped env
        self.env = new_env

    @property
    def current_material(self) -> Optional[str]:
        """Return the currently active material family name."""
        return self._current_material

    @property
    def current_params(self) -> Dict[str, float]:
        """Return the currently active material parameters."""
        return self._current_params


def wrap_with_domain_rand(
    env: gymnasium.Env,
    families: Optional[List[str]] = None,
    rebuild: bool = False,
    seed: int = 42,
    **kwargs: Any,
) -> DomainRandWrapper:
    """Convenience function to wrap an env with domain randomisation.

    Parameters
    ----------
    env:
        Base GenesisGymWrapper.
    families:
        Material families to randomise over.
    rebuild:
        Whether to rebuild the scene on each reset.
    seed:
        RNG seed.
    **kwargs:
        Extra kwargs forwarded to DomainRandWrapper.

    Returns
    -------
    DomainRandWrapper
        The wrapped environment.
    """
    return DomainRandWrapper(
        env=env,
        families=families,
        rebuild_on_reset=rebuild,
        seed=seed,
        **kwargs,
    )
