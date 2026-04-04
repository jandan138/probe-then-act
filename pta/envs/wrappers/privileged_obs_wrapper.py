"""Privileged observation wrapper for the teacher policy (M8).

Appends ground-truth material parameters to the base observation so
that the teacher can learn with full access to hidden physics.  The
student will never see these features -- they exist only to establish
an upper-bound on performance.

Privileged features (7-D, appended after base obs):
  - material_family_onehot (4):  one-hot over [sand, snow, elastoplastic, liquid]
  - E_norm              (1):  log-normalized Young's modulus
  - nu_norm             (1):  Poisson's ratio (already in [0, 0.5])
  - rho_norm            (1):  log-normalized density
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import gymnasium
import numpy as np

from pta.envs.materials.material_family import MaterialFamily


# ---------------------------------------------------------------------------
# Normalization constants (log-scale for E and rho)
# ---------------------------------------------------------------------------
# These cover the full training range across all families:
#   E   : [1e3, 1e6]   -> log10: [3, 6]
#   nu  : [0.1, 0.5]   -> used directly
#   rho : [100, 2500]   -> log10: [2, 3.4]

_LOG_E_MIN = 3.0    # log10(1e3)
_LOG_E_MAX = 6.0    # log10(1e6)
_NU_MIN = 0.0
_NU_MAX = 0.5
_LOG_RHO_MIN = 2.0  # log10(100)
_LOG_RHO_MAX = 3.5  # log10(~3162), covers up to 2500

# Family name -> MaterialFamily enum for one-hot encoding
_FAMILY_INDEX = {
    "sand": MaterialFamily.SAND,
    "snow": MaterialFamily.SNOW,
    "elastoplastic": MaterialFamily.ELASTOPLASTIC,
    "liquid": MaterialFamily.LIQUID,
}

# Number of material families for one-hot
_N_FAMILIES = len(MaterialFamily)

# Total privileged feature dimension: 4 (one-hot) + 3 (E, nu, rho)
PRIVILEGED_DIM = _N_FAMILIES + 3


class PrivilegedObsWrapper(gymnasium.Wrapper):
    """Gymnasium wrapper that appends privileged material params to obs.

    The privileged features are constant within an episode (the material
    does not change mid-episode) and are set from the ``scene_config``
    provided at construction time.

    Parameters
    ----------
    env:
        The base GenesisGymWrapper to wrap.
    scene_config:
        Scene configuration dict containing ``"particle_material"``
        (family name) and optionally ``"particle_params"`` with keys
        ``"E"``, ``"nu"``, ``"rho"``.  If continuous params are not
        given, defaults from :class:`MaterialParams` are used.
    """

    def __init__(
        self,
        env: gymnasium.Env,
        scene_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(env)

        scene_config = scene_config or {}
        self._base_obs_dim = env.observation_space.shape[0]

        # Extract material info
        family_name = scene_config.get("particle_material", "sand").lower()
        params = scene_config.get("particle_params", {})

        # Build the privileged feature vector (constant per episode)
        self._privileged_features = self._build_privileged_features(
            family_name, params,
        )

        # Expand observation space
        total_dim = self._base_obs_dim + PRIVILEGED_DIM
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(total_dim,),
            dtype=np.float32,
        )

    def _build_privileged_features(
        self,
        family_name: str,
        params: Dict[str, Any],
    ) -> np.ndarray:
        """Construct the normalized privileged feature vector.

        Returns
        -------
        ndarray of shape (PRIVILEGED_DIM,)
            [one_hot(4), E_norm(1), nu_norm(1), rho_norm(1)]
        """
        # --- One-hot family encoding ---
        family_idx = _FAMILY_INDEX.get(family_name, 0)
        one_hot = np.zeros(_N_FAMILIES, dtype=np.float32)
        one_hot[int(family_idx)] = 1.0

        # --- Continuous params with defaults per family ---
        defaults = _get_default_params(family_name)
        E = float(params.get("E", defaults["E"]))
        nu = float(params.get("nu", defaults["nu"]))
        rho = float(params.get("rho", defaults["rho"]))

        # --- Normalize to roughly [0, 1] ---
        E_norm = _normalize_log(E, _LOG_E_MIN, _LOG_E_MAX)
        nu_norm = _normalize_linear(nu, _NU_MIN, _NU_MAX)
        rho_norm = _normalize_log(rho, _LOG_RHO_MIN, _LOG_RHO_MAX)

        return np.concatenate([
            one_hot,
            np.array([E_norm, nu_norm, rho_norm], dtype=np.float32),
        ])

    def update_material(
        self,
        family_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update the privileged features for a new material.

        Useful when combined with domain randomization or curriculum
        wrappers that change the material between episodes.

        Parameters
        ----------
        family_name:
            New material family name.
        params:
            New continuous material parameters.
        """
        self._privileged_features = self._build_privileged_features(
            family_name, params or {},
        )

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        aug_obs = np.concatenate(
            [obs, self._privileged_features],
        ).astype(np.float32)
        info["privileged_features"] = self._privileged_features.copy()
        return aug_obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        aug_obs = np.concatenate(
            [obs, self._privileged_features],
        ).astype(np.float32)
        return aug_obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_log(value: float, log_min: float, log_max: float) -> float:
    """Normalize *value* to [0, 1] on a log10 scale, clamped."""
    log_val = np.log10(max(value, 1e-12))
    return float(np.clip((log_val - log_min) / (log_max - log_min), 0.0, 1.0))


def _normalize_linear(value: float, v_min: float, v_max: float) -> float:
    """Normalize *value* to [0, 1] linearly, clamped."""
    denom = v_max - v_min
    if denom < 1e-12:
        return 0.5
    return float(np.clip((value - v_min) / denom, 0.0, 1.0))


def _get_default_params(family_name: str) -> Dict[str, float]:
    """Return sensible default E/nu/rho for a material family."""
    defaults = {
        "sand": {"E": 2.0e4, "nu": 0.3, "rho": 1500.0},
        "snow": {"E": 1.4e4, "nu": 0.2, "rho": 400.0},
        "elastoplastic": {"E": 1.0e5, "nu": 0.35, "rho": 1200.0},
        "liquid": {"E": 1.0e3, "nu": 0.45, "rho": 1000.0},
    }
    return defaults.get(family_name, {"E": 1e4, "nu": 0.2, "rho": 1500.0})
