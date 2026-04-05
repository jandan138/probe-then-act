"""Evaluate a policy across out-of-distribution splits."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def evaluate_ood(
    model: Any,
    method: str,
    splits: List[Dict[str, Any]],
    n_episodes: int = 20,
    deterministic: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run a policy on multiple evaluation splits.

    For each split, creates a fresh environment with the split's material
    configuration and evaluates the policy.

    Parameters
    ----------
    model :
        Trained SB3 model.
    method :
        Method name (for wrapper selection).
    splits :
        List of split configurations from split_id/split_ood_material.
    n_episodes :
        Episodes per material config within each split.
    deterministic :
        Whether to use deterministic actions.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from split name to aggregated evaluation metrics.
    """
    from pta.eval.runners.eval_policy import evaluate_policy

    results = {}

    for split in splits:
        split_name = split["name"]
        materials = split.get("materials", [{}])
        print(f"\n{'='*50}")
        print(f"Evaluating split: {split_name}")
        print(f"{'='*50}")

        # Evaluate on each material config within the split
        all_metrics: Dict[str, List[float]] = {}

        for mat_cfg in materials:
            family = mat_cfg.get("family", "sand")
            params = mat_cfg.get("params", {})
            label = mat_cfg.get("label", family)

            print(f"  Material: {label} (family={family}, params={params})")

            # Create environment with this material
            env = _make_eval_env(method, family, params)
            if env is None:
                print(f"  SKIP: could not create env for {label}")
                continue

            try:
                metrics = evaluate_policy(
                    model=model,
                    env=env,
                    n_episodes=n_episodes,
                    deterministic=deterministic,
                )
                print(f"    success_rate={metrics['success_rate']:.3f}  "
                      f"return={metrics['mean_return']:.2f}  "
                      f"spill={metrics['mean_spill_ratio']:.3f}")

                # Accumulate for averaging across materials
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        all_metrics.setdefault(k, []).append(float(v))
            finally:
                try:
                    env.close()
                except Exception:
                    pass

        # Average metrics across materials in this split
        if all_metrics:
            import numpy as np
            results[split_name] = {
                k: float(np.mean(v)) for k, v in all_metrics.items()
            }
        else:
            results[split_name] = {}

    return results


def _make_eval_env(
    method: str,
    family: str,
    params: Dict[str, Any],
) -> Any:
    """Create a Gymnasium environment for evaluation.

    Applies the correct wrapper based on the method (e.g., FixedProbeWrapper
    for fixed_probe_ppo, PrivilegedObsWrapper for teacher).
    """
    from pta.envs.wrappers.gym_wrapper import GenesisGymWrapper

    scene_config = {
        "particle_material": family,
        "particle_params": params,
    }
    base_env = GenesisGymWrapper(scene_config=scene_config)

    if method == "fixed_probe_ppo":
        from pta.envs.wrappers.fixed_probe_wrapper import FixedProbeWrapper
        return FixedProbeWrapper(env=base_env)

    if method in ("teacher", "privileged_teacher"):
        from pta.envs.wrappers.privileged_obs_wrapper import PrivilegedObsWrapper
        return PrivilegedObsWrapper(env=base_env, scene_config=scene_config)

    return base_env
