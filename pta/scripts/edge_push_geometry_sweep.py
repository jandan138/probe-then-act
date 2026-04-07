"""Edge-push geometry sweep: test multiple platform/target/particle configs.

Tests configs B-E (A = baseline already measured) across sand, snow, and
elastoplastic materials to find a geometry that:
  1. Gets at least one material > 30% transfer efficiency
  2. Produces a max-min material gap > 15%

Usage::

    source /home/zhuzihou/dev/Genesis/.venv/bin/activate
    export PYOPENGL_PLATFORM=osmesa
    export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
    export PYTHONPATH=/home/zhuzihou/dev/probe-then-act:$PYTHONPATH
    python pta/scripts/edge_push_geometry_sweep.py
"""

from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Geometry configurations (each changes ONE thing from default A)
# ---------------------------------------------------------------------------
# Default A (baseline):
#   platform_pos=(0.55, 0.0, 0.15), edge_target_pos=(0.55, 0.15, 0.01),
#   particle_pos=(0.55, -0.03, 0.20)

CONFIGS: Dict[str, Dict[str, Any]] = {
    "B_target_closer": {
        "edge_target_pos": (0.55, 0.10, 0.01),
    },
    "C_lower_platform": {
        "platform_pos": (0.55, 0.0, 0.10),
    },
    "D_particles_closer": {
        "particle_pos": (0.55, 0.02, 0.20),
    },
}

MATERIALS = ["sand", "snow", "elastoplastic"]
EPISODES_PER = 3

# Scoop waypoints (7-DOF, copied from run_scripted_baseline.py)
HOME_S = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
EXTEND_FWD_S = [0.0, 0.5, 0.0, -1.8, 0.0, 1.8, 0.0]
BEHIND_EP = [-0.10, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]
PUSH_END_EP = [0.40, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0]

# Lower-platform variants: J2=0.6 -> z~0.11, right at platform surface
BEHIND_EP_LOW = [-0.10, 0.6, 0.0, -1.5, 0.0, 1.5, 0.0]
PUSH_END_EP_LOW = [0.40, 0.6, 0.0, -1.5, 0.0, 1.5, 0.0]


def interpolate_waypoints(
    env: Any, start: List[float], end: List[float],
    n_steps: int, settle_per_step: int = 1,
) -> None:
    s = torch.tensor(start, dtype=torch.float32, device="cuda")
    e = torch.tensor(end, dtype=torch.float32, device="cuda")
    for i in range(n_steps):
        alpha = (i + 1) / n_steps
        qpos = s * (1 - alpha) + e * alpha
        env.robot.set_qpos(qpos)
        for _ in range(settle_per_step):
            env.scene.step()


def settle(env: Any, n_steps: int = 80) -> None:
    for _ in range(n_steps):
        env.scene.step()


def run_edge_push(env: Any, low_platform: bool = False) -> Dict[str, Any]:
    """Run Sequence E edge-push and return metrics."""
    env.reset()

    behind = BEHIND_EP_LOW if low_platform else BEHIND_EP
    push_end = PUSH_END_EP_LOW if low_platform else PUSH_END_EP

    # Approach
    interpolate_waypoints(env, HOME_S, EXTEND_FWD_S, 20, settle_per_step=1)
    interpolate_waypoints(env, EXTEND_FWD_S, behind, 30, settle_per_step=2)

    # Multi-pass push (3 passes)
    for pass_idx in range(3):
        interpolate_waypoints(env, behind, push_end, 100, settle_per_step=3)
        if pass_idx < 2:
            interpolate_waypoints(env, push_end, behind, 30, settle_per_step=1)

    # Settle
    settle(env, 80)

    return env.compute_metrics()


def build_env(material: str, overrides: Dict[str, Any]) -> Any:
    """Build a ScoopTransferTask with given material and geometry overrides."""
    from pta.envs.tasks.scoop_transfer import ScoopTransferTask

    scene_cfg = {
        "particle_material": material,
        "n_envs": 0,
        "tool_type": "scoop",
    }
    scene_cfg.update(overrides)

    task_cfg = {"horizon": 500}
    return ScoopTransferTask(config=task_cfg, scene_config=scene_cfg)


def main() -> None:
    output_path = "results/edge_push_material_sweep/geometry_sweep.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_rows: List[Dict[str, Any]] = []

    # Add baseline A results (from prior runs) for reference in summary
    baseline_a = {
        "snow": 0.223,
        "liquid": 0.138,
        "sand": 0.126,
        "elastoplastic": 0.0,
    }

    print("=" * 70)
    print("Edge-Push Geometry Sweep")
    print("=" * 70)
    print(f"  Configs: {list(CONFIGS.keys())}")
    print(f"  Materials: {MATERIALS}")
    print(f"  Episodes per combo: {EPISODES_PER}")
    print()

    # Track best two individual configs for combined Config E
    config_improvements: Dict[str, float] = {}

    for config_name, overrides in CONFIGS.items():
        print(f"\n{'='*60}")
        print(f"Config: {config_name}")
        print(f"  Overrides: {overrides}")
        print(f"{'='*60}")

        low_platform = "lower_platform" in config_name

        for material in MATERIALS:
            print(f"\n  Material: {material}")

            # Each material needs a fresh env (Genesis scene rebuild)
            import genesis as gs
            if gs._initialized:
                # Force re-init by building new scene
                pass

            env = build_env(material, overrides)
            print(f"    Built env: {env._total_particles} particles")

            for ep in range(EPISODES_PER):
                t0 = time.time()
                metrics = run_edge_push(env, low_platform=low_platform)
                dt = time.time() - t0

                te = metrics.get("transfer_efficiency", 0.0)
                sr = metrics.get("spill_ratio", 0.0)
                n_in = metrics.get("n_in_target", 0)
                n_sp = metrics.get("n_spilled", 0)
                total = metrics.get("total_particles", 0)

                print(
                    f"    ep {ep+1}/{EPISODES_PER}  "
                    f"transfer={te:.4f}  spill={sr:.4f}  "
                    f"in_target={n_in}  spilled={n_sp}  "
                    f"total={total}  ({dt:.1f}s)"
                )

                all_rows.append({
                    "config": config_name,
                    "material": material,
                    "episode": ep + 1,
                    "transfer_efficiency": te,
                    "spill_ratio": sr,
                    "n_in_target": n_in,
                    "n_spilled": n_sp,
                    "total_particles": total,
                })

            # Cleanup env to free GPU memory
            del env

        # Compute mean improvement for this config (across materials)
        config_te = {}
        for material in MATERIALS:
            rows = [r for r in all_rows
                    if r["config"] == config_name and r["material"] == material]
            config_te[material] = np.mean([r["transfer_efficiency"] for r in rows])

        mean_te = np.mean(list(config_te.values()))
        baseline_mean = np.mean([baseline_a.get(m, 0.0) for m in MATERIALS])
        improvement = mean_te - baseline_mean
        config_improvements[config_name] = improvement
        print(f"\n  >> Config {config_name} mean TE: {mean_te:.4f} (delta vs A: {improvement:+.4f})")

    # ---------------------------------------------------------------------------
    # Config E: combine the two best individual improvements
    # ---------------------------------------------------------------------------
    sorted_configs = sorted(config_improvements.items(), key=lambda x: -x[1])
    best_two = [c[0] for c in sorted_configs[:2]]
    print(f"\n{'='*60}")
    print(f"Config E: Combined best two: {best_two}")

    combined_overrides: Dict[str, Any] = {}
    for cfg_name in best_two:
        combined_overrides.update(CONFIGS[cfg_name])
    print(f"  Combined overrides: {combined_overrides}")
    print(f"{'='*60}")

    low_platform = any("lower_platform" in c for c in best_two)

    for material in MATERIALS:
        print(f"\n  Material: {material}")
        env = build_env(material, combined_overrides)
        print(f"    Built env: {env._total_particles} particles")

        for ep in range(EPISODES_PER):
            t0 = time.time()
            metrics = run_edge_push(env, low_platform=low_platform)
            dt = time.time() - t0

            te = metrics.get("transfer_efficiency", 0.0)
            sr = metrics.get("spill_ratio", 0.0)
            n_in = metrics.get("n_in_target", 0)
            n_sp = metrics.get("n_spilled", 0)
            total = metrics.get("total_particles", 0)

            print(
                f"    ep {ep+1}/{EPISODES_PER}  "
                f"transfer={te:.4f}  spill={sr:.4f}  "
                f"in_target={n_in}  spilled={n_sp}  "
                f"total={total}  ({dt:.1f}s)"
            )

            all_rows.append({
                "config": "E_combined",
                "material": material,
                "episode": ep + 1,
                "transfer_efficiency": te,
                "spill_ratio": sr,
                "n_in_target": n_in,
                "n_spilled": n_sp,
                "total_particles": total,
            })

        del env

    # ---------------------------------------------------------------------------
    # Save CSV
    # ---------------------------------------------------------------------------
    print(f"\nSaving {len(all_rows)} rows to {output_path}")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["config", "material", "episode",
                        "transfer_efficiency", "spill_ratio",
                        "n_in_target", "n_spilled", "total_particles"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    # ---------------------------------------------------------------------------
    # Summary table
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Config':<22s} {'Material':<16s} {'Mean TE':>10s} {'Std TE':>10s} {'Mean Spill':>12s}")
    print("-" * 80)

    # Include baseline A for comparison
    for mat in MATERIALS:
        bte = baseline_a.get(mat, 0.0)
        print(f"{'A_baseline':<22s} {mat:<16s} {bte:>10.4f} {'(prior)':>10s} {'(prior)':>12s}")

    config_material_te: Dict[str, Dict[str, float]] = {}
    for config_name in list(CONFIGS.keys()) + ["E_combined"]:
        config_material_te[config_name] = {}
        for mat in MATERIALS:
            rows = [r for r in all_rows
                    if r["config"] == config_name and r["material"] == mat]
            if not rows:
                continue
            tes = [r["transfer_efficiency"] for r in rows]
            sps = [r["spill_ratio"] for r in rows]
            mean_te = np.mean(tes)
            std_te = np.std(tes)
            mean_sp = np.mean(sps)
            config_material_te[config_name][mat] = mean_te
            print(f"{config_name:<22s} {mat:<16s} {mean_te:>10.4f} {std_te:>10.4f} {mean_sp:>12.4f}")

    # Material discriminability summary
    print("\n" + "-" * 80)
    print("DISCRIMINABILITY ANALYSIS")
    print("-" * 80)
    print(f"{'Config':<22s} {'Max TE':>10s} {'Min TE':>10s} {'Gap':>10s} {'Max>30%?':>10s} {'Gap>15%?':>10s}")
    print("-" * 80)

    # Baseline A
    a_vals = [baseline_a.get(m, 0.0) for m in MATERIALS]
    a_max, a_min = max(a_vals), min(a_vals)
    a_gap = a_max - a_min
    print(f"{'A_baseline':<22s} {a_max:>10.4f} {a_min:>10.4f} {a_gap:>10.4f} "
          f"{'NO' if a_max < 0.3 else 'YES':>10s} {'NO' if a_gap < 0.15 else 'YES':>10s}")

    for config_name in list(CONFIGS.keys()) + ["E_combined"]:
        if config_name not in config_material_te:
            continue
        vals = [config_material_te[config_name].get(m, 0.0) for m in MATERIALS]
        mx, mn = max(vals), min(vals)
        gap = mx - mn
        print(f"{config_name:<22s} {mx:>10.4f} {mn:>10.4f} {gap:>10.4f} "
              f"{'NO' if mx < 0.3 else 'YES':>10s} {'NO' if gap < 0.15 else 'YES':>10s}")

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
