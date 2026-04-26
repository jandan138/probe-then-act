"""OOD evaluation for M1/M7/M8 + ablations on Config D.

Evaluates all trained methods on:
  - ID: Sand (training distribution)
  - OOD-Material: Snow, ElastoPlastic
  - OOD-Params: Sand with extreme E/nu/rho

Generates results/main_results.csv with mean ± std over seeds.

Usage::

    python pta/scripts/run_ood_eval_v2.py
    python pta/scripts/run_ood_eval_v2.py --n-episodes 5  # quick test
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
if "/usr/lib/wsl/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib:" + os.environ.get(
        "LD_LIBRARY_PATH", ""
    )


# ---- Method × Seed → Checkpoint mapping ----

METHODS = {
    "m1_reactive": {
        "seeds": [42, 0, 1],
        "ckpt_pattern": "checkpoints/m1_reactive_seed{seed}/best/best_model",
        "use_privileged": False,
        "use_m7_env": False,
    },
    "m8_teacher": {
        "seeds": [42, 0, 1],
        "ckpt_pattern": "checkpoints/m8_teacher_seed{seed}/best/best_model",
        "use_privileged": True,
        "use_m7_env": False,
    },
    "m7_pta": {
        "seeds": [42, 0, 1],
        "ckpt_pattern": "checkpoints/m7_pta_seed{seed}/best/best_model",
        "use_privileged": False,
        "use_m7_env": True,
        "ablation": "none",
    },
    "m7_noprobe": {
        "seeds": [42, 0, 1],
        "ckpt_pattern": "checkpoints/m7_pta_noprobe_seed{seed}/best/best_model",
        "use_privileged": False,
        "use_m7_env": True,
        "ablation": "no_probe",
    },
    "m7_nobelief": {
        "seeds": [42, 0, 1],
        "ckpt_pattern": "checkpoints/m7_pta_nobelief_seed{seed}/best/best_model",
        "use_privileged": False,
        "use_m7_env": True,
        "ablation": "no_belief",
    },
}

# ---- Material splits ----

SPLITS = {
    "id_sand": {
        "particle_material": "sand",
        "particle_params": {},
    },
    "ood_snow": {
        "particle_material": "snow",
        "particle_params": {},
    },
    "ood_elastoplastic": {
        "particle_material": "elastoplastic",
        "particle_params": {},
    },
    "ood_sand_soft": {
        "particle_material": "sand",
        "particle_params": {"E": 1e3},  # very soft sand
    },
    "ood_sand_hard": {
        "particle_material": "sand",
        "particle_params": {"E": 1e5},  # very hard sand
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="OOD Evaluation v2")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing OOD CSV progress and start from scratch.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods to eval (default: all)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Subset of splits to eval (default: all)",
    )
    return parser.parse_args()


def make_eval_env(
    split_config,
    use_privileged,
    use_m7_env,
    ablation="none",
    horizon=500,
    residual_scale=0.05,
    seed=0,
):
    """Create evaluation environment for a given split."""
    from pta.envs.wrappers.gym_wrapper import GenesisGymWrapper
    from pta.envs.wrappers.joint_residual_wrapper import JointResidualWrapper
    from pta.envs.wrappers.privileged_obs_wrapper import PrivilegedObsWrapper

    scene_config = {
        "tool_type": "scoop",
        "n_envs": 0,
        "particle_material": split_config["particle_material"],
    }
    if split_config.get("particle_params"):
        scene_config["particle_params"] = split_config["particle_params"]

    task_config = {
        "horizon": horizon,
        "success_threshold": 0.3,
    }

    base_env = GenesisGymWrapper(
        task_config=task_config,
        scene_config=scene_config,
    )

    env = JointResidualWrapper(
        base_env,
        residual_scale=residual_scale,
        trajectory="edge_push",
    )

    if use_m7_env:
        from pta.envs.wrappers.probe_phase_wrapper import ProbePhaseWrapper

        env = ProbePhaseWrapper(
            env,
            latent_dim=16,
            n_probes=3,
            ablation=ablation,
            device="cpu",
        )
    elif use_privileged:
        env = PrivilegedObsWrapper(env=env, scene_config=scene_config)

    env.reset(seed=seed)
    return env


AGGREGATE_METRICS = [
    "mean_reward",
    "mean_transfer",
    "mean_spill",
    "success_rate",
    "n_failed_episodes",
]

RESULT_FIELDNAMES = [
    "method",
    "seed",
    "split",
    "mean_reward",
    "std_reward",
    "mean_transfer",
    "std_transfer",
    "mean_spill",
    "std_spill",
    "success_rate",
    "n_failed_episodes",
]
RESULT_FLOAT_FIELDS = [
    "mean_reward",
    "std_reward",
    "mean_transfer",
    "std_transfer",
    "mean_spill",
    "std_spill",
    "success_rate",
]


def result_key(row):
    return (row["method"], int(row["seed"]), row["split"])


def append_result_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in RESULT_FIELDNAMES})


def coerce_nonnegative_int(value, field: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError(f"invalid {field}")
    return int(number)


def coerce_result_row(raw: dict) -> dict:
    if any(raw.get(field) in (None, "") for field in RESULT_FIELDNAMES):
        raise ValueError("missing result fields")
    floats = {}
    for field in RESULT_FLOAT_FIELDS:
        value = float(raw[field])
        if not math.isfinite(value):
            raise ValueError(f"non-finite {field}")
        floats[field] = value
    failed_episodes = coerce_nonnegative_int(
        raw["n_failed_episodes"], "n_failed_episodes"
    )
    return {
        "method": raw["method"],
        "seed": int(raw["seed"]),
        "split": raw["split"],
        **floats,
        "n_failed_episodes": failed_episodes,
    }


def write_result_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in RESULT_FIELDNAMES})
    tmp_path.replace(path)


def load_completed_rows(path: Path, resume: bool = True):
    if not resume or not path.exists():
        return [], set()

    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            try:
                rows.append(coerce_result_row(raw))
            except (KeyError, OverflowError, TypeError, ValueError):
                continue
    return rows, {result_key(row) for row in rows}


def prepare_result_files(per_seed_path: Path, main_results_path: Path, resume: bool) -> None:
    if resume:
        return
    for path in [per_seed_path, main_results_path]:
        if path.exists():
            path.unlink()


def _is_nan_simulator_failure(exc: Exception) -> bool:
    msg = str(exc).lower()
    exc_name = type(exc).__name__.lower()
    if re.search(r"(?<![a-z])nan(?![a-z])", msg) is None:
        return False
    return (
        "genesis" in exc_name
        or "invalid constraint forces" in msg
        or "simulator produced" in msg
    )


def evaluate_one(model, env, n_episodes, deterministic=True):
    """Run n_episodes and collect metrics."""
    rewards = []
    transfers = []
    spills = []
    successes = []
    failed_episodes = 0

    for ep in range(n_episodes):
        try:
            obs, info = env.reset()
            ep_reward = 0.0
            done = False
            last_info = {}

            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                done = terminated or truncated
                last_info = info

            rewards.append(ep_reward)
            transfers.append(last_info.get("transfer_efficiency", 0.0))
            spills.append(last_info.get("spill_ratio", 0.0))
            successes.append(1.0 if last_info.get("success_rate", 0.0) >= 0.5 else 0.0)
        except Exception as exc:
            if not _is_nan_simulator_failure(exc):
                raise
            failed_episodes += 1
            rewards.append(0.0)
            transfers.append(0.0)
            spills.append(1.0)
            successes.append(0.0)
            print(f"episode {ep} failed: {exc}", flush=True)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_transfer": float(np.mean(transfers)),
        "std_transfer": float(np.std(transfers)),
        "mean_spill": float(np.mean(spills)),
        "std_spill": float(np.std(spills)),
        "success_rate": float(np.mean(successes)),
        "n_failed_episodes": failed_episodes,
    }


def aggregate_results(all_rows):
    agg = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        key = (row["method"], row["split"])
        for metric in AGGREGATE_METRICS:
            agg[key][metric].append(row[metric])

    agg_rows = []
    for (method, split), metrics in sorted(agg.items()):
        agg_row = {
            "method": method,
            "split": split,
            "n_seeds": len(metrics["mean_reward"]),
        }
        for metric, values in metrics.items():
            agg_row[f"{metric}_mean"] = float(np.mean(values))
            agg_row[f"{metric}_std"] = float(np.std(values))
            if metric == "n_failed_episodes":
                agg_row[f"{metric}_sum"] = int(np.sum(values))
        agg_rows.append(agg_row)
    return agg_rows


def write_aggregate_results(path: Path, all_rows: list[dict]):
    agg_rows = aggregate_results(all_rows)
    if not agg_rows:
        return []
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)
    tmp_path.replace(path)
    return agg_rows


def main():
    args = parse_args()

    from stable_baselines3 import PPO

    methods_to_eval = args.methods or list(METHODS.keys())
    splits_to_eval = args.splits or list(SPLITS.keys())

    results_dir = _PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    per_seed_path = results_dir / "ood_eval_per_seed.csv"
    main_results_path = results_dir / "main_results.csv"
    prepare_result_files(per_seed_path, main_results_path, resume=args.resume)
    all_rows, completed_keys = load_completed_rows(per_seed_path, resume=args.resume)
    if args.resume and per_seed_path.exists():
        write_result_rows(per_seed_path, all_rows)

    print("=" * 70)
    print("OOD EVALUATION v2")
    print(f"  Methods: {methods_to_eval}")
    print(f"  Splits:  {splits_to_eval}")
    print(f"  Episodes: {args.n_episodes}")
    print(f"  Horizon:  {args.horizon}")
    print("=" * 70)

    for method_name in methods_to_eval:
        method_cfg = METHODS.get(method_name)
        if method_cfg is None:
            print(f"  SKIP: Unknown method '{method_name}'")
            continue

        for seed in method_cfg["seeds"]:
            ckpt_path = _PROJECT_ROOT / method_cfg["ckpt_pattern"].format(seed=seed)
            if not ckpt_path.exists() and not ckpt_path.with_suffix(".zip").exists():
                print(
                    f"  SKIP: {method_name} seed={seed} — checkpoint not found: {ckpt_path}"
                )
                continue

            ckpt_str = str(ckpt_path)
            if ckpt_path.with_suffix(".zip").exists() and not ckpt_path.exists():
                ckpt_str = str(ckpt_path.with_suffix(".zip"))

            print(f"\n>>> {method_name} seed={seed}")
            model = PPO.load(ckpt_str)

            for split_name in splits_to_eval:
                split_cfg = SPLITS.get(split_name)
                if split_cfg is None:
                    continue

                key = (method_name, seed, split_name)
                if key in completed_keys:
                    print(f"    {split_name}... SKIP existing", flush=True)
                    continue

                print(f"    {split_name}...", end=" ", flush=True)

                env = make_eval_env(
                    split_config=split_cfg,
                    use_privileged=method_cfg["use_privileged"],
                    use_m7_env=method_cfg.get("use_m7_env", False),
                    ablation=method_cfg.get("ablation", "none"),
                    horizon=args.horizon,
                    residual_scale=args.residual_scale,
                    seed=seed + 2000,
                )

                metrics = evaluate_one(model, env, args.n_episodes)
                env.close()

                row = {
                    "method": method_name,
                    "seed": seed,
                    "split": split_name,
                    **metrics,
                }
                all_rows.append(row)
                append_result_row(per_seed_path, row)
                completed_keys.add(key)
                write_aggregate_results(main_results_path, all_rows)

                print(
                    f"reward={metrics['mean_reward']:.2f} "
                    f"transfer={metrics['mean_transfer']:.3f} "
                    f"success={metrics['success_rate']:.2f} "
                    f"failed_ep={metrics['n_failed_episodes']}"
                )

    # ---- Aggregate: mean ± std over seeds ----
    if all_rows:
        write_result_rows(per_seed_path, all_rows)
    agg_rows = write_aggregate_results(main_results_path, all_rows)
    if all_rows:
        print(f"\nPer-seed results: {per_seed_path}")
    if agg_rows:
        print(f"Main results (mean±std): {main_results_path}")

    # ---- Print summary table ----
    print("\n" + "=" * 102)
    print("MAIN RESULTS TABLE (mean ± std over seeds)")
    print("=" * 102)
    header = (
        f"{'Method':<18} {'Split':<20} {'Transfer':>14} {'Spill':>12} "
        f"{'Success':>10} {'FailEp':>8}"
    )
    print(header)
    print("-" * 102)
    for row in agg_rows:
        t_mean = row["mean_transfer_mean"]
        t_std = row["mean_transfer_std"]
        s_mean = row["mean_spill_mean"]
        s_std = row["mean_spill_std"]
        suc = row["success_rate_mean"]
        failed = row["n_failed_episodes_sum"]
        print(
            f"{row['method']:<18} {row['split']:<20} "
            f"{t_mean:.3f}±{t_std:.3f}   "
            f"{s_mean:.3f}±{s_std:.3f}  "
            f"{suc:.2f} {failed:8d}"
        )
    print("=" * 102)


if __name__ == "__main__":
    main()
