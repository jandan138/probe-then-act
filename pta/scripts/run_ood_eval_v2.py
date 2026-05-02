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
import hashlib
import math
import os
import random
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
        "ckpt_pattern": "checkpoints/m7_pta_noprobe_seed{seed}/m7_pta_final",
        "use_privileged": False,
        "use_m7_env": True,
        "ablation": "no_probe",
    },
    "m7_nobelief": {
        "seeds": [42, 0, 1],
        "ckpt_pattern": "checkpoints/m7_pta_nobelief_seed{seed}/m7_pta_final",
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


def parse_args(argv=None):
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
    parser.add_argument(
        "--m7-encoder-mode",
        choices=("matched", "random-stress"),
        default="matched",
        help="M7 encoder protocol: load matched sidecar or seed a fresh random encoder.",
    )
    parser.add_argument(
        "--m7-random-encoder-seed",
        type=int,
        default=None,
        help="Required when --m7-encoder-mode=random-stress.",
    )
    return parser.parse_args(argv)


def _load_genesis_gym_wrapper():
    from pta.envs.wrappers.gym_wrapper import GenesisGymWrapper

    return GenesisGymWrapper


def _load_joint_residual_wrapper():
    from pta.envs.wrappers.joint_residual_wrapper import JointResidualWrapper

    return JointResidualWrapper


def _load_privileged_obs_wrapper():
    from pta.envs.wrappers.privileged_obs_wrapper import PrivilegedObsWrapper

    return PrivilegedObsWrapper


def _load_probe_phase_wrapper():
    from pta.envs.wrappers.probe_phase_wrapper import ProbePhaseWrapper

    return ProbePhaseWrapper


def make_eval_env(
    split_config,
    use_privileged,
    use_m7_env,
    ablation="none",
    horizon=500,
    residual_scale=0.05,
    seed=0,
    belief_encoder=None,
):
    """Create evaluation environment for a given split."""
    GenesisGymWrapper = _load_genesis_gym_wrapper()
    JointResidualWrapper = _load_joint_residual_wrapper()

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
        ProbePhaseWrapper = _load_probe_phase_wrapper()

        env = ProbePhaseWrapper(
            env,
            latent_dim=16,
            n_probes=3,
            belief_encoder=belief_encoder,
            ablation=ablation,
            device="cpu",
        )
    elif use_privileged:
        PrivilegedObsWrapper = _load_privileged_obs_wrapper()
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
    "encoder_mode",
    "encoder_seed",
    "encoder_sha256",
    "policy_sha256",
    "protocol",
    "mean_reward",
    "std_reward",
    "mean_transfer",
    "std_transfer",
    "mean_spill",
    "std_spill",
    "success_rate",
    "n_failed_episodes",
]
RESULT_IDENTITY_FIELDS = [
    "encoder_mode",
    "encoder_seed",
    "encoder_sha256",
    "policy_sha256",
    "protocol",
]
LEGACY_IDENTITY = {
    "encoder_mode": "legacy-unversioned",
    "encoder_seed": "",
    "encoder_sha256": "",
    "policy_sha256": "",
    "protocol": "legacy_unversioned",
}
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
    identity = result_identity(row)
    return (
        row["method"],
        int(row["seed"]),
        row["split"],
        identity["encoder_mode"],
        str(identity["encoder_seed"]),
        identity["encoder_sha256"],
        identity["protocol"],
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def result_identity(row: dict) -> dict:
    if any(field not in row for field in RESULT_IDENTITY_FIELDS):
        return dict(LEGACY_IDENTITY)
    return {field: str(row.get(field, "")) for field in RESULT_IDENTITY_FIELDS}


def policy_only_identity(policy_path: Path) -> dict:
    return {
        "encoder_mode": "policy-only",
        "encoder_seed": "",
        "encoder_sha256": "",
        "policy_sha256": sha256_file(policy_path),
        "protocol": "policy_only",
    }


def legacy_identity_defaults() -> dict:
    return dict(LEGACY_IDENTITY)


def _new_m7_random_encoder(
    trace_dim: int = 30,
    latent_dim: int = 16,
    hidden_dim: int = 128,
    num_layers: int = 2,
):
    from pta.models.belief.latent_belief_encoder import LatentBeliefEncoder

    return LatentBeliefEncoder(
        trace_dim=trace_dim,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )


def resolve_m7_belief_encoder(
    policy_path: Path,
    ablation: str,
    encoder_mode: str,
    encoder_seed: int | None,
    expected: dict,
):
    policy_sha256 = sha256_file(policy_path)
    if ablation in ("no_probe", "no_belief"):
        return None, {
            "encoder_mode": "zero-z",
            "encoder_seed": "",
            "encoder_sha256": "",
            "policy_sha256": policy_sha256,
            "protocol": "ablation_zero_z",
        }
    if encoder_mode == "random-stress":
        if encoder_seed is None:
            raise ValueError("random-stress requires --m7-random-encoder-seed")
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        try:
            import torch
        except ImportError:
            torch = None
        try:
            if torch is not None and hasattr(torch, "random") and hasattr(
                torch.random, "fork_rng"
            ):
                with torch.random.fork_rng():
                    torch.manual_seed(encoder_seed)
                    encoder = _new_m7_random_encoder()
                    encoder.eval()
            else:
                torch_state = None
                if torch is not None and hasattr(torch, "random") and hasattr(
                    torch.random, "get_rng_state"
                ):
                    torch_state = torch.random.get_rng_state()
                if torch is not None:
                    torch.manual_seed(encoder_seed)
                encoder = _new_m7_random_encoder()
                encoder.eval()
                if torch_state is not None:
                    torch.random.set_rng_state(torch_state)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
        return encoder, {
            "encoder_mode": "random-stress",
            "encoder_seed": str(encoder_seed),
            "encoder_sha256": "",
            "policy_sha256": policy_sha256,
            "protocol": "random_stress",
        }
    from pta.training.utils.checkpoint_io import load_m7_encoder_artifact

    encoder, metadata = load_m7_encoder_artifact(policy_path, expected=expected)
    return encoder, {
        "encoder_mode": "matched",
        "encoder_seed": "",
        "encoder_sha256": metadata["belief_encoder_sha256"],
        "policy_sha256": metadata["paired_policy_sha256"],
        "protocol": metadata["protocol"],
    }


def resolve_checkpoint_path(project_root: Path, ckpt_pattern: str, seed: int) -> Path | None:
    ckpt_path = project_root / ckpt_pattern.format(seed=seed)
    if ckpt_path.exists():
        return ckpt_path
    zip_path = ckpt_path.with_suffix(".zip")
    if zip_path.exists():
        return zip_path
    return None


def append_result_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in RESULT_FIELDNAMES})


def coerce_nonnegative_int(value, field: str) -> int:
    number = float(value)
    if not math.isfinite(number) or not number.is_integer() or number < 0:
        raise ValueError(f"invalid {field}")
    return int(number)


def coerce_result_row(raw: dict) -> dict:
    if any(field not in raw for field in RESULT_IDENTITY_FIELDS):
        raw = {**legacy_identity_defaults(), **raw}
    required_fields = [
        field for field in RESULT_FIELDNAMES if field not in RESULT_IDENTITY_FIELDS
    ]
    if any(raw.get(field) in (None, "") for field in required_fields):
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
        "encoder_mode": raw["encoder_mode"],
        "encoder_seed": str(raw["encoder_seed"]),
        "encoder_sha256": raw["encoder_sha256"],
        "policy_sha256": raw["policy_sha256"],
        "protocol": raw["protocol"],
        **floats,
        "n_failed_episodes": failed_episodes,
    }


def write_result_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES, lineterminator="\n")
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
        identity = result_identity(row)
        key = (
            row["method"],
            row["split"],
            identity["encoder_mode"],
            str(identity["encoder_seed"]),
            identity["encoder_sha256"],
            identity["protocol"],
        )
        for metric in AGGREGATE_METRICS:
            agg[key][metric].append(row[metric])

    agg_rows = []
    for (
        method,
        split,
        encoder_mode,
        encoder_seed,
        encoder_sha256,
        protocol,
    ), metrics in sorted(agg.items()):
        agg_row = {
            "method": method,
            "split": split,
            "encoder_mode": encoder_mode,
            "encoder_seed": encoder_seed,
            "encoder_sha256": encoder_sha256,
            "protocol": protocol,
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
        writer = csv.DictWriter(
            f, fieldnames=list(agg_rows[0].keys()), lineterminator="\n"
        )
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
            ckpt_path = resolve_checkpoint_path(
                _PROJECT_ROOT,
                method_cfg["ckpt_pattern"],
                seed,
            )
            if ckpt_path is None:
                missing_path = _PROJECT_ROOT / method_cfg["ckpt_pattern"].format(seed=seed)
                print(
                    f"  SKIP: {method_name} seed={seed} — checkpoint not found: {missing_path}"
                )
                continue

            print(f"\n>>> {method_name} seed={seed}")
            belief_encoder = None
            if method_cfg.get("use_m7_env", False):
                belief_encoder, encoder_identity = resolve_m7_belief_encoder(
                    ckpt_path,
                    ablation=method_cfg.get("ablation", "none"),
                    encoder_mode=args.m7_encoder_mode,
                    encoder_seed=args.m7_random_encoder_seed,
                    expected={
                        "method": method_name,
                        "seed": seed,
                        "ablation": method_cfg.get("ablation", "none"),
                    },
                )
            else:
                encoder_identity = policy_only_identity(ckpt_path)
            model = PPO.load(str(ckpt_path))

            for split_name in splits_to_eval:
                split_cfg = SPLITS.get(split_name)
                if split_cfg is None:
                    continue

                key = result_key(
                    {
                        "method": method_name,
                        "seed": seed,
                        "split": split_name,
                        **encoder_identity,
                    }
                )
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
                    belief_encoder=belief_encoder,
                )

                metrics = evaluate_one(model, env, args.n_episodes)
                env.close()

                row = {
                    "method": method_name,
                    "seed": seed,
                    "split": split_name,
                    **encoder_identity,
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
