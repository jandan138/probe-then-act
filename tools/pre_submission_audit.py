from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

METHODS = {
    "m1_reactive": {
        "ckpt_pattern": "checkpoints/m1_reactive_seed{seed}/best/best_model",
        "use_privileged": False,
        "use_m7_env": False,
        "ablation": "none",
    },
    "m7_pta": {
        "ckpt_pattern": "checkpoints/m7_pta_seed{seed}/best/best_model",
        "use_privileged": False,
        "use_m7_env": True,
        "ablation": "none",
    },
}

SPLIT_NAMES = [
    "id_sand",
    "ood_elastoplastic",
    "ood_sand_hard",
    "ood_sand_soft",
    "ood_snow",
]


def displacement_stats(before: np.ndarray, after: np.ndarray) -> dict[str, float]:
    """Return RMS, mean, and max per-particle displacement in metres."""
    before = np.asarray(before, dtype=float)
    after = np.asarray(after, dtype=float)
    if before.shape != after.shape:
        raise ValueError("before and after particle arrays must have the same shape")
    if before.ndim != 2 or before.shape[1] != 3:
        raise ValueError("particle arrays must have shape (n_particles, 3)")

    distances = np.linalg.norm(after - before, axis=1)
    return {
        "rms_m": float(np.sqrt(np.mean(distances * distances))),
        "mean_m": float(np.mean(distances)),
        "max_m": float(np.max(distances)),
    }


def persistent_fraction(
    probe_stats: dict[str, float],
    settle_stats: dict[str, float],
) -> float | None:
    """Return post-settle RMS displacement as a fraction of probe RMS displacement."""
    probe_rms = float(probe_stats["rms_m"])
    if probe_rms <= 1e-12:
        return None
    return float(settle_stats["rms_m"]) / probe_rms


def existing_keys(path: Path) -> set[tuple[str, int, str]]:
    """Return completed (method, seed, split) rows for resumable CSV writes."""
    path = Path(path)
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (row["method"], int(row["seed"]), row["split"])
            for row in csv.DictReader(handle)
        }


def append_eval_row(path: Path, row: dict) -> None:
    """Append one evaluator row using the same schema as run_ood_eval_v2."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDNAMES, lineterminator="\n")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row[field] for field in RESULT_FIELDNAMES})


def summarize_paired_elastoplastic(inputs: Iterable[Path]) -> dict:
    """Summarize paired M7-M1 transfer deltas on OOD elastoplastic."""
    by_key: dict[tuple[str, int, str], float] = {}
    for path in inputs:
        with Path(path).open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("method") in {"m1_reactive", "m7_pta"}:
                    by_key[(row["method"], int(row["seed"]), row["split"])] = float(
                        row["mean_transfer"]
                    )

    seeds = sorted({seed for _, seed, split in by_key if split == "ood_elastoplastic"})
    pairs = []
    for seed in seeds:
        m1 = by_key.get(("m1_reactive", seed, "ood_elastoplastic"))
        m7 = by_key.get(("m7_pta", seed, "ood_elastoplastic"))
        if m1 is None or m7 is None:
            continue
        pairs.append(
            {
                "seed": seed,
                "m1_transfer": m1,
                "m7_transfer": m7,
                "delta": m7 - m1,
            }
        )

    deltas = np.array([pair["delta"] for pair in pairs], dtype=float)
    if len(deltas) == 0:
        return {
            "pairs": [],
            "n_pairs": 0,
            "positive_pairs": 0,
            "mean_delta_pp": math.nan,
            "std_delta_pp": math.nan,
        }
    return {
        "pairs": pairs,
        "n_pairs": len(pairs),
        "positive_pairs": int(np.sum(deltas > 0)),
        "mean_delta_pp": float(np.mean(deltas) * 100.0),
        "std_delta_pp": float(np.std(deltas, ddof=1) * 100.0) if len(deltas) > 1 else 0.0,
    }


def encoder_sensitivity_gate(
    rows: list[dict],
    *,
    max_transfer_range_pp: float,
) -> dict:
    """Evaluate whether encoder sensitivity rows are safe enough for extra seeds."""
    transfers = [float(row["mean_transfer"]) for row in rows]
    transfer_range_pp = float((max(transfers) - min(transfers)) * 100.0) if transfers else math.nan
    total_failed_episodes = sum(int(row.get("n_failed_episodes", 0)) for row in rows)
    reasons = []
    if total_failed_episodes:
        reasons.append("encoder sensitivity eval had failed episodes")
    if not math.isfinite(transfer_range_pp) or transfer_range_pp > max_transfer_range_pp:
        reasons.append("encoder sensitivity transfer range exceeded threshold")
    return {
        "passes": not reasons,
        "transfer_range_pp": transfer_range_pp,
        "total_failed_episodes": total_failed_episodes,
        "reasons": reasons,
    }


def _load_eval_module():
    from pta.scripts import run_ood_eval_v2

    return run_ood_eval_v2


def _load_ppo():
    from stable_baselines3 import PPO

    return PPO


def _load_torch():
    import torch

    return torch


def _particle_positions(task) -> np.ndarray:
    positions = task.particles.get_particles_pos()
    if positions.dim() == 3:
        positions = positions[0]
    return positions.detach().cpu().numpy().copy()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(path)
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


def run_probe_integrity(args: argparse.Namespace) -> None:
    ev = _load_eval_module()
    env = ev.make_eval_env(
        split_config=ev.SPLITS[args.split],
        use_privileged=False,
        use_m7_env=False,
        ablation="none",
        horizon=args.horizon,
        residual_scale=args.residual_scale,
        seed=args.seed,
    )
    try:
        task = env._task
        obs, info = env.reset(seed=args.seed)
        before = _particle_positions(task)
        zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
        for _ in range(args.n_probes):
            obs, reward, terminated, truncated, info = env.step(zero_action)
        after_probe = _particle_positions(task)
        for _ in range(args.settle_steps):
            task.scene.step()
            task.post_physics_update()
        after_settle = _particle_positions(task)
    finally:
        env.close()

    probe = displacement_stats(before, after_probe)
    settle = displacement_stats(before, after_settle)
    payload = {
        "mode": "probe_integrity",
        "split": args.split,
        "seed": args.seed,
        "n_probes": args.n_probes,
        "settle_steps": args.settle_steps,
        "probe_displacement": probe,
        "after_settle_displacement": settle,
        "persistent_fraction_rms": persistent_fraction(probe, settle),
    }
    _write_json(
        Path(args.output_dir) / f"audit_probe_{args.split}_seed{args.seed}.json",
        payload,
    )


def run_encoder_sensitivity(args: argparse.Namespace) -> None:
    ev = _load_eval_module()
    PPO = _load_ppo()
    torch = _load_torch()
    method_cfg = METHODS[args.method]
    checkpoint = ev.resolve_checkpoint_path(PROJECT_ROOT, method_cfg["ckpt_pattern"], args.seed)
    if checkpoint is None:
        raise FileNotFoundError(method_cfg["ckpt_pattern"].format(seed=args.seed))
    model = PPO.load(str(checkpoint), device="auto")

    rows = []
    for encoder_seed in args.encoder_seeds:
        random.seed(encoder_seed)
        np.random.seed(encoder_seed)
        torch.manual_seed(encoder_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(encoder_seed)
        env = ev.make_eval_env(
            split_config=ev.SPLITS[args.split],
            use_privileged=method_cfg["use_privileged"],
            use_m7_env=method_cfg["use_m7_env"],
            ablation=method_cfg["ablation"],
            horizon=args.horizon,
            residual_scale=args.residual_scale,
            seed=args.seed + 2000,
        )
        try:
            metrics = ev.evaluate_one(model, env, args.n_episodes, deterministic=True)
        finally:
            env.close()
        rows.append({"encoder_seed": encoder_seed, **metrics})

    gate = encoder_sensitivity_gate(rows, max_transfer_range_pp=args.max_transfer_range_pp)
    payload = {
        "mode": "encoder_sensitivity",
        "method": args.method,
        "policy_seed": args.seed,
        "split": args.split,
        "n_episodes": args.n_episodes,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        "rows": rows,
        **gate,
    }
    _write_json(
        Path(args.output_dir) / f"audit_encoder_{args.method}_s{args.seed}_{args.split}.json",
        payload,
    )


def run_eval_extra_seeds(args: argparse.Namespace) -> None:
    ev = _load_eval_module()
    PPO = _load_ppo()
    output = Path(args.output)
    completed = existing_keys(output)
    for method in args.methods:
        method_cfg = METHODS[method]
        for seed in args.seeds:
            checkpoint = ev.resolve_checkpoint_path(PROJECT_ROOT, method_cfg["ckpt_pattern"], seed)
            if checkpoint is None:
                raise FileNotFoundError(method_cfg["ckpt_pattern"].format(seed=seed))
            model = PPO.load(str(checkpoint), device="auto")
            for split in args.splits:
                key = (method, seed, split)
                if key in completed:
                    print(f"SKIP existing {key}", flush=True)
                    continue
                env = ev.make_eval_env(
                    split_config=ev.SPLITS[split],
                    use_privileged=method_cfg["use_privileged"],
                    use_m7_env=method_cfg["use_m7_env"],
                    ablation=method_cfg["ablation"],
                    horizon=args.horizon,
                    residual_scale=args.residual_scale,
                    seed=seed + 2000,
                )
                try:
                    metrics = ev.evaluate_one(model, env, args.n_episodes, deterministic=True)
                finally:
                    env.close()
                row = {"method": method, "seed": seed, "split": split, **metrics}
                append_eval_row(output, row)
                completed.add(key)
                print(f"WROTE {key} {metrics}", flush=True)


def run_summarize_five_seed(args: argparse.Namespace) -> None:
    summary = summarize_paired_elastoplastic([Path(path) for path in args.inputs])
    print("seed,m1_transfer,m7_transfer,delta")
    for pair in summary["pairs"]:
        print(
            f"{pair['seed']},{pair['m1_transfer']:.6f},"
            f"{pair['m7_transfer']:.6f},{pair['delta']:.6f}"
        )
    print(f"n_pairs={summary['n_pairs']}")
    print(f"mean_delta_pp={summary['mean_delta_pp']:.3f}")
    print(f"std_delta_pp={summary['std_delta_pp']:.3f}")
    print(f"positive_pairs={summary['positive_pairs']}/{summary['n_pairs']}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-submission DLC audit helpers")
    parser.add_argument(
        "--mode",
        choices=[
            "probe-integrity",
            "encoder-sensitivity",
            "eval-extra-seeds",
            "summarize-five-seed",
        ],
        required=True,
    )
    parser.add_argument("--split", default="ood_elastoplastic", choices=SPLIT_NAMES)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["id_sand", "ood_elastoplastic", "ood_snow", "ood_sand_hard", "ood_sand_soft"],
        choices=SPLIT_NAMES,
    )
    parser.add_argument("--method", default="m7_pta", choices=sorted(METHODS))
    parser.add_argument("--methods", nargs="+", default=["m1_reactive", "m7_pta"], choices=sorted(METHODS))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--encoder-seeds", nargs="+", type=int, default=[11, 22, 33])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--max-transfer-range-pp", type=float, default=5.0)
    parser.add_argument("--n-probes", type=int, default=3)
    parser.add_argument("--settle-steps", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument("--output-dir", default="results/presub")
    parser.add_argument("--output", default="results/presub/ood_eval_extra_seeds.csv")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["results/ood_eval_per_seed.csv", "results/presub/ood_eval_extra_seeds.csv"],
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "probe-integrity":
        run_probe_integrity(args)
    elif args.mode == "encoder-sensitivity":
        run_encoder_sensitivity(args)
    elif args.mode == "eval-extra-seeds":
        run_eval_extra_seeds(args)
    elif args.mode == "summarize-five-seed":
        run_summarize_five_seed(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
