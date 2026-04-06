"""CLI entry point for main OOD evaluation.

Evaluates trained baselines (M1-M4, M8) across ID and OOD-Material splits.

Usage::

    # Evaluate all methods on ID + OOD-Material
    python pta/scripts/run_eval_main.py

    # Evaluate specific method
    python pta/scripts/run_eval_main.py --methods reactive_ppo rnn_ppo

    # Fewer episodes for quick check
    python pta/scripts/run_eval_main.py --n-episodes 5
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# Method -> (checkpoint_path, model_class_hint)
_METHOD_CHECKPOINTS = {
    "reactive_ppo": "checkpoints/reactive_ppo_v2/seed42/final_model",
    "rnn_ppo": "checkpoints/rnn_ppo/best/best_model",
    "domain_rand_ppo": "checkpoints/domain_rand_ppo/best/best_model",
    "fixed_probe_ppo": "checkpoints/fixed_probe_ppo/best/best_model",
    "teacher": "checkpoints/teacher/best/best_model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained policies on ID + OOD splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--methods", nargs="+",
        default=list(_METHOD_CHECKPOINTS.keys()),
        help="Methods to evaluate",
    )
    parser.add_argument(
        "--splits", nargs="+",
        default=["id", "ood_material"],
        help="Evaluation splits",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=20,
        help="Episodes per material config",
    )
    parser.add_argument(
        "--deterministic", action="store_true", default=True,
        help="Use deterministic actions",
    )
    parser.add_argument(
        "--horizon", type=int, default=200,
        help="Episode horizon (must match training horizon)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Override checkpoint directory (defaults to project checkpoints/)",
    )
    return parser.parse_args()


def _get_split_configs(split_names):
    """Get split configurations by name."""
    from pta.eval.splits.split_id import get_id_split
    from pta.eval.splits.split_ood_material import get_ood_material_split

    _SPLIT_MAP = {
        "id": get_id_split,
        "ood_material": get_ood_material_split,
    }

    splits = []
    for name in split_names:
        if name in _SPLIT_MAP:
            splits.append(_SPLIT_MAP[name]())
        else:
            print(f"WARNING: Unknown split '{name}', skipping")
    return splits


def main() -> None:
    """Parse CLI arguments and run the main evaluation suite."""
    args = parse_args()

    # Set WSL2 rendering env vars
    os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")
    if "/usr/lib/wsl/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = (
            "/usr/lib/wsl/lib:" + os.environ.get("LD_LIBRARY_PATH", "")
        )

    from pta.eval.runners.eval_ood import evaluate_ood
    from pta.eval.runners.eval_policy import load_sb3_model

    ckpt_base = Path(args.checkpoint_dir) if args.checkpoint_dir else _PROJECT_ROOT
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = _get_split_configs(args.splits)
    if not splits:
        print("ERROR: No valid splits specified")
        sys.exit(1)

    all_results = {}
    task_config = {"horizon": args.horizon}
    print("=" * 60)
    print(f"Phase 3: OOD Evaluation")
    print(f"  Methods:  {args.methods}")
    print(f"  Splits:   {args.splits}")
    print(f"  Episodes: {args.n_episodes} per material config")
    print(f"  Horizon:  {args.horizon}")
    print("=" * 60)

    for method in args.methods:
        print(f"\n{'#'*60}")
        print(f"# Method: {method}")
        print(f"{'#'*60}")

        # Find checkpoint
        ckpt_rel = _METHOD_CHECKPOINTS.get(method)
        if ckpt_rel is None:
            print(f"  SKIP: No checkpoint mapping for '{method}'")
            continue

        ckpt_path = ckpt_base / ckpt_rel
        # SB3 saves as .zip; try with and without extension
        if not ckpt_path.exists() and not ckpt_path.with_suffix(".zip").exists():
            print(f"  SKIP: Checkpoint not found at {ckpt_path}")
            continue

        # Load model
        print(f"  Loading checkpoint: {ckpt_path}")
        model = load_sb3_model(str(ckpt_path), method)
        print(f"  Model loaded successfully")

        # Evaluate across splits
        method_results = evaluate_ood(
            model=model,
            method=method,
            splits=splits,
            n_episodes=args.n_episodes,
            deterministic=args.deterministic,
            task_config=task_config,
        )
        all_results[method] = method_results

    # Save results
    results_path = output_dir / "eval_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Generate summary table
    _print_summary_table(all_results)

    # Save CSV
    csv_path = output_dir / "eval_results.csv"
    _save_csv(all_results, csv_path)
    print(f"CSV saved to {csv_path}")


def _print_summary_table(results: dict) -> None:
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 80)

    methods = list(results.keys())
    if not methods:
        print("No results to display.")
        return

    splits = set()
    for m in methods:
        splits.update(results[m].keys())
    splits = sorted(splits)

    # Header
    header = f"{'Method':<20}"
    for s in splits:
        header += f" | {'Return':>8} {'Success':>8} {'Spill':>8}"
    print(header)
    print("-" * len(header))

    for method in methods:
        row = f"{method:<20}"
        for s in splits:
            metrics = results[method].get(s, {})
            ret = metrics.get("mean_return", float("nan"))
            suc = metrics.get("success_rate", float("nan"))
            spi = metrics.get("mean_spill_ratio", float("nan"))
            row += f" | {ret:>8.2f} {suc:>8.3f} {spi:>8.3f}"
        print(row)

    print("=" * 80)


def _save_csv(results: dict, path: Path) -> None:
    """Save results as CSV."""
    rows = []
    for method, splits in results.items():
        for split_name, metrics in splits.items():
            row = {"method": method, "split": split_name}
            row.update(metrics)
            rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
