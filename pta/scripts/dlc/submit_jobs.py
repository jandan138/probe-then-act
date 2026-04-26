#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


SUPPORTED_VARIANTS = {"no_probe", "no_belief"}
DEFAULT_SEEDS = [42, 0, 1]
DEFAULT_VARIANTS = ["no_probe", "no_belief"]
DEFAULT_DATA_SOURCES = "d-mzps5b7joy2axmqpa8,d-d49o5g0h2818sw8j1g,d-8wz4emfs21s5ajs9oz"


@dataclass(frozen=True)
class JobSpec:
    suite: str
    job_name: str
    chunk_id: int
    chunk_total: int
    command_args: str
    gpu_count: int
    data_sources: str | None = None
    variant: str | None = None
    seed: int | None = None


def _validate_gpu_count(gpu_count: int) -> None:
    if gpu_count not in {1, 2, 4, 8}:
        raise ValueError("gpu_count must be one of 1, 2, 4, 8")


def build_job_specs(
    *,
    suite: str,
    name: str,
    variants: list[str],
    seeds: list[int],
    gpu_count: int,
    data_sources: str | None,
    skips: list[str],
) -> list[JobSpec]:
    _validate_gpu_count(gpu_count)
    if suite == "ablation":
        chosen_variants = variants or DEFAULT_VARIANTS
        chosen_seeds = seeds or DEFAULT_SEEDS
        skip_pairs = _parse_skip_pairs(skips)
        jobs: list[JobSpec] = []
        candidate_count = 0
        for variant in chosen_variants:
            if variant not in SUPPORTED_VARIANTS:
                raise ValueError(f"unsupported variant: {variant}")
            for seed in chosen_seeds:
                if not isinstance(seed, int):
                    raise ValueError(f"seed must be int: {seed!r}")
                candidate_count += 1
                if (variant, seed) in skip_pairs:
                    continue
                jobs.append(
                    JobSpec(
                        suite=suite,
                        job_name=f"{name}_{variant}_s{seed}",
                        chunk_id=len(jobs),
                        chunk_total=0,
                        command_args=f"train_ablation {variant} {seed}",
                        gpu_count=gpu_count,
                        data_sources=data_sources,
                        variant=variant,
                        seed=seed,
                    )
                )
        if candidate_count and not jobs:
            raise ValueError("all ablation jobs were skipped")
        total = len(jobs)
        return [
            JobSpec(
                suite=job.suite,
                job_name=job.job_name,
                chunk_id=job.chunk_id,
                chunk_total=total,
                command_args=job.command_args,
                gpu_count=job.gpu_count,
                data_sources=job.data_sources,
                variant=job.variant,
                seed=job.seed,
            )
            for job in jobs
        ]
    if suite == "ood-ablation":
        return [
            JobSpec(
                suite=suite,
                job_name=name,
                chunk_id=0,
                chunk_total=1,
                command_args=(
                    "eval_ood --residual-scale 0.05 "
                    "--methods m7_noprobe m7_nobelief"
                ),
                gpu_count=gpu_count,
                data_sources=data_sources,
            )
        ]
    if suite == "smoke":
        return [
            JobSpec(
                suite=suite,
                job_name=name,
                chunk_id=0,
                chunk_total=1,
                command_args="smoke_env",
                gpu_count=gpu_count,
                data_sources=data_sources,
            )
        ]
    raise ValueError(f"unsupported suite: {suite}")


def _parse_skip_pairs(skips: list[str]) -> set[tuple[str, int]]:
    pairs: set[tuple[str, int]] = set()
    for skip in skips:
        try:
            variant, seed_text = skip.split(":", 1)
        except ValueError as exc:
            raise ValueError(f"skip must use variant:seed format: {skip}") from exc
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"unsupported skip variant: {variant}")
        try:
            seed = int(seed_text)
        except ValueError as exc:
            raise ValueError(f"skip seed must be int: {seed_text}") from exc
        pairs.add((variant, seed))
    return pairs


def append_manifest(
    path: Path,
    spec: JobSpec,
    *,
    dry_run: bool,
    returncode: int | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        **asdict(spec),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "returncode": returncode,
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def submit_specs(
    specs: list[JobSpec],
    *,
    repo_root: Path,
    launch_script: Path,
    dry_run: bool,
    manifest_path: Path,
) -> None:
    for spec in specs:
        data_sources = spec.data_sources or DEFAULT_DATA_SOURCES
        cmd = [
            "bash",
            str(launch_script),
            spec.job_name,
            str(spec.chunk_id),
            str(spec.chunk_total),
            data_sources,
            spec.command_args,
        ]
        env = os.environ.copy()
        env["DLC_GPU_COUNT"] = str(spec.gpu_count)
        print(shlex.join(cmd), flush=True)
        if dry_run:
            append_manifest(manifest_path, spec, dry_run=True, returncode=None)
            continue
        completed = subprocess.run(cmd, cwd=repo_root, env=env, check=True)
        append_manifest(
            manifest_path,
            spec,
            dry_run=False,
            returncode=completed.returncode,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit probe-then-act DLC jobs")
    parser.add_argument("--suite", choices=["ablation", "ood-ablation", "smoke"], required=True)
    parser.add_argument("--name", default=None)
    parser.add_argument("--variants", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="Skip one ablation job by variant:seed, e.g. --skip no_probe:42.",
    )
    parser.add_argument("--gpu-count", type=int, default=int(os.environ.get("DLC_GPU_COUNT", "1")))
    parser.add_argument("--data-sources", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = Path(__file__).resolve().parents[3]
    default_names = {
        "ablation": "pta_ablation",
        "ood-ablation": "pta_ood_ablation",
        "smoke": "pta_smoke",
    }
    specs = build_job_specs(
        suite=args.suite,
        name=args.name or default_names[args.suite],
        variants=args.variants or [],
        seeds=args.seeds or [],
        gpu_count=args.gpu_count,
        data_sources=args.data_sources,
        skips=args.skip,
    )
    submit_specs(
        specs,
        repo_root=repo_root,
        launch_script=repo_root / "pta" / "scripts" / "dlc" / "launch_job.sh",
        dry_run=args.dry_run,
        manifest_path=repo_root / "results" / "dlc" / "jobs.jsonl",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
