#!/usr/bin/env python3
"""Build a deterministic manifest for experiment checkpoint handoff artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ARTIFACT_POLICY = "checkpoints are not stored in normal Git"


@dataclass(frozen=True)
class ArtifactCandidate:
    id: str
    path: str
    kind: str
    required_for: str
    required: bool = True


def _best_checkpoint(method: str, seed: int) -> str:
    return f"checkpoints/{method}_seed{seed}/best/best_model.zip"


ARTIFACTS: tuple[ArtifactCandidate, ...] = (
    *(
        ArtifactCandidate(
            id=f"m1_reactive_seed{seed}_best",
            path=_best_checkpoint("m1_reactive", seed),
            kind="corrected_ood",
            required_for="corrected_ood_replay",
        )
        for seed in (42, 0, 1)
    ),
    *(
        ArtifactCandidate(
            id=f"m7_pta_seed{seed}_best",
            path=_best_checkpoint("m7_pta", seed),
            kind="corrected_ood",
            required_for="corrected_ood_replay",
        )
        for seed in (42, 0, 1)
    ),
    ArtifactCandidate(
        id="m8_teacher_seed42_best",
        path=_best_checkpoint("m8_teacher", 42),
        kind="corrected_ood",
        required_for="corrected_ood_replay",
    ),
    ArtifactCandidate(
        id="m7_pta_noprobe_seed42_best",
        path=_best_checkpoint("m7_pta_noprobe", 42),
        kind="ablation_resume",
        required_for="resume_current_noprobe_seed42",
    ),
    ArtifactCandidate(
        id="m7_pta_noprobe_seed42_50000_steps",
        path="checkpoints/m7_pta_noprobe_seed42/m7_pta_50000_steps.zip",
        kind="ablation_resume",
        required_for="resume_current_noprobe_seed42",
    ),
    *(
        ArtifactCandidate(
            id=f"m7_pta_noprobe_seed{seed}_best",
            path=_best_checkpoint("m7_pta_noprobe", seed),
            kind="ablation_output",
            required_for="post_ablation_eval",
            required=False,
        )
        for seed in (0, 1)
    ),
    *(
        ArtifactCandidate(
            id=f"m7_pta_nobelief_seed{seed}_best",
            path=_best_checkpoint("m7_pta_nobelief", seed),
            kind="ablation_output",
            required_for="post_ablation_eval",
            required=False,
        )
        for seed in (42, 0, 1)
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_candidate(
    candidate: ArtifactCandidate,
    repo_root: Path,
    stage_d_root: Path | None,
) -> dict[str, object]:
    search_roots: list[tuple[str, Path]] = [("repo_root", repo_root)]
    if stage_d_root is not None:
        search_roots.append(("stage_d_root", stage_d_root))

    for root_name, root in search_roots:
        path = root / candidate.path
        if path.exists():
            return {
                "id": candidate.id,
                "path": candidate.path,
                "kind": candidate.kind,
                "required_for": candidate.required_for,
                "required": candidate.required,
                "exists": True,
                "source_root": root_name,
                "source_path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }

    return {
        "id": candidate.id,
        "path": candidate.path,
        "kind": candidate.kind,
        "required_for": candidate.required_for,
        "required": candidate.required,
        "exists": False,
        "source_root": None,
        "source_path": None,
        "size_bytes": 0,
        "sha256": None,
    }


def build_manifest(repo_root: Path, stage_d_root: Path | None) -> dict[str, object]:
    repo_root = repo_root.resolve()
    if stage_d_root is not None:
        stage_d_root = stage_d_root.resolve()

    return {
        "generated_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "artifact_policy": ARTIFACT_POLICY,
        "repo_root": str(repo_root),
        "stage_d_root": str(stage_d_root) if stage_d_root is not None else None,
        "artifacts": [
            resolve_candidate(candidate, repo_root, stage_d_root)
            for candidate in ARTIFACTS
        ],
    }


def write_manifest(manifest: dict[str, object], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_archive(manifest: dict[str, object], manifest_path: Path, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(manifest_path, arcname="checkpoint_manifest.json")
        for artifact in manifest["artifacts"]:
            if not isinstance(artifact, dict) or not artifact.get("exists"):
                continue
            source_path = artifact["source_path"]
            rel_path = artifact["path"]
            if not isinstance(source_path, str) or not isinstance(rel_path, str):
                continue
            archive.add(source_path, arcname=rel_path)


def missing_required(manifest: dict[str, object]) -> Iterable[str]:
    for artifact in manifest["artifacts"]:
        if (
            isinstance(artifact, dict)
            and artifact.get("required")
            and not artifact.get("exists")
        ):
            artifact_id = artifact.get("id")
            if isinstance(artifact_id, str):
                yield artifact_id


def parse_args() -> argparse.Namespace:
    repo_root_default = Path(__file__).resolve().parents[1]
    return argparse.ArgumentParser(description=__doc__).parse_args()


def main() -> int:
    repo_root_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=repo_root_default)
    parser.add_argument(
        "--stage-d-root",
        type=Path,
        default=repo_root_default / ".worktrees" / "aris-resume-stage-d",
        help="Optional worktree to search after --repo-root.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root_default / "results" / "dlc" / "checkpoint_manifest.json",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Optional .tar.gz path containing the manifest and present checkpoint files.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any required checkpoint is missing.",
    )
    args = parser.parse_args()

    stage_d_root = args.stage_d_root if args.stage_d_root.exists() else None
    manifest = build_manifest(args.repo_root, stage_d_root)
    write_manifest(manifest, args.manifest)
    if args.archive is not None:
        create_archive(manifest, args.manifest, args.archive)

    missing = list(missing_required(manifest))
    present_count = sum(
        1
        for artifact in manifest["artifacts"]
        if isinstance(artifact, dict) and artifact.get("exists")
    )
    print(
        f"wrote {args.manifest} with {present_count}/{len(manifest['artifacts'])} present artifacts"
    )
    if missing:
        print("missing required artifacts: " + ", ".join(missing))
    return 1 if args.strict and missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
