#!/usr/bin/env python3
"""Scan Probe-Then-Act artifact requirements."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = Path("/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act")
SCHEMA_VERSION = 1
PROJECT_NAME = "probe-then-act"
SAFE_ENV_KEYS = (
    "PYTHONPATH",
    "PYOPENGL_PLATFORM",
    "EGL_DEVICE_ID",
    "CUDA_VISIBLE_DEVICES",
    "DLC_JOB_ID",
    "DLC_RUN_ID",
)
RESTORE_TOP_LEVEL_FILES = ("artifact_manifest.json", "command.txt", "env.json")
RESTORE_TOP_LEVEL_DIRS = ("checkpoints", "logs", "results")
RESTORE_TOP_LEVEL_NAMES = set(RESTORE_TOP_LEVEL_FILES) | set(RESTORE_TOP_LEVEL_DIRS)
SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ArtifactCandidate:
    logical_name: str
    relative_path: str
    kind: str = "policy_checkpoint"
    required_for: str = "manual"
    required: bool = True
    min_num_timesteps: int | None = None


def _best_checkpoint(method: str, seed: int) -> str:
    return f"checkpoints/{method}_seed{seed}/best/best_model.zip"


def _best_policy_metadata(method: str, seed: int) -> str:
    return f"checkpoints/{method}_seed{seed}/best/best_model.json"


def _best_belief_encoder(method: str, seed: int) -> str:
    return f"checkpoints/{method}_seed{seed}/best/belief_encoder.pt"


def _best_belief_encoder_metadata(method: str, seed: int) -> str:
    return f"checkpoints/{method}_seed{seed}/best/belief_encoder_metadata.json"


REQUIREMENTS: dict[str, tuple[ArtifactCandidate, ...]] = {
    "presub-g2": (
        ArtifactCandidate(
            "m7_pta_seed42_best",
            _best_checkpoint("m7_pta", 42),
            kind="policy_checkpoint",
            required_for="presub-g2",
            min_num_timesteps=50_000,
        ),
    ),
    "g2-matched-encoder": (
        ArtifactCandidate(
            "m7_pta_seed42_best",
            _best_checkpoint("m7_pta", 42),
            kind="policy_checkpoint",
            required_for="g2-matched-encoder",
            min_num_timesteps=50_000,
        ),
        ArtifactCandidate(
            "m7_pta_seed42_best_metadata",
            _best_policy_metadata("m7_pta", 42),
            kind="policy_metadata",
            required_for="g2-matched-encoder",
        ),
        ArtifactCandidate(
            "m7_pta_seed42_belief_encoder",
            _best_belief_encoder("m7_pta", 42),
            kind="belief_encoder",
            required_for="g2-matched-encoder",
        ),
        ArtifactCandidate(
            "m7_pta_seed42_belief_encoder_metadata",
            _best_belief_encoder_metadata("m7_pta", 42),
            kind="belief_encoder_metadata",
            required_for="g2-matched-encoder",
        ),
    ),
    "presub-extra-eval": tuple(
        ArtifactCandidate(
            f"{method}_seed{seed}_best",
            _best_checkpoint(method, seed),
            kind="policy_checkpoint",
            required_for="presub-extra-eval",
            min_num_timesteps=50_000,
        )
        for method in ("m1_reactive", "m7_pta")
        for seed in (2, 3)
    ),
    "corrected-ood-replay": (
        *tuple(
            ArtifactCandidate(
                f"m1_reactive_seed{seed}_best",
                _best_checkpoint("m1_reactive", seed),
                kind="policy_checkpoint",
                required_for="corrected-ood-replay",
                min_num_timesteps=50_000,
            )
            for seed in (42, 0, 1)
        ),
        *tuple(
            ArtifactCandidate(
                f"m7_pta_seed{seed}_best",
                _best_checkpoint("m7_pta", seed),
                kind="policy_checkpoint",
                required_for="corrected-ood-replay",
                min_num_timesteps=50_000,
            )
            for seed in (42, 0, 1)
        ),
        ArtifactCandidate(
            "m8_teacher_seed42_best",
            _best_checkpoint("m8_teacher", 42),
            kind="policy_checkpoint",
            required_for="corrected-ood-replay",
            min_num_timesteps=50_000,
        ),
    ),
    "ablation-replay": (
        *tuple(
            ArtifactCandidate(
                f"m7_pta_noprobe_seed{seed}_final",
                f"checkpoints/m7_pta_noprobe_seed{seed}/m7_pta_final.zip",
                kind="policy_checkpoint",
                required_for="ablation-replay",
                min_num_timesteps=500_000,
            )
            for seed in (42, 0, 1)
        ),
        *tuple(
            ArtifactCandidate(
                f"m7_pta_nobelief_seed{seed}_final",
                f"checkpoints/m7_pta_nobelief_seed{seed}/m7_pta_final.zip",
                kind="policy_checkpoint",
                required_for="ablation-replay",
                min_num_timesteps=500_000,
            )
            for seed in (42, 0, 1)
        ),
    ),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def run_date_from_id(run_id: str) -> str:
    if len(run_id) >= 8 and run_id[:8].isdigit():
        return run_id[:8]
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def validate_run_id(run_id: str) -> None:
    if not SAFE_RUN_ID_RE.fullmatch(run_id):
        raise ValueError(f"unsafe run_id: {run_id}")


def validate_registry_relative_path(relative_path: str, allowed_top_levels: set[str], label: str) -> str:
    path = PurePosixPath(relative_path)
    parts = path.parts
    if path.is_absolute() or not parts:
        raise ValueError(f"unsafe {label}: {relative_path}")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe {label}: {relative_path}")
    if parts[0] not in allowed_top_levels:
        raise ValueError(f"unsafe {label}: {relative_path}")
    return path.as_posix()


def path_is_within(child: Path, parent: Path) -> bool:
    return child == parent or parent in child.parents


def selected_env() -> dict[str, str]:
    return {key: os.environ[key] for key in SAFE_ENV_KEYS if key in os.environ}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def requirement_paths(name: str) -> list[str]:
    return [candidate.relative_path for candidate in REQUIREMENTS[name]]


def logical_name_from_path(relative_path: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", relative_path).strip("_")


def explicit_artifact_candidates(artifact_paths: Iterable[str]) -> list[ArtifactCandidate]:
    return [
        ArtifactCandidate(
            logical_name=logical_name_from_path(relative_path),
            relative_path=validate_registry_relative_path(
                relative_path, {"checkpoints", "results"}, "artifact path"
            ),
            required_for="explicit-artifact-path",
        )
        for relative_path in artifact_paths
    ]


def selected_candidates(
    requirements: Iterable[str],
    artifact_paths: Iterable[str] | None = None,
) -> list[ArtifactCandidate]:
    candidates: list[ArtifactCandidate] = []
    seen: set[str] = set()
    for requirement in requirements:
        for candidate in REQUIREMENTS[requirement]:
            if candidate.logical_name in seen:
                continue
            seen.add(candidate.logical_name)
            candidates.append(candidate)
    for candidate in explicit_artifact_candidates(artifact_paths or []):
        if candidate.logical_name in seen:
            continue
        seen.add(candidate.logical_name)
        candidates.append(candidate)
    if not candidates:
        raise ValueError("at least one --requirement or --artifact-path is required")
    return candidates


def artifact_status(candidate: ArtifactCandidate, repo_root: Path) -> dict[str, object]:
    repo_root = repo_root.resolve()
    source_path = repo_root / candidate.relative_path
    exists = source_path.exists()
    row = {
        "logical_name": candidate.logical_name,
        "relative_path": candidate.relative_path,
        "source_path": None,
        "storage_path": None,
        "size_bytes": 0,
        "sha256": None,
        "kind": candidate.kind,
        "required_for": candidate.required_for,
        "required": candidate.required,
        "min_num_timesteps": candidate.min_num_timesteps,
        "exists": exists,
        "num_timesteps": None,
        "load_status": "not_checked",
    }
    if not exists:
        return row
    if source_path.is_symlink():
        row["load_status"] = "failed"
        row["load_error"] = f"checkpoint source is symlink: {candidate.relative_path}"
        return row
    resolved_source = source_path.resolve()
    if not path_is_within(resolved_source, repo_root):
        row["load_status"] = "failed"
        row["load_error"] = f"checkpoint source escapes repo root: {candidate.relative_path}"
        return row
    row["source_path"] = str(resolved_source)
    row["size_bytes"] = source_path.stat().st_size
    row["sha256"] = sha256_file(source_path)
    return row


def git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def build_scan_manifest(
    repo_root: Path,
    requirements: list[str],
    artifact_paths: list[str] | None = None,
) -> dict[str, object]:
    repo_root = repo_root.resolve()
    artifacts = [
        artifact_status(candidate, repo_root)
        for candidate in selected_candidates(requirements, artifact_paths)
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "project": PROJECT_NAME,
        "created_at_utc": utc_now(),
        "git_commit": git_commit(repo_root),
        "repo_root_at_registration": str(repo_root),
        "artifact_root": str(DEFAULT_ARTIFACT_ROOT),
        "requirements": requirements,
        "artifact_paths": artifact_paths or [],
        "artifacts": artifacts,
        "result_files": [],
    }


def _load_ppo():
    from stable_baselines3 import PPO

    return PPO


def _load_torch():
    import torch

    return torch


def _load_latent_belief_encoder():
    from pta.models.belief.latent_belief_encoder import LatentBeliefEncoder

    return LatentBeliefEncoder


def _load_json_object(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("metadata JSON must be an object")
    return payload


def _verify_policy_checkpoint(row: dict[str, object], ppo: object) -> None:
    model = ppo.load(str(row["source_path"]), device="auto")
    num_timesteps = getattr(model, "num_timesteps", None)
    row["num_timesteps"] = num_timesteps
    min_num_timesteps = row.get("min_num_timesteps")
    if min_num_timesteps is not None and (
        num_timesteps is None or num_timesteps < min_num_timesteps
    ):
        raise RuntimeError(
            f"num_timesteps {num_timesteps} below required minimum {min_num_timesteps}"
        )


def _expected_policy_metadata_paths(row: dict[str, object]) -> tuple[str, str]:
    metadata_path = PurePosixPath(str(row["relative_path"]))
    encoder_path = metadata_path.with_name("belief_encoder.pt")
    encoder_metadata_path = metadata_path.with_name("belief_encoder_metadata.json")
    return encoder_path.as_posix(), encoder_metadata_path.as_posix()


def _expected_encoder_metadata_paths(row: dict[str, object]) -> tuple[str, str]:
    metadata_path = PurePosixPath(str(row["relative_path"]))
    policy_path = metadata_path.with_name("best_model.zip")
    encoder_path = metadata_path.with_name("belief_encoder.pt")
    return policy_path.as_posix(), encoder_path.as_posix()


def _require_matched_policy_metadata(metadata: dict[str, object]) -> None:
    expected = {
        "protocol": "matched_encoder_v1",
        "encoder_mode": "matched",
        "legacy_policy_only": False,
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"policy metadata {key} must be {value!r}")


def _verify_policy_metadata(row: dict[str, object], repo_root: Path) -> None:
    metadata = _load_json_object(Path(str(row["source_path"])))
    _require_matched_policy_metadata(metadata)
    encoder_path, encoder_metadata_path = _expected_policy_metadata_paths(row)
    _metadata_hash_target(
        repo_root,
        metadata,
        "belief_encoder_path",
        "belief_encoder_sha256",
        expected_relative_path=encoder_path,
    )
    _metadata_hash_target(
        repo_root,
        metadata,
        "belief_encoder_metadata_path",
        "belief_encoder_metadata_sha256",
        expected_relative_path=encoder_metadata_path,
    )


def _verify_belief_encoder(row: dict[str, object], torch_module: object) -> None:
    payload = torch_module.load(
        str(row["source_path"]),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(payload, dict):
        raise ValueError("belief encoder checkpoint must be a dict")
    if payload.get("format_version") != 1:
        raise ValueError("belief encoder format_version must be 1")
    for field in ("state_dict", "config"):
        if field not in payload:
            raise ValueError(f"belief encoder missing {field}")
    if not isinstance(payload["config"], dict):
        raise ValueError("belief encoder config must be a dict")
    if not isinstance(payload["state_dict"], dict):
        raise ValueError("belief encoder state_dict must be a dict")
    encoder_cls = _load_latent_belief_encoder()
    encoder = encoder_cls(**payload["config"])
    encoder.load_state_dict(payload["state_dict"])


def _metadata_hash_target(
    repo_root: Path,
    metadata: dict[str, object],
    path_key: str,
    hash_key: str,
    expected_relative_path: str | None = None,
) -> None:
    repo_root = repo_root.resolve()
    relative_path = metadata.get(path_key)
    expected_sha256 = metadata.get(hash_key)
    if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
        raise ValueError(f"metadata missing {path_key}/{hash_key}")
    relative_path = validate_registry_relative_path(relative_path, {"checkpoints"}, path_key)
    if expected_relative_path is not None and relative_path != expected_relative_path:
        raise RuntimeError(
            f"{path_key} {relative_path} does not match expected {expected_relative_path}"
        )
    target = repo_root / relative_path
    if not target.is_file():
        raise FileNotFoundError(f"metadata target missing for {path_key}: {relative_path}")
    if target.is_symlink():
        raise RuntimeError(f"metadata target is symlink for {path_key}: {relative_path}")
    resolved_target = target.resolve()
    if not path_is_within(resolved_target, repo_root):
        raise RuntimeError(f"metadata target escapes repo root for {path_key}: {relative_path}")
    actual_sha256 = sha256_file(target)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"{hash_key} mismatch for {relative_path}")


def _verify_belief_encoder_metadata(row: dict[str, object], repo_root: Path) -> None:
    metadata = _load_json_object(Path(str(row["source_path"])))
    if metadata.get("protocol") != "matched_encoder_v1":
        raise ValueError("belief encoder metadata protocol must be 'matched_encoder_v1'")
    policy_path, encoder_path = _expected_encoder_metadata_paths(row)
    _metadata_hash_target(
        repo_root,
        metadata,
        "paired_policy_path",
        "paired_policy_sha256",
        expected_relative_path=policy_path,
    )
    _metadata_hash_target(
        repo_root,
        metadata,
        "belief_encoder_path",
        "belief_encoder_sha256",
        expected_relative_path=encoder_path,
    )


def verify_artifacts(
    repo_root: Path,
    requirements: list[str],
    artifact_paths: list[str] | None = None,
) -> dict[str, object]:
    manifest = build_scan_manifest(repo_root, requirements, artifact_paths)
    repo_root = repo_root.resolve()
    present_rows = [
        row
        for row in manifest["artifacts"]
        if isinstance(row, dict) and row.get("exists") and row.get("load_status") != "failed"
    ]
    ppo = None
    torch_module = None
    for row in present_rows:
        try:
            kind = row.get("kind")
            if kind == "policy_checkpoint":
                if ppo is None:
                    ppo = _load_ppo()
                _verify_policy_checkpoint(row, ppo)
            elif kind == "policy_metadata":
                _verify_policy_metadata(row, repo_root)
            elif kind == "belief_encoder":
                if torch_module is None:
                    torch_module = _load_torch()
                _verify_belief_encoder(row, torch_module)
            elif kind == "belief_encoder_metadata":
                _verify_belief_encoder_metadata(row, repo_root)
            else:
                raise ValueError(f"unknown artifact kind: {kind}")
        except Exception as exc:
            row["load_status"] = "failed"
            row["load_error"] = str(exc)
            continue
        row["load_status"] = "loaded"
    return manifest


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def missing_required(manifest: dict[str, object]) -> list[str]:
    return [
        str(row["logical_name"])
        for row in manifest["artifacts"]
        if isinstance(row, dict) and row.get("required") and not row.get("exists")
    ]


def failed_required_loads(manifest: dict[str, object]) -> list[str]:
    return [
        str(row["logical_name"])
        for row in manifest["artifacts"]
        if isinstance(row, dict) and row.get("required") and row.get("load_status") == "failed"
    ]


def remove_staging_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def validate_include_path(relative_path: str) -> None:
    validate_registry_relative_path(relative_path, {"logs", "results"}, "include path")


def reject_nested_symlinks(source: Path, relative_path: str) -> None:
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"include path contains symlink: {relative_path}")


def result_file_entry(relative_path: str, path: Path, run_dir: Path) -> dict[str, object]:
    return {
        "relative_path": relative_path,
        "storage_path": str((run_dir / relative_path).resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def copy_include_path(repo_root: Path, staging_dir: Path, run_dir: Path, relative_path: str) -> list[dict[str, object]]:
    relative_path = validate_registry_relative_path(relative_path, {"logs", "results"}, "include path")
    source = repo_root / relative_path
    if not source.exists():
        raise FileNotFoundError(f"include path not found: {relative_path}")
    if source.is_symlink():
        raise ValueError(f"include path is symlink: {relative_path}")
    resolved_repo_root = repo_root.resolve()
    resolved_source = source.resolve()
    if not path_is_within(resolved_source, resolved_repo_root):
        raise ValueError(f"include path escapes repo root: {relative_path}")
    destination = staging_dir / relative_path
    result_files: list[dict[str, object]] = []
    if source.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        if Path(relative_path).parts[0] == "results":
            result_files.append(result_file_entry(relative_path, destination, run_dir))
        return result_files
    if not source.is_dir():
        raise ValueError(f"include path is not a file or directory: {relative_path}")
    reject_nested_symlinks(source, relative_path)
    shutil.copytree(source, destination, dirs_exist_ok=True)
    if Path(relative_path).parts[0] == "results":
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                result_files.append(result_file_entry(path.relative_to(staging_dir).as_posix(), path, run_dir))
    return result_files


def copy_inferred_logs(repo_root: Path, staging_dir: Path, relative_path: str) -> None:
    parts = Path(relative_path).parts
    if len(parts) < 2 or parts[0] != "checkpoints":
        return
    run_name = parts[1]
    log_relative_path = validate_registry_relative_path(
        f"logs/{run_name}", {"logs"}, "inferred log path"
    )
    source = repo_root / log_relative_path
    if source.is_symlink():
        raise ValueError(f"inferred log path is symlink: {log_relative_path}")
    if not source.exists():
        return
    resolved_repo_root = repo_root.resolve()
    resolved_source = source.resolve()
    if not path_is_within(resolved_source, resolved_repo_root):
        raise ValueError(f"inferred log path escapes repo root: {log_relative_path}")
    if source.is_dir():
        reject_nested_symlinks(source, log_relative_path)
        shutil.copytree(source, staging_dir / log_relative_path, dirs_exist_ok=True)


def validate_manifest_file(run_dir: Path, row: dict[str, object], label: str, require_fields: bool = False) -> None:
    required_fields = ("relative_path", "storage_path", "size_bytes", "sha256")
    if require_fields:
        for field in required_fields:
            if row.get(field) is None:
                raise RuntimeError(f"manifest missing {field} for {label}")
    relative_path = row.get("relative_path")
    if not relative_path:
        return
    relative_path = validate_registry_relative_path(
        str(relative_path), set(RESTORE_TOP_LEVEL_DIRS), "manifest relative_path"
    )
    path = run_dir / relative_path
    if not path.is_file():
        raise RuntimeError(f"manifest mismatch for {label}: missing {relative_path}")
    size_bytes = path.stat().st_size
    sha256 = sha256_file(path)
    if size_bytes != row.get("size_bytes") or sha256 != row.get("sha256"):
        raise RuntimeError(f"manifest mismatch for {label}: {relative_path}")


def validate_run_manifest(run_dir: Path) -> None:
    manifest_path = run_dir / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for row in manifest.get("artifacts", []):
        if not isinstance(row, dict):
            continue
        if row.get("required") and row.get("load_status") != "loaded":
            raise RuntimeError(f"required artifact not loaded: {row.get('logical_name')}")
        require_fields = bool(
            row.get("required")
            or row.get("exists")
            or row.get("storage_path")
            or row.get("load_status") == "loaded"
        )
        if require_fields or row.get("relative_path"):
            validate_manifest_file(run_dir, row, str(row.get("logical_name")), require_fields)
    for row in manifest.get("result_files", []):
        if isinstance(row, dict):
            validate_manifest_file(run_dir, row, str(row.get("relative_path")))


def register_run(
    repo_root: Path,
    artifact_root: Path,
    run_id: str,
    origin: str,
    requirements: list[str],
    command: str,
    dlc_job_id: str | None = None,
    dlc_display_name: str | None = None,
    include_paths: list[str] | None = None,
    artifact_paths: list[str] | None = None,
) -> dict[str, object]:
    validate_run_id(run_id)
    repo_root = repo_root.resolve()
    manifest = verify_artifacts(repo_root, requirements, artifact_paths)
    missing = missing_required(manifest)
    if missing:
        raise FileNotFoundError("missing required artifacts: " + ", ".join(missing))
    failed = failed_required_loads(manifest)
    if failed:
        raise RuntimeError("failed required artifact loads: " + ", ".join(failed))

    artifact_root = artifact_root.resolve()
    run_dir = artifact_root / run_date_from_id(run_id) / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    staging_dir = run_dir.with_name(run_dir.name + ".tmp")
    if staging_dir.exists():
        raise FileExistsError(f"staging directory already exists: {staging_dir}")

    env = selected_env()
    manifest.update(
        {
            "run_id": run_id,
            "origin": origin,
            "dlc_job_id": dlc_job_id,
            "dlc_display_name": dlc_display_name,
            "command": command,
            "env": env,
            "artifact_root": str(artifact_root),
            "run_dir": str(run_dir.resolve()),
        }
    )

    try:
        staging_dir.mkdir(parents=True)
        for row in manifest["artifacts"]:
            if not isinstance(row, dict) or not row.get("exists"):
                continue
            relative_path = str(row["relative_path"])
            staging_destination = staging_dir / relative_path
            final_destination = run_dir / relative_path
            staging_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(Path(str(row["source_path"])), staging_destination)
            copied_size = staging_destination.stat().st_size
            copied_sha256 = sha256_file(staging_destination)
            if copied_size != row["size_bytes"] or copied_sha256 != row["sha256"]:
                raise RuntimeError(f"copied artifact mismatch: {row['logical_name']}")
            row["storage_path"] = str(final_destination.resolve())
            copy_inferred_logs(repo_root, staging_dir, relative_path)

        result_files: list[dict[str, object]] = []
        for include_path in include_paths or []:
            result_files.extend(copy_include_path(repo_root, staging_dir, run_dir, include_path))
        manifest["result_files"] = result_files

        (staging_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
        write_json(staging_dir / "env.json", env)
        write_json(staging_dir / "artifact_manifest.json", manifest)
        staging_dir.rename(run_dir)
    except Exception:
        remove_staging_dir(staging_dir)
        raise
    return manifest


def bundle_run(run_dir: Path, output: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    archive_path = (output or run_dir / "checkpoint_bundle.tar.gz").resolve()
    for name in RESTORE_TOP_LEVEL_DIRS:
        archived_dir = run_dir / name
        if archive_path != archived_dir and archived_dir in archive_path.parents:
            raise ValueError(f"output cannot be inside archived directory: {archived_dir}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    validate_run_manifest(run_dir)

    with tarfile.open(archive_path, "w:gz") as bundle:
        for name in RESTORE_TOP_LEVEL_FILES:
            path = run_dir / name
            if path.exists():
                bundle.add(path, arcname=name)
        for name in RESTORE_TOP_LEVEL_DIRS:
            path = run_dir / name
            if path.exists():
                bundle.add(path, arcname=name)
    return archive_path


def _safe_member_path(target_root: Path, member_name: str) -> Path:
    try:
        normalized_name = validate_registry_relative_path(
            member_name, RESTORE_TOP_LEVEL_NAMES, "archive member"
        )
    except ValueError as exc:
        raise ValueError(f"unsafe archive member: {member_name}") from exc
    parts = PurePosixPath(normalized_name).parts
    if parts[0] in RESTORE_TOP_LEVEL_FILES and len(parts) != 1:
        raise ValueError(f"unsafe archive member: {member_name}")

    target_root = target_root.resolve()
    destination = (target_root / normalized_name).resolve()
    if not path_is_within(destination, target_root):
        raise ValueError(f"unsafe archive member: {member_name}")
    return destination


def restore_bundle(archive_path: Path, target_root: Path) -> None:
    target_root = target_root.resolve()
    if target_root.exists() and any(target_root.iterdir()):
        raise FileExistsError(f"restore target must be empty: {target_root}")
    staging_root = target_root.with_name(target_root.name + ".tmp")
    if staging_root.exists():
        raise FileExistsError(f"staging directory already exists: {staging_root}")
    with tarfile.open(archive_path, "r:gz") as bundle:
        members = bundle.getmembers()
        for member in members:
            _safe_member_path(target_root, member.name)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsafe archive member: {member.name}")
        try:
            staging_root.mkdir(parents=True)
            bundle.extractall(staging_root, members=members)
            if not (staging_root / "artifact_manifest.json").is_file():
                raise RuntimeError("restore manifest missing: artifact_manifest.json")
            validate_run_manifest(staging_root)

            target_root.mkdir(parents=True, exist_ok=True)
            for name in RESTORE_TOP_LEVEL_FILES:
                source = staging_root / name
                if source.exists():
                    shutil.copy2(source, target_root / name)
            for name in RESTORE_TOP_LEVEL_DIRS:
                source = staging_root / name
                if source.exists():
                    shutil.copytree(source, target_root / name, dirs_exist_ok=True)
        finally:
            remove_staging_dir(staging_root)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe-Then-Act artifact registry")
    subparsers = parser.add_subparsers(dest="action", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    common.add_argument(
        "--requirement",
        action="append",
        choices=sorted(REQUIREMENTS),
        default=[],
    )
    common.add_argument("--artifact-path", action="append", default=[])
    common.add_argument("--manifest", type=Path)

    subparsers.add_parser("scan", parents=[common])
    subparsers.add_parser("verify", parents=[common])
    register_parser = subparsers.add_parser("register-run", parents=[common])
    register_parser.add_argument("--artifact-root", type=Path, required=True)
    register_parser.add_argument("--run-id", required=True)
    register_parser.add_argument(
        "--origin",
        choices=("local", "dlc", "recovered_by_retraining"),
        required=True,
    )
    register_parser.add_argument("--command", dest="run_command", required=True)
    register_parser.add_argument("--dlc-job-id")
    register_parser.add_argument("--dlc-display-name")
    register_parser.add_argument("--include-path", action="append", default=[])
    bundle_parser = subparsers.add_parser("bundle")
    bundle_parser.add_argument("--run-dir", type=Path, required=True)
    bundle_parser.add_argument("--output", type=Path)
    restore_parser = subparsers.add_parser("restore")
    restore_parser.add_argument("--archive", type=Path, required=True)
    restore_parser.add_argument("--target-root", type=Path, required=True)
    return parser.parse_args(argv)


def run_scan(args: argparse.Namespace) -> int:
    try:
        manifest = build_scan_manifest(args.repo_root, args.requirement, args.artifact_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    manifest_path = args.manifest or args.repo_root / "results" / "artifact_registry" / "scan_manifest.json"
    write_json(manifest_path, manifest)

    missing = missing_required(manifest)
    present = sum(
        1 for row in manifest["artifacts"] if isinstance(row, dict) and row.get("exists")
    )
    print(f"wrote {manifest_path} with {present}/{len(manifest['artifacts'])} present artifacts")
    if missing:
        print("missing required artifacts: " + ", ".join(missing))
        return 1
    return 0


def run_verify(args: argparse.Namespace) -> int:
    try:
        manifest = verify_artifacts(args.repo_root, args.requirement, args.artifact_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    manifest_path = args.manifest or args.repo_root / "results" / "artifact_registry" / "verify_manifest.json"
    write_json(manifest_path, manifest)

    missing = missing_required(manifest)
    failed = failed_required_loads(manifest)
    loaded = sum(
        1 for row in manifest["artifacts"] if isinstance(row, dict) and row.get("load_status") == "loaded"
    )
    print(f"wrote {manifest_path} with {loaded}/{len(manifest['artifacts'])} loaded artifacts")
    if missing:
        print("missing required artifacts: " + ", ".join(missing))
    if failed:
        print("failed required artifact loads: " + ", ".join(failed))
    return 1 if missing or failed else 0


def run_register(args: argparse.Namespace) -> int:
    try:
        manifest = register_run(
            repo_root=args.repo_root,
            artifact_root=args.artifact_root,
            run_id=args.run_id,
            origin=args.origin,
            requirements=args.requirement,
            command=args.run_command,
            dlc_job_id=args.dlc_job_id,
            dlc_display_name=args.dlc_display_name,
            include_paths=args.include_path,
            artifact_paths=args.artifact_path,
        )
    except (FileNotFoundError, RuntimeError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"registered run {args.run_id} at {manifest['run_dir']}")
    return 0


def run_bundle(args: argparse.Namespace) -> int:
    try:
        archive_path = bundle_run(args.run_dir, args.output)
    except (OSError, RuntimeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"wrote bundle {archive_path}")
    return 0


def run_restore(args: argparse.Namespace) -> int:
    try:
        restore_bundle(args.archive, args.target_root)
    except (tarfile.TarError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"restored bundle to {args.target_root}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.action == "scan":
        return run_scan(args)
    if args.action == "verify":
        return run_verify(args)
    if args.action == "register-run":
        return run_register(args)
    if args.action == "bundle":
        return run_bundle(args)
    if args.action == "restore":
        return run_restore(args)
    raise ValueError(args.action)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
