# Artifact Storage Registry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a durable artifact registry CLI that can scan, verify, register, bundle, and restore claim-critical Probe-Then-Act checkpoints without storing checkpoint binaries in Git.

**Architecture:** Add one focused stdlib module, `tools/artifact_registry.py`, with pure functions for requirement definitions, hashing, checkpoint verification, manifest construction, copy/bundle/restore, and a thin argparse CLI. Add `tests/test_artifact_registry.py` with tiny fake artifacts and monkeypatched SB3 loading so tests stay fast and do not require Genesis or GPU runtime.

**Tech Stack:** Python standard library, `argparse`, `hashlib`, `json`, `shutil`, `tarfile`, lazy `stable_baselines3.PPO` import, and pytest with `tmp_path`/`monkeypatch`.

---

### Task 1: Registry Contract Tests

**Files:**
- Create: `tests/test_artifact_registry.py`

- [ ] **Step 1: Write failing tests for requirement sets and missing required artifacts**

Create `tests/test_artifact_registry.py` with these imports, helpers, and tests:

```python
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tools import artifact_registry as registry


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "artifact_registry.py"


def _write(path: Path, data: bytes = b"checkpoint") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_requirement_sets_include_presub_artifacts():
    assert registry.requirement_paths("presub-g2") == [
        "checkpoints/m7_pta_seed42/best/best_model.zip"
    ]
    assert set(registry.requirement_paths("presub-extra-eval")) == {
        "checkpoints/m1_reactive_seed2/best/best_model.zip",
        "checkpoints/m1_reactive_seed3/best/best_model.zip",
        "checkpoints/m7_pta_seed2/best/best_model.zip",
        "checkpoints/m7_pta_seed3/best/best_model.zip",
    }


def test_scan_missing_required_exits_nonzero(tmp_path):
    manifest_path = tmp_path / "scan.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "scan",
            "--repo-root",
            str(tmp_path),
            "--requirement",
            "presub-g2",
            "--manifest",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "m7_pta_seed42_best" in result.stdout
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["artifacts"][0]["exists"] is False
    assert manifest["artifacts"][0]["sha256"] is None
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_artifact_registry.py -q`

Expected: FAIL during import because `tools.artifact_registry` does not exist.

### Task 2: Scan Command And Requirement Model

**Files:**
- Create: `tools/artifact_registry.py`
- Modify: `tests/test_artifact_registry.py`

- [ ] **Step 1: Implement minimal registry data model and `scan`**

Create `tools/artifact_registry.py` with this structure:

```python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACT_ROOT = Path("/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act")
SCHEMA_VERSION = 1
PROJECT_NAME = "probe-then-act"


@dataclass(frozen=True)
class ArtifactCandidate:
    logical_name: str
    relative_path: str
    kind: str = "checkpoint"
    required_for: str = "manual"
    required: bool = True


def _best_checkpoint(method: str, seed: int) -> str:
    return f"checkpoints/{method}_seed{seed}/best/best_model.zip"


REQUIREMENTS: dict[str, tuple[ArtifactCandidate, ...]] = {
    "presub-g2": (
        ArtifactCandidate(
            "m7_pta_seed42_best",
            _best_checkpoint("m7_pta", 42),
            required_for="presub-g2",
        ),
    ),
    "presub-extra-eval": tuple(
        ArtifactCandidate(
            f"{method}_seed{seed}_best",
            _best_checkpoint(method, seed),
            required_for="presub-extra-eval",
        )
        for method in ("m1_reactive", "m7_pta")
        for seed in (2, 3)
    ),
    "corrected-ood-replay": (
        *tuple(
            ArtifactCandidate(
                f"m1_reactive_seed{seed}_best",
                _best_checkpoint("m1_reactive", seed),
                required_for="corrected-ood-replay",
            )
            for seed in (42, 0, 1)
        ),
        *tuple(
            ArtifactCandidate(
                f"m7_pta_seed{seed}_best",
                _best_checkpoint("m7_pta", seed),
                required_for="corrected-ood-replay",
            )
            for seed in (42, 0, 1)
        ),
        ArtifactCandidate(
            "m8_teacher_seed42_best",
            _best_checkpoint("m8_teacher", 42),
            required_for="corrected-ood-replay",
        ),
    ),
    "ablation-replay": (
        *tuple(
            ArtifactCandidate(
                f"m7_pta_noprobe_seed{seed}_final",
                f"checkpoints/m7_pta_noprobe_seed{seed}/m7_pta_final.zip",
                required_for="ablation-replay",
            )
            for seed in (42, 0, 1)
        ),
        *tuple(
            ArtifactCandidate(
                f"m7_pta_nobelief_seed{seed}_final",
                f"checkpoints/m7_pta_nobelief_seed{seed}/m7_pta_final.zip",
                required_for="ablation-replay",
            )
            for seed in (42, 0, 1)
        ),
    ),
}
```

Implement helpers:

```python
def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def requirement_paths(name: str) -> list[str]:
    return [candidate.relative_path for candidate in REQUIREMENTS[name]]


def selected_candidates(requirements: Iterable[str]) -> list[ArtifactCandidate]:
    candidates: list[ArtifactCandidate] = []
    seen: set[str] = set()
    for requirement in requirements:
        for candidate in REQUIREMENTS[requirement]:
            if candidate.logical_name in seen:
                continue
            seen.add(candidate.logical_name)
            candidates.append(candidate)
    return candidates


def artifact_status(candidate: ArtifactCandidate, repo_root: Path) -> dict[str, object]:
    source_path = repo_root / candidate.relative_path
    exists = source_path.exists()
    return {
        "logical_name": candidate.logical_name,
        "relative_path": candidate.relative_path,
        "source_path": str(source_path.resolve()) if exists else None,
        "storage_path": None,
        "size_bytes": source_path.stat().st_size if exists else 0,
        "sha256": sha256_file(source_path) if exists else None,
        "kind": candidate.kind,
        "required_for": candidate.required_for,
        "required": candidate.required,
        "exists": exists,
        "num_timesteps": None,
        "load_status": "not_checked",
    }
```

Implement manifest and CLI dispatch:

```python
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


def build_scan_manifest(repo_root: Path, requirements: list[str]) -> dict[str, object]:
    repo_root = repo_root.resolve()
    artifacts = [artifact_status(candidate, repo_root) for candidate in selected_candidates(requirements)]
    return {
        "schema_version": SCHEMA_VERSION,
        "project": PROJECT_NAME,
        "created_at_utc": utc_now(),
        "git_commit": git_commit(repo_root),
        "repo_root_at_registration": str(repo_root),
        "requirements": requirements,
        "artifacts": artifacts,
        "result_files": [],
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def missing_required(manifest: dict[str, object]) -> list[str]:
    return [
        str(row["logical_name"])
        for row in manifest["artifacts"]
        if isinstance(row, dict) and row.get("required") and not row.get("exists")
    ]
```

CLI expectations:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe-Then-Act artifact registry")
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    common.add_argument("--requirement", action="append", choices=sorted(REQUIREMENTS), required=True)
    common.add_argument("--manifest", type=Path)
    scan = subparsers.add_parser("scan", parents=[common])
    return parser.parse_args(argv)


def run_scan(args: argparse.Namespace) -> int:
    manifest = build_scan_manifest(args.repo_root, args.requirement)
    manifest_path = args.manifest or args.repo_root / "results" / "artifact_registry" / "scan_manifest.json"
    write_json(manifest_path, manifest)
    missing = missing_required(manifest)
    present = sum(1 for row in manifest["artifacts"] if isinstance(row, dict) and row.get("exists"))
    print(f"wrote {manifest_path} with {present}/{len(manifest['artifacts'])} present artifacts")
    if missing:
        print("missing required artifacts: " + ", ".join(missing))
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "scan":
        return run_scan(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
```

- [ ] **Step 2: Run focused tests and verify GREEN for scan**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_artifact_registry.py -q`

Expected: PASS for Task 1 tests.

### Task 3: SB3 Verification Command

**Files:**
- Modify: `tools/artifact_registry.py`
- Modify: `tests/test_artifact_registry.py`

- [ ] **Step 1: Add failing tests for checkpoint load success and load failure**

Append tests:

```python
class FakeModel:
    num_timesteps = 500000


class FakePPO:
    loaded_paths = []

    @staticmethod
    def load(path: str, device: str = "auto"):
        FakePPO.loaded_paths.append((path, device))
        return FakeModel()


def test_verify_loads_checkpoint_and_records_timesteps(tmp_path, monkeypatch):
    checkpoint = _write(
        tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip",
        b"verified checkpoint",
    )
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)

    manifest = registry.verify_artifacts(tmp_path, ["presub-g2"])

    row = manifest["artifacts"][0]
    assert row["load_status"] == "loaded"
    assert row["num_timesteps"] == 500000
    assert row["sha256"] == hashlib.sha256(b"verified checkpoint").hexdigest()
    assert FakePPO.loaded_paths[-1] == (str(checkpoint.resolve()), "auto")


def test_verify_load_failure_marks_required_artifact_failed(tmp_path, monkeypatch):
    _write(tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip")

    class BrokenPPO:
        @staticmethod
        def load(path: str, device: str = "auto"):
            raise RuntimeError("bad zip")

    monkeypatch.setattr(registry, "_load_ppo", lambda: BrokenPPO)

    manifest = registry.verify_artifacts(tmp_path, ["presub-g2"])

    row = manifest["artifacts"][0]
    assert row["load_status"] == "failed"
    assert row["load_error"] == "bad zip"
    assert registry.failed_required_loads(manifest) == ["m7_pta_seed42_best"]
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_artifact_registry.py -q`

Expected: FAIL because `verify_artifacts`, `_load_ppo`, and `failed_required_loads` do not exist.

- [ ] **Step 3: Implement lazy SB3 load verification and `verify` CLI**

Add to `tools/artifact_registry.py`:

```python
def _load_ppo():
    from stable_baselines3 import PPO

    return PPO


def verify_artifacts(repo_root: Path, requirements: list[str]) -> dict[str, object]:
    manifest = build_scan_manifest(repo_root, requirements)
    PPO = _load_ppo()
    for row in manifest["artifacts"]:
        if not isinstance(row, dict) or not row.get("exists"):
            continue
        source_path = row["source_path"]
        if not isinstance(source_path, str):
            continue
        try:
            model = PPO.load(source_path, device="auto")
        except Exception as exc:  # noqa: BLE001 - record third-party load errors verbatim.
            row["load_status"] = "failed"
            row["load_error"] = str(exc)
            continue
        row["load_status"] = "loaded"
        row["num_timesteps"] = getattr(model, "num_timesteps", None)
    return manifest


def failed_required_loads(manifest: dict[str, object]) -> list[str]:
    return [
        str(row["logical_name"])
        for row in manifest["artifacts"]
        if isinstance(row, dict) and row.get("required") and row.get("load_status") == "failed"
    ]
```

Add `verify` subparser using the same common options as `scan`, then dispatch:

```python
def run_verify(args: argparse.Namespace) -> int:
    manifest = verify_artifacts(args.repo_root, args.requirement)
    manifest_path = args.manifest or args.repo_root / "results" / "artifact_registry" / "verify_manifest.json"
    write_json(manifest_path, manifest)
    missing = missing_required(manifest)
    failed = failed_required_loads(manifest)
    print(f"wrote {manifest_path}")
    if missing:
        print("missing required artifacts: " + ", ".join(missing))
    if failed:
        print("failed required loads: " + ", ".join(failed))
    return 1 if missing or failed else 0
```

- [ ] **Step 4: Run focused tests and verify GREEN for scan/verify**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_artifact_registry.py -q`

Expected: PASS.

### Task 4: Register-Run Manifest And Durable Copy

**Files:**
- Modify: `tools/artifact_registry.py`
- Modify: `tests/test_artifact_registry.py`

- [ ] **Step 1: Add failing register-run test**

Append test:

```python
def test_register_run_copies_artifact_and_writes_manifest(tmp_path, monkeypatch):
    checkpoint = _write(
        tmp_path / "repo" / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip",
        b"durable checkpoint",
    )
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)

    manifest = registry.register_run(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifact-root",
        run_id="20260501_recovery",
        origin="recovered_by_retraining",
        requirements=["presub-g2"],
        command="python train_m7.py --seed 42",
        dlc_job_id="dlc1hn82yye94ojd",
        dlc_display_name="pta_recover_m7_s42_0_1",
    )

    run_dir = tmp_path / "artifact-root" / "20260501" / "20260501_recovery"
    copied = run_dir / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip"
    assert copied.read_bytes() == b"durable checkpoint"
    assert (run_dir / "command.txt").read_text(encoding="utf-8") == "python train_m7.py --seed 42\n"
    saved = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
    row = saved["artifacts"][0]
    assert manifest["run_id"] == "20260501_recovery"
    assert saved["origin"] == "recovered_by_retraining"
    assert saved["dlc_job_id"] == "dlc1hn82yye94ojd"
    assert row["storage_path"] == str(copied.resolve())
    assert row["sha256"] == hashlib.sha256(b"durable checkpoint").hexdigest()
    assert Path(row["source_path"]) == checkpoint.resolve()
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_artifact_registry.py -q`

Expected: FAIL because `register_run` does not exist.

- [ ] **Step 3: Implement register-run**

Add imports and helpers:

```python
import shutil


def run_date_from_id(run_id: str) -> str:
    prefix = run_id[:8]
    if len(prefix) == 8 and prefix.isdigit():
        return prefix
    return datetime.now(timezone.utc).strftime("%Y%m%d")


def selected_env() -> dict[str, str]:
    keys = ["PYTHONPATH", "PYOPENGL_PLATFORM", "EGL_DEVICE_ID", "CUDA_VISIBLE_DEVICES", "DLC_JOB_ID", "DLC_RUN_ID"]
    return {key: os.environ[key] for key in keys if key in os.environ}
```

Implement `register_run`:

```python
def register_run(
    *,
    repo_root: Path,
    artifact_root: Path,
    run_id: str,
    origin: str,
    requirements: list[str],
    command: str,
    dlc_job_id: str | None = None,
    dlc_display_name: str | None = None,
) -> dict[str, object]:
    manifest = verify_artifacts(repo_root, requirements)
    missing = missing_required(manifest)
    failed = failed_required_loads(manifest)
    if missing:
        raise FileNotFoundError("missing required artifacts: " + ", ".join(missing))
    if failed:
        raise RuntimeError("failed required loads: " + ", ".join(failed))

    run_dir = artifact_root / run_date_from_id(run_id) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    for row in manifest["artifacts"]:
        if not isinstance(row, dict) or not row.get("exists"):
            continue
        source_path = row.get("source_path")
        relative_path = row.get("relative_path")
        if not isinstance(source_path, str) or not isinstance(relative_path, str):
            continue
        destination = run_dir / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        row["storage_path"] = str(destination.resolve())

    manifest.update(
        {
            "run_id": run_id,
            "origin": origin,
            "dlc_job_id": dlc_job_id,
            "dlc_display_name": dlc_display_name,
            "command": command,
            "env": selected_env(),
            "artifact_root": str(artifact_root.resolve()),
            "run_dir": str(run_dir.resolve()),
        }
    )
    (run_dir / "command.txt").write_text(command.rstrip() + "\n", encoding="utf-8")
    write_json(run_dir / "env.json", manifest["env"])
    write_json(run_dir / "artifact_manifest.json", manifest)
    return manifest
```

Add `register-run` CLI args: `--artifact-root`, `--run-id`, `--origin` choices `local`, `dlc`, `recovered_by_retraining`, `--command`, `--dlc-job-id`, and `--dlc-display-name`. Dispatch returns nonzero with clear error text when required artifacts are missing or load verification fails.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_artifact_registry.py -q`

Expected: PASS.

### Task 5: Bundle And Restore

**Files:**
- Modify: `tools/artifact_registry.py`
- Modify: `tests/test_artifact_registry.py`

- [ ] **Step 1: Add failing bundle/restore test**

Append test:

```python
def test_bundle_and_restore_recreates_checkpoint_tree(tmp_path, monkeypatch):
    _write(
        tmp_path / "repo" / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip",
        b"bundle checkpoint",
    )
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)
    manifest = registry.register_run(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifact-root",
        run_id="20260501_bundle",
        origin="local",
        requirements=["presub-g2"],
        command="train",
    )
    run_dir = Path(manifest["run_dir"])
    archive = registry.bundle_run(run_dir)

    restore_root = tmp_path / "restore"
    registry.restore_bundle(archive, restore_root)

    restored = restore_root / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip"
    assert restored.read_bytes() == b"bundle checkpoint"
    assert (restore_root / "artifact_manifest.json").exists()
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_artifact_registry.py -q`

Expected: FAIL because `bundle_run` and `restore_bundle` do not exist.

- [ ] **Step 3: Implement bundle and restore**

Add import `tarfile` and functions:

```python
def bundle_run(run_dir: Path, output: Path | None = None) -> Path:
    run_dir = run_dir.resolve()
    output = output or run_dir / "checkpoint_bundle.tar.gz"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for relative in ("artifact_manifest.json", "command.txt", "env.json"):
            path = run_dir / relative
            if path.exists():
                archive.add(path, arcname=relative)
        for directory in ("checkpoints", "logs", "results"):
            path = run_dir / directory
            if path.exists():
                archive.add(path, arcname=directory)
    return output


def _safe_member_path(target_root: Path, member_name: str) -> Path:
    destination = (target_root / member_name).resolve()
    root = target_root.resolve()
    if destination != root and root not in destination.parents:
        raise ValueError(f"refusing to restore unsafe archive member: {member_name}")
    return destination


def restore_bundle(archive_path: Path, target_root: Path) -> None:
    target_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            _safe_member_path(target_root, member.name)
        archive.extractall(target_root)
```

Add `bundle` CLI with `--run-dir` and optional `--output`. Add `restore` CLI with `--archive` and `--target-root`.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_artifact_registry.py -q`

Expected: PASS.

### Task 6: Integration Verification And Recovery Hook Readiness

**Files:**
- Read: `tools/artifact_registry.py`
- Read: `tests/test_artifact_registry.py`
- Read: `docs/superpowers/specs/2026-05-01-artifact-storage-registry-design.md`

- [ ] **Step 1: Run focused artifact tests**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_artifact_registry.py -q`

Expected: PASS.

- [ ] **Step 2: Run existing relevant tests**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_remote_handoff_assets.py tests/test_pre_submission_audit.py tests/test_dlc_submit_jobs.py tests/test_dlc_shell_contract.py -q`

Expected: PASS.

- [ ] **Step 3: Run CLI scan against the live DLC staged tree**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python tools/artifact_registry.py scan --repo-root /cpfs/shared/simulation/zhuzihou/dev/probe-then-act --requirement presub-g2 --manifest /tmp/pta_presub_g2_scan.json`

Expected while recovery is still training: exit `1`, stdout includes `missing required artifacts: m7_pta_seed42_best`, and `/tmp/pta_presub_g2_scan.json` records `exists=false`.

- [ ] **Step 4: Run diff and syntax checks**

Run: `git diff --check`

Expected: no output and exit `0`.

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m py_compile tools/artifact_registry.py`

Expected: exit `0`.

- [ ] **Step 5: Check git status without committing**

Run: `git status --short --branch`

Expected: source changes only for the artifact registry spec, plan, tool, and tests. Do not commit unless the operator explicitly asks for a commit.

### Task 7: Register Recovery Artifact When Available

**Files:**
- Generated: `/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/<date>/<run_id>/artifact_manifest.json`
- Generated: `/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/<date>/<run_id>/checkpoint_bundle.tar.gz`

- [ ] **Step 1: Poll the recovery checkpoint condition**

Run: `python tools/artifact_registry.py scan --repo-root /cpfs/shared/simulation/zhuzihou/dev/probe-then-act --requirement presub-g2 --manifest /tmp/pta_presub_g2_scan.json`

Expected before completion: exit `1`; after checkpoint appears: exit `0`.

- [ ] **Step 2: Verify checkpoint load**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python tools/artifact_registry.py verify --repo-root /cpfs/shared/simulation/zhuzihou/dev/probe-then-act --requirement presub-g2 --manifest /tmp/pta_presub_g2_verify.json`

Expected after checkpoint appears: exit `0`, manifest row has `load_status=loaded`, `sha256` set, and `num_timesteps` recorded.

- [ ] **Step 3: Register the recovered run**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python tools/artifact_registry.py register-run --repo-root /cpfs/shared/simulation/zhuzihou/dev/probe-then-act --artifact-root /cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act --run-id 20260501_dlc1hn82yye94ojd_m7_pta_seed42_recovery --origin recovered_by_retraining --dlc-job-id dlc1hn82yye94ojd --dlc-display-name pta_recover_m7_s42_0_1 --command "bash /cpfs/shared/simulation/zhuzihou/dev/probe-then-act/pta/scripts/dlc/run_task.sh custom env PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 /cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -u pta/scripts/train_m7.py --seed 42 --total-timesteps 500000 --residual-scale 0.05 --horizon 500 --latent-dim 16 --n-probes 3 --eval-freq 50000" --requirement presub-g2`

Expected: manifest and copied checkpoint under `/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/20260501/20260501_dlc1hn82yye94ojd_m7_pta_seed42_recovery/`.

- [ ] **Step 4: Bundle the registered run**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python tools/artifact_registry.py bundle --run-dir /cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act/20260501/20260501_dlc1hn82yye94ojd_m7_pta_seed42_recovery`

Expected: `checkpoint_bundle.tar.gz` exists in the run directory.
