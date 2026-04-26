import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

ENV_FILES = [
    REPO_ROOT / ".env.dsw.example",
    REPO_ROOT / ".env.local.example",
]

SHELL_SCRIPTS = [
    REPO_ROOT / "scripts" / "bootstrap_remote.sh",
    REPO_ROOT / "scripts" / "smoke_remote.sh",
    REPO_ROOT / "scripts" / "download_artifacts.sh",
    REPO_ROOT / "pta" / "scripts" / "dlc" / "preflight_remote.sh",
]

DOCS = [
    REPO_ROOT / "docs" / "30_records" / "REMOTE_REPRODUCTION_RUNBOOK.md",
    REPO_ROOT / "docs" / "30_records" / "GENESIS_RUNTIME_LOCK.md",
    REPO_ROOT / "docs" / "30_records" / "CHECKPOINT_MANIFEST.md",
]

MANIFEST_SCRIPT = REPO_ROOT / "scripts" / "build_checkpoint_manifest.py"

REQUIRED_ARTIFACT_IDS = {
    "m1_reactive_seed42_best",
    "m1_reactive_seed0_best",
    "m1_reactive_seed1_best",
    "m7_pta_seed42_best",
    "m7_pta_seed0_best",
    "m7_pta_seed1_best",
    "m8_teacher_seed42_best",
    "m7_pta_noprobe_seed42_best",
    "m7_pta_noprobe_seed42_50000_steps",
}


def test_remote_handoff_assets_exist_and_reference_expected_remotes():
    for path in [*ENV_FILES, *SHELL_SCRIPTS, *DOCS, MANIFEST_SCRIPT]:
        assert path.exists(), f"missing {path.relative_to(REPO_ROOT)}"

    text = "\n".join(path.read_text(encoding="utf-8") for path in [*ENV_FILES, *DOCS])
    assert "git@github.com:jandan138/probe-then-act.git" in text
    assert "git@github.com:jandan138/Genesis.git" in text
    assert "checkpoints are not stored in normal Git" in text
    assert "Current shortest path: do not upload checkpoints before training" in text
    assert "--skip no_probe:42" in text
    assert "five remaining ablation jobs" in text
    assert "Do not set `BOOTSTRAP_SKIP_INSTALL=1` on a fresh machine" in text
    assert "quadrants" in text


def test_handoff_shell_scripts_are_executable_and_parse():
    for script in SHELL_SCRIPTS:
        mode = script.stat().st_mode
        assert mode & stat.S_IXUSR, f"{script.relative_to(REPO_ROOT)} is not executable"
        result = subprocess.run(
            ["bash", "-n", str(script)],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr


def test_checkpoint_manifest_builder_lists_required_artifacts(tmp_path):
    repo_root = tmp_path / "probe"
    stage_d_root = tmp_path / "stage-d"
    repo_root.mkdir()
    stage_d_root.mkdir()
    existing_main = (
        repo_root
        / "checkpoints"
        / "m1_reactive_seed42"
        / "best"
        / "best_model.zip"
    )
    existing_stage_d = (
        stage_d_root
        / "checkpoints"
        / "m7_pta_noprobe_seed42"
        / "m7_pta_50000_steps.zip"
    )
    existing_main.parent.mkdir(parents=True)
    existing_stage_d.parent.mkdir(parents=True)
    existing_main.write_bytes(b"main checkpoint")
    existing_stage_d.write_bytes(b"stage checkpoint")

    manifest_path = tmp_path / "manifest.json"
    result = subprocess.run(
        [
            sys.executable,
            str(MANIFEST_SCRIPT),
            "--repo-root",
            str(repo_root),
            "--stage-d-root",
            str(stage_d_root),
            "--manifest",
            str(manifest_path),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_id = {row["id"]: row for row in manifest["artifacts"]}
    assert REQUIRED_ARTIFACT_IDS <= set(by_id)
    assert by_id["m1_reactive_seed42_best"]["exists"] is True
    assert by_id["m1_reactive_seed42_best"]["source_root"] == "repo_root"
    assert by_id["m1_reactive_seed42_best"]["sha256"] == hashlib.sha256(
        b"main checkpoint"
    ).hexdigest()
    assert by_id["m7_pta_noprobe_seed42_50000_steps"]["exists"] is True
    assert by_id["m7_pta_noprobe_seed42_50000_steps"]["source_root"] == "stage_d_root"
    assert by_id["m7_pta_seed1_best"]["exists"] is False
    assert manifest["artifact_policy"] == "checkpoints are not stored in normal Git"
    assert os.path.isabs(manifest["repo_root"])
