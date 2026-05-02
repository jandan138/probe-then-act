import hashlib
import io
import json
import pytest
import subprocess
import sys
import tarfile
from pathlib import Path

from tools import artifact_registry as registry


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "tools" / "artifact_registry.py"


def _write(path: Path, data: bytes = b"checkpoint") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_matched_policy_metadata(
    repo_root: Path,
    *,
    encoder_sha256: str | None = None,
) -> Path:
    encoder_path = repo_root / "checkpoints" / "m7_pta_seed42" / "best" / "belief_encoder.pt"
    metadata_path = (
        repo_root
        / "checkpoints"
        / "m7_pta_seed42"
        / "best"
        / "belief_encoder_metadata.json"
    )
    return _write_json(
        repo_root / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.json",
        {
            "protocol": "matched_encoder_v1",
            "encoder_mode": "matched",
            "legacy_policy_only": False,
            "belief_encoder_path": "checkpoints/m7_pta_seed42/best/belief_encoder.pt",
            "belief_encoder_sha256": encoder_sha256
            or hashlib.sha256(encoder_path.read_bytes()).hexdigest(),
            "belief_encoder_metadata_path": (
                "checkpoints/m7_pta_seed42/best/belief_encoder_metadata.json"
            ),
            "belief_encoder_metadata_sha256": hashlib.sha256(
                metadata_path.read_bytes()
            ).hexdigest(),
        },
    )


def _write_matched_encoder_artifacts(
    repo_root: Path,
    *,
    policy_bytes: bytes = b"policy checkpoint",
    encoder_bytes: bytes = b"encoder checkpoint",
    policy_sha256: str | None = None,
    encoder_sha256: str | None = None,
) -> None:
    policy_path = _write(
        repo_root / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip",
        policy_bytes,
    )
    encoder_path = _write(
        repo_root / "checkpoints" / "m7_pta_seed42" / "best" / "belief_encoder.pt",
        encoder_bytes,
    )
    _write_json(
        repo_root
        / "checkpoints"
        / "m7_pta_seed42"
        / "best"
        / "belief_encoder_metadata.json",
        {
            "protocol": "matched_encoder_v1",
            "paired_policy_path": "checkpoints/m7_pta_seed42/best/best_model.zip",
            "paired_policy_sha256": policy_sha256 or hashlib.sha256(policy_bytes).hexdigest(),
            "belief_encoder_path": "checkpoints/m7_pta_seed42/best/belief_encoder.pt",
            "belief_encoder_sha256": encoder_sha256 or hashlib.sha256(encoder_bytes).hexdigest(),
        },
    )
    _write_matched_policy_metadata(repo_root, encoder_sha256=encoder_sha256)


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


def test_requirement_paths_include_matched_encoder_sidecars():
    assert registry.requirement_paths("g2-matched-encoder") == [
        "checkpoints/m7_pta_seed42/best/best_model.zip",
        "checkpoints/m7_pta_seed42/best/best_model.json",
        "checkpoints/m7_pta_seed42/best/belief_encoder.pt",
        "checkpoints/m7_pta_seed42/best/belief_encoder_metadata.json",
    ]


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


class FakeModel:
    num_timesteps = 500000


class FakePPO:
    loaded_paths = []

    @staticmethod
    def load(path: str, device: str = "auto"):
        FakePPO.loaded_paths.append((path, device))
        return FakeModel()


class FakeTorch:
    loaded_paths = []

    @staticmethod
    def load(path: str, **kwargs):
        FakeTorch.loaded_paths.append((path, kwargs))
        return {
            "format_version": 1,
            "state_dict": {"encoder.weight": b"weights"},
            "config": {"trace_dim": 30, "latent_dim": 16},
        }


class FakeLatentBeliefEncoder:
    def __init__(self, **config):
        self.config = config

    def load_state_dict(self, state_dict):
        if state_dict.get("wrong_shape"):
            raise RuntimeError("size mismatch")
        return None


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


def test_verify_uses_type_specific_validation_for_matched_encoder_artifacts(tmp_path, monkeypatch):
    _write_matched_encoder_artifacts(tmp_path)
    FakePPO.loaded_paths.clear()
    FakeTorch.loaded_paths.clear()
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)
    monkeypatch.setattr(registry, "_load_torch", lambda: FakeTorch)
    monkeypatch.setattr(
        registry, "_load_latent_belief_encoder", lambda: FakeLatentBeliefEncoder, raising=False
    )

    manifest = registry.verify_artifacts(tmp_path, ["g2-matched-encoder"])

    rows = {row["logical_name"]: row for row in manifest["artifacts"]}
    assert rows["m7_pta_seed42_best"]["kind"] == "policy_checkpoint"
    assert rows["m7_pta_seed42_best"]["load_status"] == "loaded"
    assert rows["m7_pta_seed42_best_metadata"]["kind"] == "policy_metadata"
    assert rows["m7_pta_seed42_best_metadata"]["load_status"] == "loaded"
    assert rows["m7_pta_seed42_belief_encoder"]["kind"] == "belief_encoder"
    assert rows["m7_pta_seed42_belief_encoder"]["load_status"] == "loaded"
    assert rows["m7_pta_seed42_belief_encoder_metadata"]["kind"] == "belief_encoder_metadata"
    assert rows["m7_pta_seed42_belief_encoder_metadata"]["load_status"] == "loaded"
    assert FakePPO.loaded_paths == [
        (str((tmp_path / "checkpoints/m7_pta_seed42/best/best_model.zip").resolve()), "auto")
    ]
    assert FakeTorch.loaded_paths == [
        (
            str((tmp_path / "checkpoints/m7_pta_seed42/best/belief_encoder.pt").resolve()),
            {"map_location": "cpu", "weights_only": True},
        )
    ]


def test_verify_metadata_hash_mismatch_fails_required_matched_encoder_load(tmp_path, monkeypatch):
    _write_matched_encoder_artifacts(tmp_path)
    _write_json(
        tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "belief_encoder_metadata.json",
        {
            "protocol": "matched_encoder_v1",
            "paired_policy_path": "checkpoints/m7_pta_seed42/best/best_model.zip",
            "paired_policy_sha256": hashlib.sha256(b"policy checkpoint").hexdigest(),
            "belief_encoder_path": "checkpoints/m7_pta_seed42/best/belief_encoder.pt",
            "belief_encoder_sha256": "0" * 64,
        },
    )
    _write_matched_policy_metadata(tmp_path)
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)
    monkeypatch.setattr(registry, "_load_torch", lambda: FakeTorch)
    monkeypatch.setattr(
        registry, "_load_latent_belief_encoder", lambda: FakeLatentBeliefEncoder, raising=False
    )

    manifest = registry.verify_artifacts(tmp_path, ["g2-matched-encoder"])

    rows = {row["logical_name"]: row for row in manifest["artifacts"]}
    assert rows["m7_pta_seed42_belief_encoder_metadata"]["load_status"] == "failed"
    assert "belief_encoder_sha256" in rows["m7_pta_seed42_belief_encoder_metadata"]["load_error"]
    assert registry.failed_required_loads(manifest) == ["m7_pta_seed42_belief_encoder_metadata"]


def test_verify_metadata_hash_target_rejects_symlink_escaping_repo_root(tmp_path, monkeypatch):
    _write_matched_encoder_artifacts(tmp_path)
    outside_encoder = _write(tmp_path / "outside" / "belief_encoder.pt", b"escaped encoder")
    escaped_link = tmp_path / "checkpoints" / "escaped_belief_encoder.pt"
    try:
        escaped_link.symlink_to(outside_encoder)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    _write_json(
        tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "belief_encoder_metadata.json",
        {
            "protocol": "matched_encoder_v1",
            "paired_policy_path": "checkpoints/m7_pta_seed42/best/best_model.zip",
            "paired_policy_sha256": hashlib.sha256(b"policy checkpoint").hexdigest(),
            "belief_encoder_path": "checkpoints/escaped_belief_encoder.pt",
            "belief_encoder_sha256": hashlib.sha256(b"escaped encoder").hexdigest(),
        },
    )
    _write_matched_policy_metadata(tmp_path)
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)
    monkeypatch.setattr(registry, "_load_torch", lambda: FakeTorch)
    monkeypatch.setattr(
        registry, "_load_latent_belief_encoder", lambda: FakeLatentBeliefEncoder, raising=False
    )

    manifest = registry.verify_artifacts(tmp_path, ["g2-matched-encoder"])

    rows = {row["logical_name"]: row for row in manifest["artifacts"]}
    assert rows["m7_pta_seed42_belief_encoder_metadata"]["load_status"] == "failed"
    assert registry.failed_required_loads(manifest) == ["m7_pta_seed42_belief_encoder_metadata"]


def test_verify_invalid_belief_encoder_state_dict_fails_required_load(tmp_path, monkeypatch):
    _write_matched_encoder_artifacts(tmp_path)

    class BadTorch:
        @staticmethod
        def load(path: str, **kwargs):
            return {
                "format_version": 1,
                "state_dict": {"wrong_shape": True},
                "config": {"trace_dim": 30, "latent_dim": 16},
            }

    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)
    monkeypatch.setattr(registry, "_load_torch", lambda: BadTorch)
    monkeypatch.setattr(
        registry, "_load_latent_belief_encoder", lambda: FakeLatentBeliefEncoder, raising=False
    )

    manifest = registry.verify_artifacts(tmp_path, ["g2-matched-encoder"])

    rows = {row["logical_name"]: row for row in manifest["artifacts"]}
    assert rows["m7_pta_seed42_belief_encoder"]["load_status"] == "failed"
    assert registry.failed_required_loads(manifest) == ["m7_pta_seed42_belief_encoder"]


def test_verify_policy_metadata_hash_mismatch_fails_required_load(tmp_path, monkeypatch):
    _write_matched_encoder_artifacts(tmp_path)
    _write_matched_policy_metadata(tmp_path, encoder_sha256="0" * 64)
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)
    monkeypatch.setattr(registry, "_load_torch", lambda: FakeTorch)
    monkeypatch.setattr(
        registry, "_load_latent_belief_encoder", lambda: FakeLatentBeliefEncoder, raising=False
    )

    manifest = registry.verify_artifacts(tmp_path, ["g2-matched-encoder"])

    rows = {row["logical_name"]: row for row in manifest["artifacts"]}
    assert rows["m7_pta_seed42_best_metadata"]["load_status"] == "failed"
    assert registry.failed_required_loads(manifest) == ["m7_pta_seed42_best_metadata"]


def test_verify_belief_encoder_metadata_rejects_different_in_repo_encoder_path(tmp_path, monkeypatch):
    _write_matched_encoder_artifacts(tmp_path)
    other_encoder = _write(
        tmp_path / "checkpoints" / "other" / "belief_encoder.pt",
        b"encoder checkpoint",
    )
    _write_json(
        tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "belief_encoder_metadata.json",
        {
            "protocol": "matched_encoder_v1",
            "paired_policy_path": "checkpoints/m7_pta_seed42/best/best_model.zip",
            "paired_policy_sha256": hashlib.sha256(b"policy checkpoint").hexdigest(),
            "belief_encoder_path": "checkpoints/other/belief_encoder.pt",
            "belief_encoder_sha256": hashlib.sha256(other_encoder.read_bytes()).hexdigest(),
        },
    )
    _write_matched_policy_metadata(tmp_path)
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)
    monkeypatch.setattr(registry, "_load_torch", lambda: FakeTorch)
    monkeypatch.setattr(
        registry, "_load_latent_belief_encoder", lambda: FakeLatentBeliefEncoder, raising=False
    )

    manifest = registry.verify_artifacts(tmp_path, ["g2-matched-encoder"])

    rows = {row["logical_name"]: row for row in manifest["artifacts"]}
    assert rows["m7_pta_seed42_belief_encoder_metadata"]["load_status"] == "failed"
    assert registry.failed_required_loads(manifest) == ["m7_pta_seed42_belief_encoder_metadata"]


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


def test_verify_fails_when_timesteps_below_requirement(tmp_path, monkeypatch):
    _write(tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip")

    class ShortModel:
        num_timesteps = 8192

    class ShortPPO:
        @staticmethod
        def load(path: str, device: str = "auto"):
            return ShortModel()

    monkeypatch.setattr(registry, "_load_ppo", lambda: ShortPPO)

    manifest = registry.verify_artifacts(tmp_path, ["presub-g2"])

    row = manifest["artifacts"][0]
    assert row["load_status"] == "failed"
    assert "num_timesteps" in row["load_error"]
    assert registry.failed_required_loads(manifest) == ["m7_pta_seed42_best"]


def test_verify_rejects_symlink_checkpoint_source(tmp_path, monkeypatch):
    outside = _write(tmp_path / "outside.zip", b"external checkpoint")
    checkpoint = tmp_path / "repo" / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip"
    checkpoint.parent.mkdir(parents=True)
    try:
        checkpoint.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")

    class UnexpectedPPO:
        loaded_paths = []

        @staticmethod
        def load(path: str, device: str = "auto"):
            UnexpectedPPO.loaded_paths.append((path, device))
            raise AssertionError("PPO.load should not be called for invalid checkpoint source")

    monkeypatch.setattr(registry, "_load_ppo", lambda: UnexpectedPPO)

    manifest = registry.verify_artifacts(tmp_path / "repo", ["presub-g2"])

    row = manifest["artifacts"][0]
    assert row["load_status"] == "failed"
    assert "symlink" in row["load_error"] or "repo root" in row["load_error"]
    assert row["sha256"] is None
    assert UnexpectedPPO.loaded_paths == []
    assert registry.failed_required_loads(manifest) == ["m7_pta_seed42_best"]


def test_verify_import_failure_marks_present_artifact_failed(tmp_path, monkeypatch):
    _write(tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip")

    def broken_loader():
        raise ImportError("missing sb3")

    monkeypatch.setattr(registry, "_load_ppo", broken_loader)

    manifest = registry.verify_artifacts(tmp_path, ["presub-g2"])

    row = manifest["artifacts"][0]
    assert row["load_status"] == "failed"
    assert row["load_error"] == "missing sb3"
    assert registry.failed_required_loads(manifest) == ["m7_pta_seed42_best"]


def test_verify_missing_required_exits_nonzero_without_loading_sb3(tmp_path):
    manifest_path = tmp_path / "verify.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "verify",
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
    assert manifest_path.exists()
    assert "missing required artifacts: m7_pta_seed42_best" in result.stdout


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


def test_register_run_copies_matched_encoder_sidecars_into_run_dir_and_bundle(tmp_path, monkeypatch):
    _write_matched_encoder_artifacts(tmp_path / "repo")
    FakePPO.loaded_paths.clear()
    FakeTorch.loaded_paths.clear()
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)
    monkeypatch.setattr(registry, "_load_torch", lambda: FakeTorch)
    monkeypatch.setattr(
        registry, "_load_latent_belief_encoder", lambda: FakeLatentBeliefEncoder, raising=False
    )

    manifest = registry.register_run(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifact-root",
        run_id="20260501_matched_encoder",
        origin="local",
        requirements=["g2-matched-encoder"],
        command="python train_m7.py --seed 42",
    )

    run_dir = Path(manifest["run_dir"])
    expected_paths = [
        "checkpoints/m7_pta_seed42/best/best_model.zip",
        "checkpoints/m7_pta_seed42/best/best_model.json",
        "checkpoints/m7_pta_seed42/best/belief_encoder.pt",
        "checkpoints/m7_pta_seed42/best/belief_encoder_metadata.json",
    ]
    for relative_path in expected_paths:
        assert (run_dir / relative_path).is_file()
    assert [row["relative_path"] for row in manifest["artifacts"]] == expected_paths

    archive = registry.bundle_run(run_dir)

    with tarfile.open(archive, "r:gz") as bundle:
        names = set(bundle.getnames())
    for relative_path in expected_paths:
        assert relative_path in names

    restore_root = tmp_path / "restore"
    registry.restore_bundle(archive, restore_root)

    for relative_path in expected_paths:
        assert (restore_root / relative_path).is_file()


def test_register_run_accepts_explicit_artifact_path(tmp_path, monkeypatch):
    _write(
        tmp_path / "repo" / "checkpoints" / "legacy_seed0" / "best" / "best_model.zip",
        b"legacy checkpoint",
    )
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)

    manifest = registry.register_run(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifact-root",
        run_id="20260501_legacy",
        origin="local",
        requirements=[],
        artifact_paths=["checkpoints/legacy_seed0/best/best_model.zip"],
        command="archive legacy checkpoints",
    )

    run_dir = tmp_path / "artifact-root" / "20260501" / "20260501_legacy"
    copied = run_dir / "checkpoints" / "legacy_seed0" / "best" / "best_model.zip"
    row = manifest["artifacts"][0]
    assert copied.read_bytes() == b"legacy checkpoint"
    assert row["relative_path"] == "checkpoints/legacy_seed0/best/best_model.zip"
    assert row["load_status"] == "loaded"
    assert row["storage_path"] == str(copied.resolve())


def test_register_run_rejects_unsafe_run_id(tmp_path, monkeypatch):
    _write(
        tmp_path / "repo" / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip",
        b"durable checkpoint",
    )
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)

    for unsafe_run_id in ("../outside", str(tmp_path / "absolute")):
        with pytest.raises(ValueError):
            registry.register_run(
                repo_root=tmp_path / "repo",
                artifact_root=tmp_path / "artifact-root",
                run_id=unsafe_run_id,
                origin="recovered_by_retraining",
                requirements=["presub-g2"],
                command="python train_m7.py --seed 42",
            )

    assert not (tmp_path / "outside").exists()
    assert not (tmp_path / "absolute").exists()


def test_register_run_copies_related_logs_and_result_files(tmp_path, monkeypatch):
    _write(
        tmp_path / "repo" / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip",
        b"durable checkpoint",
    )
    _write(tmp_path / "repo" / "logs" / "m7_pta_seed42" / "train.csv", b"step,reward\n")
    _write(tmp_path / "repo" / "results" / "presub" / "audit.json", b'{"ok": true}\n')
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)

    manifest = registry.register_run(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifact-root",
        run_id="20260501_results",
        origin="recovered_by_retraining",
        requirements=["presub-g2"],
        command="python train_m7.py --seed 42",
        include_paths=["results/presub/audit.json"],
    )

    run_dir = Path(manifest["run_dir"])
    assert (run_dir / "logs" / "m7_pta_seed42" / "train.csv").read_bytes() == b"step,reward\n"
    result_file = manifest["result_files"][0]
    assert result_file["relative_path"] == "results/presub/audit.json"
    assert result_file["storage_path"] == str((run_dir / "results" / "presub" / "audit.json").resolve())
    assert result_file["size_bytes"] == len(b'{"ok": true}\n')
    assert result_file["sha256"] == hashlib.sha256(b'{"ok": true}\n').hexdigest()


def test_register_run_rejects_symlink_include_path(tmp_path, monkeypatch):
    _write(
        tmp_path / "repo" / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip",
        b"durable checkpoint",
    )
    outside = _write(tmp_path / "outside.json", b'{"secret": true}\n')
    (tmp_path / "repo" / "results").mkdir(parents=True)
    (tmp_path / "repo" / "results" / "link.json").symlink_to(outside)
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)

    with pytest.raises(ValueError):
        registry.register_run(
            repo_root=tmp_path / "repo",
            artifact_root=tmp_path / "artifact-root",
            run_id="20260501_symlink",
            origin="recovered_by_retraining",
            requirements=["presub-g2"],
            command="python train_m7.py --seed 42",
            include_paths=["results/link.json"],
        )

    run_dir = tmp_path / "artifact-root" / "20260501" / "20260501_symlink"
    assert not run_dir.exists()


def test_register_run_rejects_symlinked_inferred_log_dir(tmp_path, monkeypatch):
    _write(
        tmp_path / "repo" / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip",
        b"durable checkpoint",
    )
    outside_logs = tmp_path / "outside-logs" / "m7_pta_seed42"
    _write(outside_logs / "train.csv", b"step,reward\n")
    (tmp_path / "repo" / "logs").mkdir(parents=True)
    try:
        (tmp_path / "repo" / "logs" / "m7_pta_seed42").symlink_to(outside_logs, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unsupported: {exc}")
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)

    with pytest.raises(ValueError, match="symlink"):
        registry.register_run(
            repo_root=tmp_path / "repo",
            artifact_root=tmp_path / "artifact-root",
            run_id="20260501_log_symlink",
            origin="recovered_by_retraining",
            requirements=["presub-g2"],
            command="python train_m7.py --seed 42",
        )

    run_dir = tmp_path / "artifact-root" / "20260501" / "20260501_log_symlink"
    assert not run_dir.exists()


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


def test_restore_bundle_rejects_manifest_hash_mismatch_before_writing_target(tmp_path):
    archive = tmp_path / "corrupt.tar.gz"
    good = b"good checkpoint"
    corrupt = b"corrupt checkpoint"
    manifest = {
        "artifacts": [
            {
                "logical_name": "m7_pta_seed42_best",
                "relative_path": "checkpoints/m7_pta_seed42/best/best_model.zip",
                "storage_path": "/artifact/checkpoints/m7_pta_seed42/best/best_model.zip",
                "size_bytes": len(good),
                "sha256": hashlib.sha256(good).hexdigest(),
                "required": True,
                "exists": True,
                "load_status": "loaded",
            }
        ],
        "result_files": [],
    }
    manifest_bytes = json.dumps(manifest).encode("utf-8")
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("artifact_manifest.json")
        info.size = len(manifest_bytes)
        bundle.addfile(info, io.BytesIO(manifest_bytes))
        info = tarfile.TarInfo("checkpoints/m7_pta_seed42/best/best_model.zip")
        info.size = len(corrupt)
        bundle.addfile(info, io.BytesIO(corrupt))

    restore_root = tmp_path / "restore"
    with pytest.raises(RuntimeError, match="manifest|mismatch"):
        registry.restore_bundle(archive, restore_root)

    assert not (restore_root / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip").exists()


def test_restore_bundle_rejects_non_empty_target(tmp_path, monkeypatch):
    _write(
        tmp_path / "repo" / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip",
        b"bundle checkpoint",
    )
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)
    manifest = registry.register_run(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifact-root",
        run_id="20260501_bundle_nonempty",
        origin="local",
        requirements=["presub-g2"],
        command="train",
    )
    archive = registry.bundle_run(Path(manifest["run_dir"]))
    restore_root = tmp_path / "restore"
    _write(restore_root / "checkpoints" / "stale.zip", b"stale")

    with pytest.raises(FileExistsError, match="restore target must be empty"):
        registry.restore_bundle(archive, restore_root)

    assert (restore_root / "checkpoints" / "stale.zip").read_bytes() == b"stale"


def test_bundle_run_revalidates_manifest_hashes(tmp_path, monkeypatch):
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
    (run_dir / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip").write_bytes(b"tampered")

    with pytest.raises(RuntimeError, match="mismatch"):
        registry.bundle_run(run_dir)


def test_bundle_run_requires_required_artifact_manifest_fields(tmp_path, monkeypatch):
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
    manifest_path = run_dir / "artifact_manifest.json"
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    del saved["artifacts"][0]["storage_path"]
    manifest_path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises((RuntimeError, ValueError), match="storage_path|manifest"):
        registry.bundle_run(run_dir)


def test_bundle_run_rejects_unsafe_manifest_relative_path(tmp_path, monkeypatch):
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
    manifest_path = run_dir / "artifact_manifest.json"
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    saved["artifacts"][0]["relative_path"] = "../outside.zip"
    manifest_path.write_text(json.dumps(saved), encoding="utf-8")

    with pytest.raises((RuntimeError, ValueError)):
        registry.bundle_run(run_dir)


def test_restore_bundle_rejects_path_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("../evil.txt")
        info.size = 0
        bundle.addfile(info)

    with pytest.raises(ValueError, match="unsafe archive member"):
        registry.restore_bundle(archive, tmp_path / "restore")

    assert not (tmp_path / "evil.txt").exists()


def test_restore_bundle_rejects_normalized_traversal(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        payload = b"not a registry artifact"
        info = tarfile.TarInfo("checkpoints/../tools/artifact_registry.py")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe archive member"):
        registry.restore_bundle(archive, tmp_path / "restore")

    assert not (tmp_path / "restore" / "tools" / "artifact_registry.py").exists()


def test_restore_bundle_rejects_unexpected_in_root_path(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        payload = b"not a registry artifact"
        info = tarfile.TarInfo("tools/artifact_registry.py")
        info.size = len(payload)
        bundle.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe archive member"):
        registry.restore_bundle(archive, tmp_path / "restore")

    assert not (tmp_path / "restore" / "tools" / "artifact_registry.py").exists()


def test_restore_bundle_rejects_symlink_member(tmp_path):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as bundle:
        info = tarfile.TarInfo("checkpoints/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "artifact_manifest.json"
        bundle.addfile(info)

    with pytest.raises(ValueError, match="unsafe archive member"):
        registry.restore_bundle(archive, tmp_path / "restore")


def test_bundle_run_rejects_output_inside_archived_directory(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)

    with pytest.raises(ValueError, match="output cannot be inside archived directory"):
        registry.bundle_run(run_dir, run_dir / "checkpoints" / "bundle.tar.gz")


def test_register_run_rejects_duplicate_run_id(tmp_path, monkeypatch):
    _write(
        tmp_path / "repo" / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip",
        b"durable checkpoint",
    )
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)

    registry.register_run(
        repo_root=tmp_path / "repo",
        artifact_root=tmp_path / "artifact-root",
        run_id="20260501_recovery",
        origin="recovered_by_retraining",
        requirements=["presub-g2"],
        command="first command",
    )

    with pytest.raises(FileExistsError):
        registry.register_run(
            repo_root=tmp_path / "repo",
            artifact_root=tmp_path / "artifact-root",
            run_id="20260501_recovery",
            origin="recovered_by_retraining",
            requirements=["presub-g2"],
            command="second command",
        )

    run_dir = tmp_path / "artifact-root" / "20260501" / "20260501_recovery"
    assert (run_dir / "command.txt").read_text(encoding="utf-8") == "first command\n"


def test_register_run_missing_artifact_leaves_no_run_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        registry.register_run(
            repo_root=tmp_path / "repo",
            artifact_root=tmp_path / "artifact-root",
            run_id="20260501_recovery",
            origin="recovered_by_retraining",
            requirements=["presub-g2"],
            command="python train_m7.py --seed 42",
        )

    run_dir = tmp_path / "artifact-root" / "20260501" / "20260501_recovery"
    assert not run_dir.exists()


def test_selected_env_uses_allowlist(monkeypatch):
    monkeypatch.setenv("PYOPENGL_PLATFORM", "egl")
    monkeypatch.setenv("SECRET_TOKEN", "do-not-save")

    env = registry.selected_env()

    assert env["PYOPENGL_PLATFORM"] == "egl"
    assert "SECRET_TOKEN" not in env


def test_register_run_copy_mismatch_cleans_up(tmp_path, monkeypatch):
    _write(
        tmp_path / "repo" / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip",
        b"durable checkpoint",
    )
    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)

    def corrupt_copy(source, destination):
        destination.write_bytes(b"corrupt")
        return destination

    monkeypatch.setattr(registry.shutil, "copy2", corrupt_copy)

    with pytest.raises(RuntimeError):
        registry.register_run(
            repo_root=tmp_path / "repo",
            artifact_root=tmp_path / "artifact-root",
            run_id="20260501_recovery",
            origin="recovered_by_retraining",
            requirements=["presub-g2"],
            command="python train_m7.py --seed 42",
        )

    date_dir = tmp_path / "artifact-root" / "20260501"
    run_dir = date_dir / "20260501_recovery"
    assert not run_dir.exists()
    assert not list(date_dir.glob("*.tmp"))


def test_register_run_cli_missing_artifact_exits_nonzero_without_traceback(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "register-run",
            "--repo-root",
            str(tmp_path / "repo"),
            "--requirement",
            "presub-g2",
            "--artifact-root",
            str(tmp_path / "artifact-root"),
            "--run-id",
            "20260501_recovery",
            "--origin",
            "recovered_by_retraining",
            "--command",
            "python train_m7.py --seed 42",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "missing required artifacts: m7_pta_seed42_best" in result.stderr
    assert "missing required artifacts: m7_pta_seed42_best" not in result.stdout
