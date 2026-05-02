import json
from pathlib import Path

import pytest
import torch

from pta.models.belief.latent_belief_encoder import LatentBeliefEncoder
from pta.training.utils import checkpoint_io


def _write_policy(path: Path, payload: bytes = b"policy") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.with_suffix(".json").write_text("{}\n", encoding="utf-8")
    return path


def _encoder(fill: float = 0.25) -> LatentBeliefEncoder:
    encoder = LatentBeliefEncoder(
        trace_dim=30,
        latent_dim=16,
        hidden_dim=128,
        num_layers=2,
    )
    for parameter in encoder.parameters():
        torch.nn.init.constant_(parameter, fill)
    return encoder


def test_encoder_sidecar_paths_resolve_final_and_best_policy_paths(tmp_path):
    final_policy = tmp_path / "checkpoints" / "m7_pta_seed42" / "m7_pta_final.zip"
    best_policy = tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip"

    final_paths = checkpoint_io.m7_encoder_sidecar_paths(final_policy)
    best_paths = checkpoint_io.m7_encoder_sidecar_paths(best_policy)

    assert final_paths.encoder_path == final_policy.parent / "belief_encoder.pt"
    assert final_paths.metadata_path == final_policy.parent / "belief_encoder_metadata.json"
    assert final_paths.policy_metadata_path == final_policy.with_suffix(".json")
    assert best_paths.encoder_path == best_policy.parent / "belief_encoder.pt"
    assert best_paths.metadata_path == best_policy.parent / "belief_encoder_metadata.json"
    assert best_paths.policy_metadata_path == best_policy.with_suffix(".json")


def test_save_m7_encoder_artifact_writes_state_metadata_and_hash_links(tmp_path):
    policy_path = _write_policy(
        tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip"
    )
    encoder = _encoder(fill=0.5)

    metadata = checkpoint_io.save_m7_encoder_artifact(
        encoder=encoder,
        policy_path=policy_path,
        repo_root=tmp_path,
        run_metadata={
            "method": "m7_pta",
            "seed": 42,
            "ablation": "none",
            "trace_dim": 30,
            "latent_dim": 16,
            "hidden_dim": 128,
            "num_layers": 2,
            "n_probes": 3,
            "stage": "best",
        },
    )

    paths = checkpoint_io.m7_encoder_sidecar_paths(policy_path)
    payload = torch.load(paths.encoder_path, map_location="cpu", weights_only=False)
    saved_metadata = json.loads(paths.metadata_path.read_text(encoding="utf-8"))
    policy_metadata = json.loads(paths.policy_metadata_path.read_text(encoding="utf-8"))

    assert payload["format_version"] == checkpoint_io.M7_ENCODER_FORMAT_VERSION
    assert payload["config"] == {
        "trace_dim": 30,
        "latent_dim": 16,
        "hidden_dim": 128,
        "num_layers": 2,
    }
    assert set(payload["state_dict"]) == set(encoder.state_dict())
    assert saved_metadata["protocol"] == "matched_encoder_v1"
    assert saved_metadata["paired_policy_path"] == (
        "checkpoints/m7_pta_seed42/best/best_model.zip"
    )
    assert saved_metadata["paired_policy_sha256"] == checkpoint_io.sha256_file(policy_path)
    assert saved_metadata["belief_encoder_sha256"] == checkpoint_io.sha256_file(
        paths.encoder_path
    )
    assert policy_metadata["protocol"] == "matched_encoder_v1"
    assert policy_metadata["encoder_mode"] == "matched"
    assert policy_metadata["legacy_policy_only"] is False
    assert policy_metadata["belief_encoder_path"] == saved_metadata["belief_encoder_path"]
    assert policy_metadata["belief_encoder_sha256"] == saved_metadata[
        "belief_encoder_sha256"
    ]
    assert policy_metadata["belief_encoder_metadata_path"] == (
        "checkpoints/m7_pta_seed42/best/belief_encoder_metadata.json"
    )
    assert policy_metadata[
        "belief_encoder_metadata_sha256"
    ] == checkpoint_io.sha256_file(paths.metadata_path)
    assert metadata == saved_metadata


def test_load_m7_encoder_artifact_restores_exact_state_dict(tmp_path):
    policy_path = _write_policy(
        tmp_path / "checkpoints" / "m7_pta_seed42" / "m7_pta_final.zip"
    )
    source = _encoder(fill=0.75)
    checkpoint_io.save_m7_encoder_artifact(
        encoder=source,
        policy_path=policy_path,
        repo_root=tmp_path,
        run_metadata={
            "method": "m7_pta",
            "seed": 42,
            "ablation": "none",
            "trace_dim": 30,
            "latent_dim": 16,
            "hidden_dim": 128,
            "num_layers": 2,
            "n_probes": 3,
            "stage": "final",
        },
    )

    loaded, metadata = checkpoint_io.load_m7_encoder_artifact(
        policy_path,
        expected={
            "method": "m7_pta",
            "seed": 42,
            "ablation": "none",
            "latent_dim": 16,
            "n_probes": 3,
        },
    )

    assert metadata["stage"] == "final"
    for key, value in source.state_dict().items():
        assert torch.equal(value, loaded.state_dict()[key])


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("method", "m7_other"),
        ("seed", 7),
        ("ablation", "no_probe"),
        ("latent_dim", 8),
        ("n_probes", 9),
    ],
)
def test_load_m7_encoder_artifact_rejects_metadata_mismatch(tmp_path, field, bad_value):
    policy_path = _write_policy(
        tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip"
    )
    checkpoint_io.save_m7_encoder_artifact(
        encoder=_encoder(fill=0.1),
        policy_path=policy_path,
        repo_root=tmp_path,
        run_metadata={
            "method": "m7_pta",
            "seed": 42,
            "ablation": "none",
            "trace_dim": 30,
            "latent_dim": 16,
            "hidden_dim": 128,
            "num_layers": 2,
            "n_probes": 3,
            "stage": "best",
        },
    )

    expected = {
        "method": "m7_pta",
        "seed": 42,
        "ablation": "none",
        "latent_dim": 16,
        "n_probes": 3,
    }
    expected[field] = bad_value
    with pytest.raises(ValueError, match=field):
        checkpoint_io.load_m7_encoder_artifact(policy_path, expected=expected)


def test_load_m7_encoder_artifact_rejects_missing_sidecar(tmp_path):
    policy_path = _write_policy(
        tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip"
    )

    with pytest.raises(FileNotFoundError, match="matched encoder"):
        checkpoint_io.load_m7_encoder_artifact(
            policy_path,
            expected={"method": "m7_pta"},
        )
