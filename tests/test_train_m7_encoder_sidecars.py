import json
from pathlib import Path

import torch

from pta.models.belief.latent_belief_encoder import LatentBeliefEncoder
from pta.scripts import train_m7
from pta.training.utils import checkpoint_io


class FakeModel:
    def __init__(self):
        self.saved_paths = []

    def save(self, path: str) -> None:
        self.saved_paths.append(path)
        zip_path = Path(path).with_suffix(".zip")
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_bytes(b"fake-policy")


def _encoder(fill: float = 0.25) -> LatentBeliefEncoder:
    encoder = LatentBeliefEncoder(
        trace_dim=30,
        latent_dim=16,
        hidden_dim=train_m7.ENCODER_HIDDEN_DIM,
        num_layers=train_m7.ENCODER_NUM_LAYERS,
    )
    for parameter in encoder.parameters():
        torch.nn.init.constant_(parameter, fill)
    return encoder


def _metadata(stage: str = "final") -> dict:
    return {
        "method": "m7_pta",
        "seed": 42,
        "ablation": "none",
        "trace_dim": 30,
        "latent_dim": 16,
        "hidden_dim": train_m7.ENCODER_HIDDEN_DIM,
        "num_layers": train_m7.ENCODER_NUM_LAYERS,
        "n_probes": 3,
        "stage": stage,
    }


def test_clone_belief_encoder_state_returns_independent_identical_eval_module():
    source = _encoder(fill=0.5)
    source.train()

    clone = train_m7.clone_belief_encoder_state(source)

    assert clone is not source
    assert clone.training is False
    for name, tensor in source.state_dict().items():
        cloned_tensor = clone.state_dict()[name]
        assert torch.equal(cloned_tensor, tensor)
        assert cloned_tensor.data_ptr() != tensor.data_ptr()

    first_parameter = next(clone.parameters())
    first_parameter.data.add_(1.0)
    assert not torch.equal(
        first_parameter,
        next(source.parameters()),
    )


def test_save_m7_policy_with_encoder_writes_policy_and_matched_sidecars(tmp_path):
    model = FakeModel()
    encoder = _encoder(fill=0.75)
    checkpoint_path = tmp_path / "checkpoints" / "m7_pta_seed42" / "m7_pta_final"

    train_m7.save_m7_policy_with_encoder(
        model,
        encoder,
        checkpoint_path,
        repo_root=tmp_path,
        metadata=_metadata(stage="final"),
    )

    policy_path = checkpoint_path.with_suffix(".zip")
    paths = checkpoint_io.m7_encoder_sidecar_paths(policy_path)
    assert policy_path.exists()
    assert policy_path.with_suffix(".json").exists()
    assert paths.encoder_path.exists()
    assert paths.metadata_path.exists()

    policy_metadata = json.loads(policy_path.with_suffix(".json").read_text())
    encoder_metadata = json.loads(paths.metadata_path.read_text())
    payload = torch.load(paths.encoder_path, map_location="cpu", weights_only=False)

    assert policy_metadata["encoder_mode"] == "matched"
    assert policy_metadata["legacy_policy_only"] is False
    assert encoder_metadata["stage"] == "final"
    assert encoder_metadata["paired_policy_path"] == (
        "checkpoints/m7_pta_seed42/m7_pta_final.zip"
    )
    assert payload["config"] == {
        "trace_dim": 30,
        "latent_dim": 16,
        "hidden_dim": train_m7.ENCODER_HIDDEN_DIM,
        "num_layers": train_m7.ENCODER_NUM_LAYERS,
    }
    for key, tensor in encoder.state_dict().items():
        assert torch.equal(payload["state_dict"][key], tensor)


def test_best_checkpoint_sidecar_callback_writes_sidecars_for_new_best(tmp_path):
    model = FakeModel()
    encoder = _encoder(fill=0.33)
    best_dir = tmp_path / "checkpoints" / "m7_pta_seed42" / "best"
    best_dir.mkdir(parents=True)
    best_policy = best_dir / "best_model.zip"
    best_policy.write_bytes(b"new-best-policy")

    callback = train_m7.M7BestModelSidecarCallback(
        encoder=encoder,
        policy_path=best_policy,
        repo_root=tmp_path,
        metadata=_metadata(stage="best"),
    )
    callback.init_callback(model)

    assert callback.on_step() is True

    paths = checkpoint_io.m7_encoder_sidecar_paths(best_policy)
    assert best_policy.exists()
    assert best_policy.with_suffix(".json").exists()
    assert paths.encoder_path.exists()
    assert paths.metadata_path.exists()
    assert json.loads(paths.metadata_path.read_text())["stage"] == "best"
