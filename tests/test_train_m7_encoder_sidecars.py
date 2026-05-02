import json
import sys
import types
from argparse import Namespace
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


def test_main_derives_trace_dim_from_m7_inner_observation_space(tmp_path, monkeypatch):
    observed_encoders = []
    inner_trace_dim = 47

    class FakeSpace:
        def __init__(self, shape):
            self.shape = shape

    class FakeGenesisGymWrapper:
        def __init__(self, task_config=None, scene_config=None):
            self.observation_space = FakeSpace((11,))
            self.action_space = FakeSpace((2,))

    class FakeJointResidualWrapper:
        def __init__(self, env, residual_scale, trajectory):
            self.env = env
            self.observation_space = FakeSpace((inner_trace_dim,))
            self.action_space = env.action_space

        def reset(self, seed=None, options=None):
            return torch.zeros(inner_trace_dim).numpy(), {}

    class FakeProbePhaseWrapper:
        def __init__(
            self,
            env,
            latent_dim,
            n_probes,
            belief_encoder,
            ablation,
            device,
        ):
            observed_encoders.append(belief_encoder)
            self.env = env
            self.observation_space = FakeSpace((inner_trace_dim + latent_dim,))
            self.action_space = env.action_space

        def reset(self, seed=None, options=None):
            return torch.zeros(self.observation_space.shape[0]).numpy(), {}

        def close(self):
            pass

    class FakePPO(FakeModel):
        def __init__(self, *args, **kwargs):
            super().__init__()

        def learn(self, total_timesteps, callback, progress_bar):
            callback.init_callback(self)
            eval_callback = callback.callbacks[1]
            best_policy = Path(eval_callback.best_model_save_path) / "best_model.zip"
            best_policy.parent.mkdir(parents=True, exist_ok=True)
            best_policy.write_bytes(b"best-policy")
            eval_callback.callback_on_new_best.on_step()
            return self

    class FakeCheckpointCallback:
        def __init__(self, *args, **kwargs):
            pass

        def init_callback(self, model):
            self.model = model

    class FakeEvalCallback(FakeCheckpointCallback):
        def __init__(self, *args, **kwargs):
            self.best_model_save_path = kwargs["best_model_save_path"]
            self.callback_on_new_best = kwargs["callback_on_new_best"]

        def init_callback(self, model):
            super().init_callback(model)
            self.callback_on_new_best.init_callback(model)

    class FakeCallbackList:
        def __init__(self, callbacks):
            self.callbacks = callbacks

        def init_callback(self, model):
            for callback in self.callbacks:
                callback.init_callback(model)

    class FakeDummyVecEnv:
        def __init__(self, env_fns):
            self.envs = [env_fn() for env_fn in env_fns]

        def close(self):
            for env in self.envs:
                if hasattr(env, "close"):
                    env.close()

    class FakeExperimentLogger:
        def __init__(self, *args, **kwargs):
            pass

        def log_config(self, config):
            self.config = config

        def close(self):
            pass

    monkeypatch.setattr(train_m7, "_PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        train_m7,
        "parse_args",
        lambda: Namespace(
            seed=42,
            total_timesteps=1,
            material="sand",
            residual_scale=0.05,
            horizon=500,
            latent_dim=16,
            n_probes=3,
            eval_freq=1,
            ablation="none",
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "pta.envs.wrappers.gym_wrapper",
        types.SimpleNamespace(GenesisGymWrapper=FakeGenesisGymWrapper),
    )
    monkeypatch.setitem(
        sys.modules,
        "pta.envs.wrappers.joint_residual_wrapper",
        types.SimpleNamespace(JointResidualWrapper=FakeJointResidualWrapper),
    )
    monkeypatch.setitem(
        sys.modules,
        "pta.envs.wrappers.probe_phase_wrapper",
        types.SimpleNamespace(ProbePhaseWrapper=FakeProbePhaseWrapper),
    )
    monkeypatch.setitem(
        sys.modules,
        "stable_baselines3",
        types.SimpleNamespace(PPO=FakePPO),
    )
    monkeypatch.setitem(
        sys.modules,
        "stable_baselines3.common.callbacks",
        types.SimpleNamespace(
            CallbackList=FakeCallbackList,
            CheckpointCallback=FakeCheckpointCallback,
            EvalCallback=FakeEvalCallback,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "stable_baselines3.common.vec_env",
        types.SimpleNamespace(DummyVecEnv=FakeDummyVecEnv),
    )
    monkeypatch.setitem(
        sys.modules,
        "pta.training.utils.logger",
        types.SimpleNamespace(ExperimentLogger=FakeExperimentLogger),
    )
    monkeypatch.setitem(
        sys.modules,
        "pta.training.utils.seed",
        types.SimpleNamespace(set_seed=lambda seed: None),
    )

    train_m7.main()

    assert [encoder.trace_dim for encoder in observed_encoders] == [inner_trace_dim] * 2

    final_policy = tmp_path / "checkpoints" / "m7_pta_seed42" / "m7_pta_final.zip"
    final_paths = checkpoint_io.m7_encoder_sidecar_paths(final_policy)
    best_policy = tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip"
    best_paths = checkpoint_io.m7_encoder_sidecar_paths(best_policy)

    final_metadata = json.loads(final_paths.metadata_path.read_text())
    best_metadata = json.loads(best_paths.metadata_path.read_text())
    final_payload = torch.load(
        final_paths.encoder_path,
        map_location="cpu",
        weights_only=False,
    )
    best_payload = torch.load(
        best_paths.encoder_path,
        map_location="cpu",
        weights_only=False,
    )

    assert final_metadata["trace_dim"] == inner_trace_dim
    assert best_metadata["trace_dim"] == inner_trace_dim
    assert final_payload["config"]["trace_dim"] == inner_trace_dim
    assert best_payload["config"]["trace_dim"] == inner_trace_dim
