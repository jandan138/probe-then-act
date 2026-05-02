# M7 Matched-Encoder Checkpoint Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist and reload the matched M7 belief encoder so full `m7_pta` evaluation no longer pairs policy checkpoints with fresh random encoders by default.

**Architecture:** Add encoder sidecar save/load helpers in `pta/training/utils/checkpoint_io.py`, use them from M7 training and evaluation, split matched evaluation from random-stress diagnostics, and extend the artifact registry to validate policy/encoder bundles by artifact kind. Keep legacy policy-only runs explicit and non-claim-bearing.

**Tech Stack:** Python 3.10, PyTorch, Stable-Baselines3 PPO, Gymnasium wrappers, pytest, CPFS artifact registry, DLC shell launcher.

---

## File Structure

- Create `tests/test_m7_encoder_artifacts.py`: lightweight TDD for encoder sidecar format, metadata validation, and exact state restoration.
- Modify `pta/training/utils/checkpoint_io.py`: add sidecar path resolution, SHA256 helpers, encoder metadata validation, encoder sidecar save/load, and policy metadata update helpers.
- Modify `pta/scripts/train_m7.py`: construct canonical frozen encoder state, pass matched state into train/eval envs, and save matched sidecars for `best` and `final` full `m7_pta` checkpoints.
- Modify `tests/test_run_ood_eval_v2.py`: prove matched mode is default, policy-only full M7 fails, random-stress is explicit, and ablations remain zero-z.
- Modify `pta/scripts/run_ood_eval_v2.py`: add encoder mode CLI, sidecar resolution, result schema identity fields, and matched/random-stress env construction.
- Modify `tests/test_pre_submission_audit.py`: separate random-eval-encoder stress audit from matched-encoder audit.
- Modify `tools/pre_submission_audit.py`: drive random-stress explicitly and add matched audit output labeling.
- Modify `tests/test_artifact_registry.py`: add matched-M7 requirement set and kind-specific validation tests.
- Modify `tools/artifact_registry.py`: add `g2-matched-encoder`, type-specific loading, and encoder sidecar registration/restore validation.
- Modify `docs/30_records/SEED_SENSITIVITY_STATUS.md` and relevant runbooks: label legacy policy-only runs and the new matched protocol.

## Task 1: Encoder Sidecar Helpers

**Files:**
- Create: `tests/test_m7_encoder_artifacts.py`
- Modify: `pta/training/utils/checkpoint_io.py`

- [ ] **Step 1: Add failing encoder sidecar tests**

Create `tests/test_m7_encoder_artifacts.py` with:

```python
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
    encoder = LatentBeliefEncoder(trace_dim=30, latent_dim=16, hidden_dim=128, num_layers=2)
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
    policy_path = _write_policy(tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip")
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
    assert set(payload["state_dict"]) == set(encoder.state_dict())
    assert saved_metadata["protocol"] == "matched_encoder_v1"
    assert saved_metadata["paired_policy_path"] == "checkpoints/m7_pta_seed42/best/best_model.zip"
    assert saved_metadata["paired_policy_sha256"] == checkpoint_io.sha256_file(policy_path)
    assert saved_metadata["belief_encoder_sha256"] == checkpoint_io.sha256_file(paths.encoder_path)
    assert policy_metadata["encoder_mode"] == "matched"
    assert policy_metadata["legacy_policy_only"] is False
    assert policy_metadata["belief_encoder_sha256"] == saved_metadata["belief_encoder_sha256"]
    assert metadata == saved_metadata


def test_load_m7_encoder_artifact_restores_exact_state_dict(tmp_path):
    policy_path = _write_policy(tmp_path / "checkpoints" / "m7_pta_seed42" / "m7_pta_final.zip")
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
        expected={"method": "m7_pta", "seed": 42, "ablation": "none", "latent_dim": 16, "n_probes": 3},
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
    policy_path = _write_policy(tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip")
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

    expected = {"method": "m7_pta", "seed": 42, "ablation": "none", "latent_dim": 16, "n_probes": 3}
    expected[field] = bad_value
    with pytest.raises(ValueError, match=field):
        checkpoint_io.load_m7_encoder_artifact(policy_path, expected=expected)


def test_load_m7_encoder_artifact_rejects_missing_sidecar(tmp_path):
    policy_path = _write_policy(tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip")

    with pytest.raises(FileNotFoundError, match="matched encoder"):
        checkpoint_io.load_m7_encoder_artifact(policy_path, expected={"method": "m7_pta"})
```

- [ ] **Step 2: Run the new tests and verify RED**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_m7_encoder_artifacts.py -q`

Expected: FAIL with missing attributes such as `m7_encoder_sidecar_paths`, `save_m7_encoder_artifact`, or `load_m7_encoder_artifact`.

- [ ] **Step 3: Add encoder sidecar helpers**

Append these imports and helpers to `pta/training/utils/checkpoint_io.py` while keeping existing SB3 helpers intact:

```python
from dataclasses import dataclass


M7_ENCODER_PROTOCOL = "matched_encoder_v1"
M7_ENCODER_FORMAT_VERSION = 1


@dataclass(frozen=True)
class M7EncoderSidecarPaths:
    policy_path: Path
    policy_metadata_path: Path
    encoder_path: Path
    metadata_path: Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _policy_zip_path(policy_path: Path) -> Path:
    policy_path = Path(policy_path)
    if policy_path.suffix == ".zip":
        return policy_path
    zip_path = policy_path.with_suffix(".zip")
    return zip_path if zip_path.exists() else policy_path


def m7_encoder_sidecar_paths(policy_path: Path) -> M7EncoderSidecarPaths:
    policy_path = _policy_zip_path(policy_path)
    return M7EncoderSidecarPaths(
        policy_path=policy_path,
        policy_metadata_path=policy_path.with_suffix(".json"),
        encoder_path=policy_path.parent / "belief_encoder.pt",
        metadata_path=policy_path.parent / "belief_encoder_metadata.json",
    )


def _relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return Path(path).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path is outside repo root: {path}") from exc


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_m7_encoder_artifact(
    *,
    encoder: Any,
    policy_path: Path,
    repo_root: Path,
    run_metadata: Dict[str, Any],
) -> Dict[str, Any]:
    paths = m7_encoder_sidecar_paths(policy_path)
    if not paths.policy_path.exists():
        raise FileNotFoundError(f"policy checkpoint not found: {paths.policy_path}")
    required = ["method", "seed", "ablation", "trace_dim", "latent_dim", "hidden_dim", "num_layers", "n_probes", "stage"]
    missing = [key for key in required if key not in run_metadata]
    if missing:
        raise ValueError("missing encoder metadata fields: " + ", ".join(missing))

    payload = {
        "format_version": M7_ENCODER_FORMAT_VERSION,
        "state_dict": encoder.state_dict(),
        "config": {
            "trace_dim": int(run_metadata["trace_dim"]),
            "latent_dim": int(run_metadata["latent_dim"]),
            "hidden_dim": int(run_metadata["hidden_dim"]),
            "num_layers": int(run_metadata["num_layers"]),
        },
    }
    paths.encoder_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, paths.encoder_path)

    metadata = {
        "protocol": M7_ENCODER_PROTOCOL,
        **run_metadata,
        "paired_policy_path": _relative_to_repo(paths.policy_path, repo_root),
        "paired_policy_sha256": sha256_file(paths.policy_path),
        "belief_encoder_path": _relative_to_repo(paths.encoder_path, repo_root),
        "belief_encoder_sha256": sha256_file(paths.encoder_path),
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    _write_json(paths.metadata_path, metadata)

    policy_metadata = _read_json(paths.policy_metadata_path)
    policy_metadata.update(
        {
            "protocol": M7_ENCODER_PROTOCOL,
            "encoder_mode": "matched",
            "legacy_policy_only": False,
            "belief_encoder_path": metadata["belief_encoder_path"],
            "belief_encoder_sha256": metadata["belief_encoder_sha256"],
            "belief_encoder_metadata_path": _relative_to_repo(paths.metadata_path, repo_root),
            "belief_encoder_metadata_sha256": sha256_file(paths.metadata_path),
        }
    )
    _write_json(paths.policy_metadata_path, policy_metadata)
    return metadata


def validate_m7_encoder_metadata(metadata: Dict[str, Any], expected: Dict[str, Any]) -> None:
    if metadata.get("protocol") != M7_ENCODER_PROTOCOL:
        raise ValueError("protocol mismatch for matched encoder")
    for field, expected_value in expected.items():
        if expected_value is not None and metadata.get(field) != expected_value:
            raise ValueError(f"{field} mismatch for matched encoder: expected {expected_value!r}, got {metadata.get(field)!r}")


def load_m7_encoder_artifact(
    policy_path: Path,
    *,
    expected: Dict[str, Any],
    map_location: str = "cpu",
) -> tuple[Any, Dict[str, Any]]:
    from pta.models.belief.latent_belief_encoder import LatentBeliefEncoder

    paths = m7_encoder_sidecar_paths(policy_path)
    if not paths.encoder_path.exists() or not paths.metadata_path.exists():
        raise FileNotFoundError(f"missing matched encoder artifact for {paths.policy_path}")
    metadata = _read_json(paths.metadata_path)
    validate_m7_encoder_metadata(metadata, expected)
    if paths.policy_path.exists() and metadata.get("paired_policy_sha256") != sha256_file(paths.policy_path):
        raise ValueError("paired_policy_sha256 mismatch for matched encoder")
    if metadata.get("belief_encoder_sha256") != sha256_file(paths.encoder_path):
        raise ValueError("belief_encoder_sha256 mismatch for matched encoder")

    payload = torch.load(paths.encoder_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != M7_ENCODER_FORMAT_VERSION:
        raise ValueError("unsupported matched encoder artifact format")
    config = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("matched encoder artifact must contain config and state_dict")
    encoder = LatentBeliefEncoder(
        trace_dim=int(config["trace_dim"]),
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
    )
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder, metadata
```

Also add `import hashlib` near the top of `checkpoint_io.py`.

- [ ] **Step 4: Run encoder sidecar tests and verify GREEN**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_m7_encoder_artifacts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

Run:

```bash
git add pta/training/utils/checkpoint_io.py tests/test_m7_encoder_artifacts.py
git commit -m "tools: add M7 encoder sidecar helpers"
```

Expected: commit succeeds.

## Task 2: Matched Encoder Evaluation Contract

**Files:**
- Modify: `tests/test_run_ood_eval_v2.py`
- Modify: `pta/scripts/run_ood_eval_v2.py`

- [ ] **Step 1: Add failing eval contract tests**

Append these tests to `tests/test_run_ood_eval_v2.py`:

```python
from pathlib import Path

import pytest

from pta.scripts import run_ood_eval_v2 as ev


def test_parse_args_defaults_to_matched_encoder_mode():
    args = ev.parse_args([])
    assert args.m7_encoder_mode == "matched"
    assert args.m7_random_encoder_seed is None


def test_result_key_includes_encoder_protocol_identity():
    row = {
        "method": "m7_pta",
        "seed": 42,
        "split": "ood_elastoplastic",
        "encoder_mode": "matched",
        "encoder_seed": "",
        "encoder_sha256": "abc",
        "policy_sha256": "def",
        "protocol": "matched_encoder_v1",
    }
    assert ev.result_key(row) == (
        "m7_pta",
        42,
        "ood_elastoplastic",
        "matched",
        "",
        "abc",
        "def",
        "matched_encoder_v1",
    )


def test_resolve_m7_encoder_rejects_policy_only_full_m7(tmp_path):
    policy = tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(b"policy")

    with pytest.raises(FileNotFoundError, match="missing matched encoder artifact"):
        ev.resolve_m7_belief_encoder(
            policy_path=policy,
            ablation="none",
            encoder_mode="matched",
            encoder_seed=None,
            expected={"method": "m7_pta", "seed": 42, "ablation": "none", "latent_dim": 16, "n_probes": 3},
        )


def test_resolve_m7_encoder_allows_ablation_zero_z_without_sidecar(tmp_path):
    policy = tmp_path / "checkpoints" / "m7_pta_noprobe_seed42" / "m7_pta_final.zip"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(b"policy")

    encoder, identity = ev.resolve_m7_belief_encoder(
        policy_path=policy,
        ablation="no_probe",
        encoder_mode="matched",
        encoder_seed=None,
        expected={"method": "m7_noprobe", "seed": 42, "ablation": "no_probe", "latent_dim": 16, "n_probes": 3},
    )

    assert encoder is None
    assert identity["encoder_mode"] == "zero-z"
    assert identity["protocol"] == "ablation_zero_z"


def test_random_stress_requires_explicit_mode_and_records_seed(tmp_path, monkeypatch):
    policy = tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip"
    policy.parent.mkdir(parents=True)
    policy.write_bytes(b"policy")

    calls = []
    monkeypatch.setattr(ev.random, "seed", lambda value: calls.append(("random", value)))
    monkeypatch.setattr(ev.np.random, "seed", lambda value: calls.append(("numpy", value)))

    encoder, identity = ev.resolve_m7_belief_encoder(
        policy_path=policy,
        ablation="none",
        encoder_mode="random-stress",
        encoder_seed=33,
        expected={"method": "m7_pta", "seed": 42, "ablation": "none", "latent_dim": 16, "n_probes": 3},
    )

    assert encoder is None
    assert identity["encoder_mode"] == "random-stress"
    assert identity["encoder_seed"] == "33"
    assert ("random", 33) in calls
    assert ("numpy", 33) in calls


def test_make_eval_env_passes_loaded_encoder_to_probe_wrapper(monkeypatch):
    captured = {}

    class FakeSpace:
        shape = (7,)

    class FakeBaseEnv:
        observation_space = FakeSpace()
        action_space = FakeSpace()

        def reset(self, seed=None, options=None):
            return [0.0] * 7, {}

    class FakeJoint(FakeBaseEnv):
        def __init__(self, env, residual_scale, trajectory):
            captured["joint"] = (env, residual_scale, trajectory)

    class FakeProbe(FakeBaseEnv):
        def __init__(self, env, latent_dim, n_probes, ablation, device, belief_encoder=None):
            captured["belief_encoder"] = belief_encoder
            captured["ablation"] = ablation

    monkeypatch.setattr(ev, "_load_genesis_gym_wrapper", lambda: lambda task_config, scene_config: FakeBaseEnv())
    monkeypatch.setattr(ev, "_load_joint_residual_wrapper", lambda: FakeJoint)
    monkeypatch.setattr(ev, "_load_privileged_obs_wrapper", lambda: object)
    monkeypatch.setattr(ev, "_load_probe_phase_wrapper", lambda: FakeProbe)

    marker = object()
    env = ev.make_eval_env(
        split_config=ev.SPLITS["id_sand"],
        use_privileged=False,
        use_m7_env=True,
        ablation="none",
        horizon=500,
        residual_scale=0.05,
        seed=42,
        belief_encoder=marker,
    )

    assert env is not None
    assert captured["belief_encoder"] is marker
    assert captured["ablation"] == "none"
```

- [ ] **Step 2: Run eval contract tests and verify RED**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_run_ood_eval_v2.py -q`

Expected: FAIL because parser options, resolver helpers, and loader seams are missing.

- [ ] **Step 3: Add eval loader seams and parser options**

Modify `pta/scripts/run_ood_eval_v2.py`:

```python
import random

from pta.training.utils.checkpoint_io import load_m7_encoder_artifact, sha256_file
```

Change `parse_args` to accept an optional `argv` and add the encoder flags:

```python
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="OOD Evaluation v2")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--residual-scale", type=float, default=0.05)
    parser.add_argument(
        "--m7-encoder-mode",
        choices=("matched", "random-stress"),
        default="matched",
        help="Full M7 encoder protocol: matched sidecar for claims, random-stress for diagnostics.",
    )
    parser.add_argument(
        "--m7-random-encoder-seed",
        type=int,
        default=None,
        help="Required seed for --m7-encoder-mode random-stress.",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing OOD CSV progress and start from scratch.",
    )
    parser.set_defaults(resume=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods to eval (default: all)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=None,
        help="Subset of splits to eval (default: all)",
    )
    return parser.parse_args(argv)
```

Add import seams above `make_eval_env`:

```python
def _load_genesis_gym_wrapper():
    from pta.envs.wrappers.gym_wrapper import GenesisGymWrapper
    return GenesisGymWrapper


def _load_joint_residual_wrapper():
    from pta.envs.wrappers.joint_residual_wrapper import JointResidualWrapper
    return JointResidualWrapper


def _load_privileged_obs_wrapper():
    from pta.envs.wrappers.privileged_obs_wrapper import PrivilegedObsWrapper
    return PrivilegedObsWrapper


def _load_probe_phase_wrapper():
    from pta.envs.wrappers.probe_phase_wrapper import ProbePhaseWrapper
    return ProbePhaseWrapper
```

Update `make_eval_env` to use those seams and accept `belief_encoder`:

```python
def make_eval_env(
    split_config,
    use_privileged,
    use_m7_env,
    ablation="none",
    horizon=500,
    residual_scale=0.05,
    seed=0,
    belief_encoder=None,
):
    GenesisGymWrapper = _load_genesis_gym_wrapper()
    JointResidualWrapper = _load_joint_residual_wrapper()
    PrivilegedObsWrapper = _load_privileged_obs_wrapper()
    scene_config = {
        "tool_type": "scoop",
        "n_envs": 0,
        "particle_material": split_config["particle_material"],
    }
    if split_config.get("particle_params"):
        scene_config["particle_params"] = split_config["particle_params"]
    task_config = {"horizon": horizon, "success_threshold": 0.3}
    base_env = GenesisGymWrapper(task_config=task_config, scene_config=scene_config)
    env = JointResidualWrapper(base_env, residual_scale=residual_scale, trajectory="edge_push")
    if use_m7_env:
        ProbePhaseWrapper = _load_probe_phase_wrapper()
        env = ProbePhaseWrapper(
            env,
            latent_dim=16,
            n_probes=3,
            ablation=ablation,
            device="cpu",
            belief_encoder=belief_encoder,
        )
    elif use_privileged:
        env = PrivilegedObsWrapper(env=env, scene_config=scene_config)
    env.reset(seed=seed)
    return env
```

- [ ] **Step 4: Add encoder identity fields and resolver**

Update `RESULT_FIELDNAMES` to include protocol fields before metrics:

```python
RESULT_FIELDNAMES = [
    "method",
    "seed",
    "split",
    "encoder_mode",
    "encoder_seed",
    "encoder_artifact",
    "encoder_sha256",
    "policy_checkpoint",
    "policy_sha256",
    "protocol",
    "mean_reward",
    "std_reward",
    "mean_transfer",
    "std_transfer",
    "mean_spill",
    "std_spill",
    "success_rate",
    "n_failed_episodes",
]
```

Replace `result_key` with:

```python
def result_key(row):
    return (
        row["method"],
        int(row["seed"]),
        row["split"],
        row.get("encoder_mode", "legacy"),
        str(row.get("encoder_seed", "")),
        str(row.get("encoder_sha256", "")),
        str(row.get("policy_sha256", "")),
        str(row.get("protocol", "legacy")),
    )
```

Add resolver:

```python
def resolve_m7_belief_encoder(
    *,
    policy_path: Path,
    ablation: str,
    encoder_mode: str,
    encoder_seed: int | None,
    expected: dict,
):
    policy_sha = sha256_file(policy_path) if Path(policy_path).exists() else ""
    policy_rel = str(Path(policy_path).relative_to(_PROJECT_ROOT)) if Path(policy_path).is_relative_to(_PROJECT_ROOT) else str(policy_path)
    if ablation in {"no_probe", "no_belief"}:
        return None, {
            "encoder_mode": "zero-z",
            "encoder_seed": "",
            "encoder_artifact": "",
            "encoder_sha256": "",
            "policy_checkpoint": policy_rel,
            "policy_sha256": policy_sha,
            "protocol": "ablation_zero_z",
        }
    if encoder_mode == "random-stress":
        if encoder_seed is None:
            raise ValueError("--m7-random-encoder-seed is required for random-stress")
        random.seed(encoder_seed)
        np.random.seed(encoder_seed)
        try:
            import torch
            torch.manual_seed(encoder_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(encoder_seed)
        except ImportError:
            pass
        return None, {
            "encoder_mode": "random-stress",
            "encoder_seed": str(encoder_seed),
            "encoder_artifact": "fresh-random",
            "encoder_sha256": "",
            "policy_checkpoint": policy_rel,
            "policy_sha256": policy_sha,
            "protocol": "random_eval_encoder_stress",
        }
    if encoder_mode != "matched":
        raise ValueError(f"unknown M7 encoder mode: {encoder_mode}")
    encoder, metadata = load_m7_encoder_artifact(policy_path, expected=expected)
    return encoder, {
        "encoder_mode": "matched",
        "encoder_seed": "",
        "encoder_artifact": metadata["belief_encoder_path"],
        "encoder_sha256": metadata["belief_encoder_sha256"],
        "policy_checkpoint": policy_rel,
        "policy_sha256": policy_sha,
        "protocol": metadata["protocol"],
    }
```

Use `Path.is_relative_to`; Python 3.10 supports it.

- [ ] **Step 5: Wire resolver into main loop**

In `main`, before constructing each M7 env, call the resolver for M7 methods:

```python
encoder = None
encoder_identity = {
    "encoder_mode": "not_applicable",
    "encoder_seed": "",
    "encoder_artifact": "",
    "encoder_sha256": "",
    "policy_checkpoint": str(ckpt_path.relative_to(_PROJECT_ROOT)),
    "policy_sha256": sha256_file(ckpt_path),
    "protocol": "not_applicable",
}
if method_cfg.get("use_m7_env", False):
    encoder, encoder_identity = resolve_m7_belief_encoder(
        policy_path=ckpt_path,
        ablation=method_cfg.get("ablation", "none"),
        encoder_mode=args.m7_encoder_mode,
        encoder_seed=args.m7_random_encoder_seed,
        expected={
            "method": method_name,
            "seed": seed,
            "ablation": method_cfg.get("ablation", "none"),
            "latent_dim": 16,
            "n_probes": 3,
        },
    )
```

Update the inner split loop to pass `belief_encoder=encoder`, compute the expanded key, and write identity fields:

```python
row_identity = {
    "method": method_name,
    "seed": seed,
    "split": split_name,
    **encoder_identity,
}
key = result_key(row_identity)
if key in completed_keys:
    print(f"    {split_name} SKIP existing", flush=True)
    continue

env = make_eval_env(
    split_config=split_cfg,
    use_privileged=method_cfg["use_privileged"],
    use_m7_env=method_cfg.get("use_m7_env", False),
    ablation=method_cfg.get("ablation", "none"),
    horizon=args.horizon,
    residual_scale=args.residual_scale,
    seed=seed + 2000,
    belief_encoder=encoder,
)
metrics = evaluate_one(model, env, args.n_episodes)
env.close()
row = {**row_identity, **metrics}
all_rows.append(row)
append_result_row(per_seed_path, row)
completed_keys.add(key)
```

- [ ] **Step 6: Run eval tests and verify GREEN**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_run_ood_eval_v2.py -q`

Expected: PASS.

- [ ] **Step 7: Commit Task 2**

Run:

```bash
git add pta/scripts/run_ood_eval_v2.py tests/test_run_ood_eval_v2.py
git commit -m "eval: require matched encoder for full M7"
```

Expected: commit succeeds.

## Task 3: M7 Training Saves Matched Sidecars

**Files:**
- Modify: `pta/scripts/train_m7.py`
- Create or modify: `tests/test_train_m7_encoder_sidecars.py`

- [ ] **Step 1: Add failing train-side sidecar tests**

Create `tests/test_train_m7_encoder_sidecars.py`:

```python
from pathlib import Path

import torch

from pta.models.belief.latent_belief_encoder import LatentBeliefEncoder
from pta.scripts import train_m7


def test_clone_belief_encoder_state_produces_independent_matching_module():
    source = LatentBeliefEncoder(trace_dim=30, latent_dim=16, hidden_dim=128, num_layers=2)
    for parameter in source.parameters():
        torch.nn.init.constant_(parameter, 0.42)

    clone = train_m7.clone_belief_encoder_state(source)

    assert clone is not source
    for key, value in source.state_dict().items():
        assert torch.equal(value, clone.state_dict()[key])


class FakeModel:
    def save(self, path):
        Path(path).with_suffix(".zip").parent.mkdir(parents=True, exist_ok=True)
        Path(path).with_suffix(".zip").write_bytes(b"policy")


def test_save_m7_policy_with_encoder_writes_policy_and_encoder_sidecars(tmp_path):
    encoder = LatentBeliefEncoder(trace_dim=30, latent_dim=16, hidden_dim=128, num_layers=2)
    path = tmp_path / "checkpoints" / "m7_pta_seed42" / "m7_pta_final"

    train_m7.save_m7_policy_with_encoder(
        model=FakeModel(),
        encoder=encoder,
        path=path,
        repo_root=tmp_path,
        metadata={
            "total_timesteps": 500000,
            "seed": 42,
            "method": "m7_pta",
            "ablation": "none",
            "trace_dim": 30,
            "latent_dim": 16,
            "hidden_dim": 128,
            "num_layers": 2,
            "n_probes": 3,
            "stage": "final",
        },
    )

    assert path.with_suffix(".zip").exists()
    assert path.with_suffix(".json").exists()
    assert (path.parent / "belief_encoder.pt").exists()
    assert (path.parent / "belief_encoder_metadata.json").exists()
```

- [ ] **Step 2: Run train-side tests and verify RED**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_train_m7_encoder_sidecars.py -q`

Expected: FAIL because `clone_belief_encoder_state` and `save_m7_policy_with_encoder` do not exist.

- [ ] **Step 3: Add train-side helpers**

In `pta/scripts/train_m7.py`, add helper imports:

```python
import copy

from pta.training.utils.checkpoint_io import save_m7_encoder_artifact
```

Add helpers near `make_m7_env`:

```python
ENCODER_HIDDEN_DIM = 128
ENCODER_NUM_LAYERS = 2


def clone_belief_encoder_state(encoder):
    clone = copy.deepcopy(encoder)
    clone.eval()
    return clone


def save_m7_policy_with_encoder(*, model, encoder, path: Path, repo_root: Path, metadata: dict) -> None:
    from pta.training.utils.checkpoint_io import save_sb3_checkpoint

    save_sb3_checkpoint(model, path, metadata=metadata)
    save_m7_encoder_artifact(
        encoder=encoder,
        policy_path=path.with_suffix(".zip"),
        repo_root=repo_root,
        run_metadata=metadata,
    )
```

- [ ] **Step 4: Update environment construction for injected encoders**

Change `make_m7_env` signature:

```python
def make_m7_env(
    task_config=None,
    scene_config=None,
    seed: int = 0,
    joint_residual_scale: float = 0.2,
    joint_residual_trajectory: str = "edge_push",
    latent_dim: int = 16,
    n_probes: int = 3,
    ablation: str = "none",
    belief_encoder=None,
):
```

Pass `belief_encoder=belief_encoder` into `ProbePhaseWrapper`.

In `main`, build a temporary M7 env after `set_seed(seed)` to get a canonical full M7 encoder state. Use that encoder state for train and eval env factories. The implementation can be:

```python
canonical_env = make_m7_env(
    task_config=task_config,
    scene_config=scene_config,
    seed=seed,
    joint_residual_scale=args.residual_scale,
    latent_dim=args.latent_dim,
    n_probes=args.n_probes,
    ablation=ablation,
)
canonical_encoder = canonical_env._belief_encoder
trace_dim = canonical_env._inner_obs_dim
canonical_env.close()
```

Then train/eval factories use independent copies:

```python
belief_encoder=clone_belief_encoder_state(canonical_encoder)
```

For `no_probe` and `no_belief`, this can still pass a cloned encoder, but sidecar saving is only required for `ablation == "none"`.

- [ ] **Step 5: Add best-checkpoint sidecar callback**

Add this class in `train_m7.py`:

```python
class EncoderSidecarEvalCallback(EvalCallback):
    def __init__(self, *args, encoder, repo_root: Path, metadata_factory, **kwargs):
        super().__init__(*args, **kwargs)
        self._encoder = encoder
        self._repo_root = repo_root
        self._metadata_factory = metadata_factory
        self._last_best_mean_reward = -float("inf")

    def _on_step(self) -> bool:
        before = self.best_mean_reward
        keep_training = super()._on_step()
        if self.best_model_save_path is not None and self.best_mean_reward > before:
            best_path = Path(self.best_model_save_path) / "best_model"
            metadata = self._metadata_factory(stage="best")
            save_m7_encoder_artifact(
                encoder=self._encoder,
                policy_path=best_path.with_suffix(".zip"),
                repo_root=self._repo_root,
                run_metadata=metadata,
            )
        return keep_training
```

Replace `EvalCallback` with `EncoderSidecarEvalCallback` for `ablation == "none"`. Keep regular `EvalCallback` for ablations.

- [ ] **Step 6: Save final matched sidecar**

Replace the final `save_sb3_checkpoint` call for `ablation == "none"` with:

```python
metadata = {
    "total_timesteps": args.total_timesteps,
    "seed": seed,
    "method": "m7_pta",
    "ablation": ablation,
    "trace_dim": trace_dim,
    "latent_dim": args.latent_dim,
    "hidden_dim": ENCODER_HIDDEN_DIM,
    "num_layers": ENCODER_NUM_LAYERS,
    "n_probes": args.n_probes,
    "stage": "final",
}
save_m7_policy_with_encoder(
    model=model,
    encoder=canonical_encoder,
    path=final_path,
    repo_root=_PROJECT_ROOT,
    metadata=metadata,
)
```

Keep the existing `save_sb3_checkpoint` path for `no_probe` and `no_belief`, and include `legacy_policy_only` or `encoder_mode=zero-z` in their metadata.

- [ ] **Step 7: Run train-side tests and focused existing tests**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_train_m7_encoder_sidecars.py tests/test_m7_encoder_artifacts.py -q`

Expected: PASS.

- [ ] **Step 8: Commit Task 3**

Run:

```bash
git add pta/scripts/train_m7.py tests/test_train_m7_encoder_sidecars.py
git commit -m "train: persist matched M7 encoder sidecars"
```

Expected: commit succeeds.

## Task 4: Pre-Submission Audit Modes

**Files:**
- Modify: `tests/test_pre_submission_audit.py`
- Modify: `tools/pre_submission_audit.py`

- [ ] **Step 1: Add failing audit-label tests**

Append to `tests/test_pre_submission_audit.py`:

```python
from tools import pre_submission_audit as audit


def test_encoder_sensitivity_mode_is_random_eval_encoder_stress_label():
    gate = audit.encoder_sensitivity_gate(
        [{"mean_transfer": 0.1, "n_failed_episodes": 0}, {"mean_transfer": 0.2, "n_failed_episodes": 0}],
        max_transfer_range_pp=15.0,
    )
    assert gate["passes"] is True
    assert audit.RANDOM_STRESS_MODE_NAME == "random_eval_encoder_stress"


def test_parse_args_accepts_matched_encoder_audit_mode():
    args = audit.parse_args(["--mode", "matched-encoder-audit", "--method", "m7_pta", "--seed", "42"])
    assert args.mode == "matched-encoder-audit"
    assert args.method == "m7_pta"
```

- [ ] **Step 2: Run audit tests and verify RED**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_pre_submission_audit.py -q`

Expected: FAIL because the constant and mode are missing.

- [ ] **Step 3: Add audit split names and matched mode**

In `tools/pre_submission_audit.py`, add:

```python
RANDOM_STRESS_MODE_NAME = "random_eval_encoder_stress"
MATCHED_ENCODER_MODE_NAME = "matched_encoder_checkpoint_audit"
```

Update parser choices to include `matched-encoder-audit`.

In `run_encoder_sensitivity`, call `ev.make_eval_env` with explicit random mode by using the resolver from `run_ood_eval_v2`:

```python
encoder, encoder_identity = ev.resolve_m7_belief_encoder(
    policy_path=checkpoint,
    ablation=method_cfg["ablation"],
    encoder_mode="random-stress",
    encoder_seed=encoder_seed,
    expected={"method": args.method, "seed": args.seed, "ablation": method_cfg["ablation"], "latent_dim": 16, "n_probes": 3},
)
env = ev.make_eval_env(
    split_config=ev.SPLITS[args.split],
    use_privileged=method_cfg["use_privileged"],
    use_m7_env=method_cfg["use_m7_env"],
    ablation=method_cfg["ablation"],
    horizon=args.horizon,
    residual_scale=args.residual_scale,
    seed=args.seed + 2000,
    belief_encoder=encoder,
)
rows.append({"encoder_seed": encoder_seed, **encoder_identity, **metrics})
```

Set payload mode to `RANDOM_STRESS_MODE_NAME`.

Add `run_matched_encoder_audit(args)`:

```python
def run_matched_encoder_audit(args: argparse.Namespace) -> None:
    ev = _load_eval_module()
    PPO = _load_ppo()
    method_cfg = METHODS[args.method]
    checkpoint = ev.resolve_checkpoint_path(PROJECT_ROOT, method_cfg["ckpt_pattern"], args.seed)
    if checkpoint is None:
        raise FileNotFoundError(method_cfg["ckpt_pattern"].format(seed=args.seed))
    model = PPO.load(str(checkpoint), device="auto")
    encoder, encoder_identity = ev.resolve_m7_belief_encoder(
        policy_path=checkpoint,
        ablation=method_cfg["ablation"],
        encoder_mode="matched",
        encoder_seed=None,
        expected={"method": args.method, "seed": args.seed, "ablation": method_cfg["ablation"], "latent_dim": 16, "n_probes": 3},
    )
    env = ev.make_eval_env(
        split_config=ev.SPLITS[args.split],
        use_privileged=method_cfg["use_privileged"],
        use_m7_env=method_cfg["use_m7_env"],
        ablation=method_cfg["ablation"],
        horizon=args.horizon,
        residual_scale=args.residual_scale,
        seed=args.seed + 2000,
        belief_encoder=encoder,
    )
    try:
        metrics = ev.evaluate_one(model, env, args.n_episodes, deterministic=True)
    finally:
        env.close()
    payload = {
        "mode": MATCHED_ENCODER_MODE_NAME,
        "method": args.method,
        "policy_seed": args.seed,
        "split": args.split,
        "n_episodes": args.n_episodes,
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
        **encoder_identity,
        **metrics,
    }
    _write_json(Path(args.output_dir) / f"audit_matched_encoder_{args.method}_s{args.seed}_{args.split}.json", payload)
```

Dispatch it in `main`.

- [ ] **Step 4: Run audit tests**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_pre_submission_audit.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 4**

Run:

```bash
git add tools/pre_submission_audit.py tests/test_pre_submission_audit.py
git commit -m "audit: split random stress from matched encoder checks"
```

Expected: commit succeeds.

## Task 5: Artifact Registry Type-Specific Validation

**Files:**
- Modify: `tests/test_artifact_registry.py`
- Modify: `tools/artifact_registry.py`

- [ ] **Step 1: Add failing registry tests for matched M7 bundles**

Append to `tests/test_artifact_registry.py`:

```python
def test_g2_matched_encoder_requirement_includes_policy_and_encoder_sidecars():
    assert set(registry.requirement_paths("g2-matched-encoder")) == {
        "checkpoints/m7_pta_seed42/best/best_model.zip",
        "checkpoints/m7_pta_seed42/best/best_model.json",
        "checkpoints/m7_pta_seed42/best/belief_encoder.pt",
        "checkpoints/m7_pta_seed42/best/belief_encoder_metadata.json",
    }


def test_verify_matched_encoder_artifacts_uses_type_specific_loaders(tmp_path, monkeypatch):
    policy = _write(tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip", b"policy")
    encoder = _write(tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "belief_encoder.pt", b"encoder")
    metadata = {
        "protocol": "matched_encoder_v1",
        "method": "m7_pta",
        "seed": 42,
        "ablation": "none",
        "latent_dim": 16,
        "n_probes": 3,
        "paired_policy_path": "checkpoints/m7_pta_seed42/best/best_model.zip",
        "paired_policy_sha256": hashlib.sha256(b"policy").hexdigest(),
        "belief_encoder_path": "checkpoints/m7_pta_seed42/best/belief_encoder.pt",
        "belief_encoder_sha256": hashlib.sha256(b"encoder").hexdigest(),
    }
    _write(tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.json", json.dumps({"protocol": "matched_encoder_v1"}).encode())
    _write(tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "belief_encoder_metadata.json", json.dumps(metadata).encode())

    class FakeTorch:
        @staticmethod
        def load(path, map_location="cpu", weights_only=False):
            assert Path(path) == encoder.resolve()
            return {"format_version": 1, "state_dict": {}, "config": {"trace_dim": 30, "latent_dim": 16, "hidden_dim": 128, "num_layers": 2}}

    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)
    monkeypatch.setattr(registry, "_load_torch", lambda: FakeTorch)

    manifest = registry.verify_artifacts(tmp_path, ["g2-matched-encoder"])
    statuses = {row["relative_path"]: row["load_status"] for row in manifest["artifacts"]}
    assert statuses["checkpoints/m7_pta_seed42/best/best_model.zip"] == "loaded"
    assert statuses["checkpoints/m7_pta_seed42/best/belief_encoder.pt"] == "loaded"
    assert statuses["checkpoints/m7_pta_seed42/best/belief_encoder_metadata.json"] == "loaded"


def test_verify_matched_encoder_metadata_rejects_policy_hash_mismatch(tmp_path, monkeypatch):
    _write(tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.zip", b"policy")
    _write(tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "best_model.json", b"{}")
    _write(tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "belief_encoder.pt", b"encoder")
    _write(
        tmp_path / "checkpoints" / "m7_pta_seed42" / "best" / "belief_encoder_metadata.json",
        json.dumps({
            "protocol": "matched_encoder_v1",
            "paired_policy_path": "checkpoints/m7_pta_seed42/best/best_model.zip",
            "paired_policy_sha256": "wrong",
            "belief_encoder_path": "checkpoints/m7_pta_seed42/best/belief_encoder.pt",
            "belief_encoder_sha256": hashlib.sha256(b"encoder").hexdigest(),
        }).encode(),
    )

    monkeypatch.setattr(registry, "_load_ppo", lambda: FakePPO)
    manifest = registry.verify_artifacts(tmp_path, ["g2-matched-encoder"])
    failed = registry.failed_required_loads(manifest)
    assert "m7_pta_seed42_belief_encoder_metadata" in failed
```

- [ ] **Step 2: Run registry tests and verify RED**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_artifact_registry.py -q`

Expected: FAIL because `g2-matched-encoder` and type-specific validation are missing.

- [ ] **Step 3: Add matched candidates and type loaders**

In `tools/artifact_registry.py`, add candidate helpers:

```python
def _best_policy_metadata(method: str, seed: int) -> str:
    return f"checkpoints/{method}_seed{seed}/best/best_model.json"


def _best_belief_encoder(method: str, seed: int) -> str:
    return f"checkpoints/{method}_seed{seed}/best/belief_encoder.pt"


def _best_belief_encoder_metadata(method: str, seed: int) -> str:
    return f"checkpoints/{method}_seed{seed}/best/belief_encoder_metadata.json"
```

Add requirement:

```python
"g2-matched-encoder": (
    ArtifactCandidate("m7_pta_seed42_best", _best_checkpoint("m7_pta", 42), kind="policy_checkpoint", required_for="g2-matched-encoder", min_num_timesteps=50_000),
    ArtifactCandidate("m7_pta_seed42_best_metadata", _best_policy_metadata("m7_pta", 42), kind="policy_metadata", required_for="g2-matched-encoder"),
    ArtifactCandidate("m7_pta_seed42_belief_encoder", _best_belief_encoder("m7_pta", 42), kind="belief_encoder", required_for="g2-matched-encoder"),
    ArtifactCandidate("m7_pta_seed42_belief_encoder_metadata", _best_belief_encoder_metadata("m7_pta", 42), kind="belief_encoder_metadata", required_for="g2-matched-encoder"),
),
```

Change existing checkpoint candidates to `kind="policy_checkpoint"` where they are SB3 zips.

Add:

```python
def _load_torch():
    import torch
    return torch


def _verify_policy_checkpoint(row, ppo):
    model = ppo.load(str(row["source_path"]), device="auto")
    row["num_timesteps"] = getattr(model, "num_timesteps", None)
    min_num_timesteps = row.get("min_num_timesteps")
    if min_num_timesteps is not None and (row["num_timesteps"] is None or row["num_timesteps"] < min_num_timesteps):
        raise RuntimeError(f"num_timesteps {row['num_timesteps']} below required minimum {min_num_timesteps}")


def _verify_json_file(row):
    json.loads(Path(str(row["source_path"])).read_text(encoding="utf-8"))


def _verify_belief_encoder(row, torch_module):
    payload = torch_module.load(str(row["source_path"]), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise RuntimeError("unsupported belief encoder artifact format")
    if "state_dict" not in payload or "config" not in payload:
        raise RuntimeError("belief encoder artifact missing state_dict or config")


def _verify_belief_encoder_metadata(row, repo_root: Path):
    metadata_path = Path(str(row["source_path"]))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    policy = repo_root / str(metadata["paired_policy_path"])
    encoder = repo_root / str(metadata["belief_encoder_path"])
    if metadata.get("paired_policy_sha256") != sha256_file(policy):
        raise RuntimeError("paired_policy_sha256 mismatch")
    if metadata.get("belief_encoder_sha256") != sha256_file(encoder):
        raise RuntimeError("belief_encoder_sha256 mismatch")
```

Update `verify_artifacts` to dispatch by `row["kind"]` and mark row `loaded` only after kind-specific validation.

- [ ] **Step 4: Run registry tests and verify GREEN**

Run: `/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest tests/test_artifact_registry.py -q`

Expected: PASS.

- [ ] **Step 5: Commit Task 5**

Run:

```bash
git add tools/artifact_registry.py tests/test_artifact_registry.py
git commit -m "tools: validate matched M7 encoder artifacts"
```

Expected: commit succeeds.

## Task 6: Documentation And Runbook Updates

**Files:**
- Modify: `docs/30_records/SEED_SENSITIVITY_STATUS.md`
- Modify: `docs/30_records/DLC_PRESUBMISSION_RUNBOOK.md`
- Modify: `docs/30_records/CHECKPOINT_MANIFEST.md`

- [ ] **Step 1: Update seed sensitivity status**

Add a section to `docs/30_records/SEED_SENSITIVITY_STATUS.md`:

```markdown
## M7 Matched-Encoder Protocol Update

The old G2 result is now classified as a `random_eval_encoder_stress` diagnostic because it paired a fixed policy checkpoint with freshly initialized evaluation encoders. It is not claim-bearing matched M7 performance evidence.

The 12 replacement six-seed DLC jobs submitted before matched-encoder persistence are `policy-only legacy diagnostics`. They can inform policy-seed variance, but they cannot support matched policy-plus-encoder claims.

Future full `m7_pta` claim artifacts must include `best_model.zip`, `best_model.json`, `belief_encoder.pt`, and `belief_encoder_metadata.json` under the matched-encoder protocol.
```

- [ ] **Step 2: Update runbook artifact requirements**

In `docs/30_records/DLC_PRESUBMISSION_RUNBOOK.md`, add:

```markdown
For corrected M7 G2, use `tools/artifact_registry.py verify --requirement g2-matched-encoder`. A policy zip alone is no longer sufficient for full `m7_pta` matched evaluation.
```

In `docs/30_records/CHECKPOINT_MANIFEST.md`, add the four matched artifacts and note that encoder `.pt` files stay out of Git.

- [ ] **Step 3: Run doc whitespace check**

Run: `git diff --check`

Expected: no output.

- [ ] **Step 4: Commit Task 6**

Run:

```bash
git add docs/30_records/SEED_SENSITIVITY_STATUS.md docs/30_records/DLC_PRESUBMISSION_RUNBOOK.md docs/30_records/CHECKPOINT_MANIFEST.md
git commit -m "docs: record matched encoder evaluation protocol"
```

Expected: commit succeeds.

## Task 7: Full Local Verification

**Files:**
- Read: all modified source and test files

- [ ] **Step 1: Run targeted test suite**

Run:

```bash
/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m pytest \
  tests/test_m7_encoder_artifacts.py \
  tests/test_train_m7_encoder_sidecars.py \
  tests/test_run_ood_eval_v2.py \
  tests/test_pre_submission_audit.py \
  tests/test_artifact_registry.py \
  tests/test_dlc_shell_contract.py \
  tests/test_dlc_submit_jobs.py
```

Expected: all selected tests pass.

- [ ] **Step 2: Run syntax and whitespace checks**

Run:

```bash
/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python -m py_compile \
  pta/training/utils/checkpoint_io.py \
  pta/scripts/train_m7.py \
  pta/scripts/run_ood_eval_v2.py \
  tools/pre_submission_audit.py \
  tools/artifact_registry.py
git diff --check
```

Expected: both commands exit `0`; `git diff --check` has no output.

- [ ] **Step 3: Commit verification-only fixes if needed**

If Step 1 or Step 2 reveals small test/import/type errors, fix only those errors, rerun the same verification commands, then commit:

```bash
git add pta tests tools docs
git commit -m "test: verify matched encoder protocol"
```

Expected: no commit is made if no fixes are needed.

## Task 8: DLC Smoke And G2 Rollout After Merge To Runtime Tree

**Files:**
- Runtime tree: `/cpfs/shared/simulation/zhuzihou/dev/probe-then-act`
- Artifact root: `/cpfs/shared/simulation/zhuzihou/artifacts/probe-then-act`

- [ ] **Step 1: Sync or merge implementation into the CPFS runtime tree**

Run the project’s normal branch integration flow after review. Then verify the runtime tree contains the matched-encoder implementation:

```bash
git -C /cpfs/shared/simulation/zhuzihou/dev/probe-then-act rev-parse HEAD
grep -R "matched_encoder_v1" /cpfs/shared/simulation/zhuzihou/dev/probe-then-act/pta /cpfs/shared/simulation/zhuzihou/dev/probe-then-act/tools
```

Expected: the intended commit is present and `matched_encoder_v1` appears in runtime code.

- [ ] **Step 2: Submit one micro/smoke M7 job**

Run from `/cpfs/shared/simulation/zhuzihou/dev/probe-then-act` with verified image exported:

```bash
export PTA_CODE_ROOT=/cpfs/shared/simulation/zhuzihou/dev/probe-then-act
export GENESIS_ROOT=/cpfs/shared/simulation/zhuzihou/dev/Genesis
export GENESIS_VENV=/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv
export PYTHON_BIN=/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python
export DLC_BIN=/cpfs/shared/simulation/zhuzihou/dev/usd-scene-physics-prep/dlc
export DLC_DATA_SOURCES=d-mzps5b7joy2axmqpa8,d-d49o5g0h2818sw8j1g,d-8wz4emfs21s5ajs9oz
export DLC_GPU_COUNT=1
export DLC_IMAGE=pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang
bash pta/scripts/dlc/launch_job.sh pta_m7_matched_encoder_smoke_s42 0 1 "$DLC_DATA_SOURCES" \
  custom env PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 "$PYTHON_BIN" -u pta/scripts/train_m7.py \
    --seed 42 --total-timesteps 8192 --residual-scale 0.05 --horizon 500 --latent-dim 16 --n-probes 3 --eval-freq 4096
```

Expected: DLC returns a JobId.

- [ ] **Step 3: Verify smoke job image and artifacts**

Run:

```bash
"$DLC_BIN" get job <SMOKE_JOB_ID> --endpoint=pai-dlc.cn-beijing.aliyuncs.com --region=cn-beijing
```

Expected: `JobSpecs[0].Image` equals `pj4090acr-registry-vpc.cn-beijing.cr.aliyuncs.com/pj4090/mahaoxiang:genmanip-mahaoxiang`.

After the job succeeds, verify:

```bash
test -f checkpoints/m7_pta_seed42/belief_encoder.pt
test -f checkpoints/m7_pta_seed42/belief_encoder_metadata.json
/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python tools/artifact_registry.py verify --requirement g2-matched-encoder --manifest /tmp/pta_g2_matched_verify.json
```

Expected: policy and encoder artifacts load; registry verification exits `0` for the smoke checkpoint path if best artifacts exist. If the smoke job does not produce `best/` artifacts because eval did not improve, inspect final sidecars and adjust the smoke command eval frequency before full run.

- [ ] **Step 4: Submit full `m7_pta seed42` matched run**

After smoke verification, submit:

```bash
bash pta/scripts/dlc/launch_job.sh pta_m7_matched_encoder_s42 0 1 "$DLC_DATA_SOURCES" \
  custom env PYOPENGL_PLATFORM=egl EGL_DEVICE_ID=0 "$PYTHON_BIN" -u pta/scripts/train_m7.py \
    --seed 42 --total-timesteps 500000 --residual-scale 0.05 --horizon 500 --latent-dim 16 --n-probes 3 --eval-freq 50000
```

Expected: DLC returns a JobId and image verification passes.

- [ ] **Step 5: Register and rerun matched G2**

After the full job succeeds and artifacts verify:

```bash
/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python tools/artifact_registry.py verify --requirement g2-matched-encoder --manifest /tmp/pta_g2_matched_verify.json
/cpfs/shared/simulation/zhuzihou/dev/Genesis/.venv/bin/python tools/pre_submission_audit.py --mode matched-encoder-audit --method m7_pta --seed 42 --split ood_elastoplastic --n-episodes 10
```

Expected: registry verification exits `0`; matched G2 writes `results/presub/audit_matched_encoder_m7_pta_s42_ood_elastoplastic.json`.

- [ ] **Step 6: Decide larger matched batch**

If the full seed42 matched path verifies, submit additional full `m7_pta` matched runs for seeds `2`, `3`, and `4` using the same verified image and archive flow. Do not submit a larger matrix before the seed42 matched G2 artifact is registered and restorable.
