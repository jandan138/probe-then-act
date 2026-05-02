"""Checkpoint save / load utilities.

Supports both SB3 model checkpoints (via model.save / model.load) and
custom PyTorch checkpoints with metadata (config, seed, step, metrics).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import torch


M7_ENCODER_PROTOCOL = "matched_encoder_v1"
M7_ENCODER_FORMAT_VERSION = 1


@dataclass(frozen=True)
class M7EncoderSidecarPaths:
    policy_path: Path
    policy_metadata_path: Path
    encoder_path: Path
    metadata_path: Path


def save_checkpoint(
    state: Dict[str, Any],
    path: Path,
    is_best: bool = False,
) -> None:
    """Persist a training checkpoint to disk.

    Parameters
    ----------
    state : dict[str, Any]
        Checkpoint payload -- typically includes ``model_state_dict``,
        ``optimizer_state_dict``, ``epoch``, ``global_step``, and any
        scheduler / curriculum state.
    path : Path
        File path for the checkpoint (e.g. ``checkpoints/step_100000.pt``).
    is_best : bool
        If ``True``, also copy the checkpoint to ``best.pt`` in the same
        directory.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Add timestamp if not present
    if "timestamp" not in state:
        state["timestamp"] = datetime.now(timezone.utc).isoformat()

    torch.save(state, path)

    if is_best:
        best_path = path.parent / "best.pt"
        shutil.copy2(path, best_path)


def load_checkpoint(
    path: Path,
    map_location: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a training checkpoint from disk.

    Parameters
    ----------
    path : Path
        Path to the checkpoint file.
    map_location : str, optional
        Device mapping string (e.g. ``"cpu"``).  Passed directly to
        ``torch.load``.

    Returns
    -------
    dict[str, Any]
        The checkpoint payload.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    return torch.load(path, map_location=map_location, weights_only=False)


def save_sb3_checkpoint(
    model: Any,
    path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save an SB3 model plus optional metadata sidecar.

    Parameters
    ----------
    model:
        A Stable-Baselines3 model (PPO, RecurrentPPO, etc.).
    path:
        Save path (without extension -- SB3 adds ``.zip``).
    metadata:
        Optional metadata dict (config, seed, step, metrics, etc.)
        saved as a JSON sidecar alongside the model.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))

    if metadata is not None:
        meta_path = path.with_suffix(".json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)


def load_sb3_metadata(path: Path) -> Dict[str, Any]:
    """Load the JSON metadata sidecar for an SB3 checkpoint.

    Parameters
    ----------
    path:
        Path to the ``.zip`` model file (the sidecar is ``.json``).

    Returns
    -------
    dict
        Metadata dict.
    """
    meta_path = Path(path).with_suffix(".json")
    if not meta_path.exists():
        return {}
    with open(meta_path) as f:
        return json.load(f)


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
    path = Path(path)
    repo_root = Path(repo_root)
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> Dict[str, Any]:
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

    required = [
        "method",
        "seed",
        "ablation",
        "trace_dim",
        "latent_dim",
        "hidden_dim",
        "num_layers",
        "n_probes",
    ]
    missing = [field for field in required if field not in run_metadata]
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
        "run_metadata": dict(run_metadata),
        **run_metadata,
        "paired_policy_path": _relative_to_repo(paths.policy_path, repo_root),
        "paired_policy_sha256": sha256_file(paths.policy_path),
        "belief_encoder_path": _relative_to_repo(paths.encoder_path, repo_root),
        "belief_encoder_sha256": sha256_file(paths.encoder_path),
        "created_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
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


def validate_m7_encoder_metadata(
    metadata: Dict[str, Any],
    expected: Dict[str, Any],
) -> None:
    if metadata.get("protocol") != M7_ENCODER_PROTOCOL:
        raise ValueError("protocol mismatch for matched encoder")
    for field, expected_value in expected.items():
        if expected_value is not None and metadata.get(field) != expected_value:
            raise ValueError(
                f"{field} mismatch for matched encoder: "
                f"expected {expected_value!r}, got {metadata.get(field)!r}"
            )


def load_m7_encoder_artifact(
    policy_path: Path,
    expected: Dict[str, Any],
    map_location: str = "cpu",
) -> tuple[Any, Dict[str, Any]]:
    from pta.models.belief.latent_belief_encoder import LatentBeliefEncoder

    paths = m7_encoder_sidecar_paths(policy_path)
    if not paths.encoder_path.exists() or not paths.metadata_path.exists():
        raise FileNotFoundError(
            f"missing matched encoder artifact for {paths.policy_path}"
        )

    metadata = _read_json(paths.metadata_path)
    validate_m7_encoder_metadata(metadata, expected)
    if not paths.policy_path.exists():
        raise FileNotFoundError(
            f"missing matched encoder paired policy file: {paths.policy_path}"
        )
    if metadata.get("paired_policy_sha256") != sha256_file(paths.policy_path):
        raise ValueError("paired_policy_sha256 mismatch for matched encoder")
    if metadata.get("belief_encoder_sha256") != sha256_file(paths.encoder_path):
        raise ValueError("belief_encoder_sha256 mismatch for matched encoder")

    payload = torch.load(paths.encoder_path, map_location=map_location, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("matched encoder artifact must be a dictionary")
    if payload.get("format_version") != M7_ENCODER_FORMAT_VERSION:
        raise ValueError("unsupported matched encoder artifact format")
    config = payload.get("config")
    state_dict = payload.get("state_dict")
    if not isinstance(config, dict) or not isinstance(state_dict, dict):
        raise ValueError("matched encoder artifact must contain config and state_dict")
    required_config = ("trace_dim", "latent_dim", "hidden_dim", "num_layers")
    missing_config = [key for key in required_config if key not in config]
    if missing_config:
        raise ValueError(
            "matched encoder config missing required keys: "
            + ", ".join(missing_config)
        )

    encoder = LatentBeliefEncoder(
        trace_dim=int(config["trace_dim"]),
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
    )
    encoder.load_state_dict(state_dict)
    encoder.eval()
    return encoder, metadata
