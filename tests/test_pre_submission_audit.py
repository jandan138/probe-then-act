import csv
import json
import math
import types

import numpy as np
import pytest


def test_displacement_stats_reports_rms_mean_and_max():
    from tools import pre_submission_audit as audit

    before = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    after = np.array([[3.0, 4.0, 0.0], [1.0, 0.0, 12.0]])

    stats = audit.displacement_stats(before, after)

    assert stats == {
        "rms_m": pytest.approx(math.sqrt((25.0 + 144.0) / 2.0)),
        "mean_m": pytest.approx(8.5),
        "max_m": pytest.approx(12.0),
    }


def test_displacement_stats_rejects_shape_mismatch():
    from tools import pre_submission_audit as audit

    with pytest.raises(ValueError, match="same shape"):
        audit.displacement_stats(np.zeros((2, 3)), np.zeros((3, 3)))


def test_persistent_fraction_returns_none_for_zero_probe_displacement():
    from tools import pre_submission_audit as audit

    assert audit.persistent_fraction({"rms_m": 0.0}, {"rms_m": 1.0}) is None


def test_append_eval_row_and_existing_keys_round_trip(tmp_path):
    from tools import pre_submission_audit as audit

    path = tmp_path / "extra.csv"
    row = {
        "method": "m7_pta",
        "seed": 2,
        "split": "ood_elastoplastic",
        "mean_reward": 1.0,
        "std_reward": 0.0,
        "mean_transfer": 0.7,
        "std_transfer": 0.1,
        "mean_spill": 0.2,
        "std_spill": 0.1,
        "success_rate": 0.8,
        "n_failed_episodes": 0,
    }

    audit.append_eval_row(path, row)

    assert audit.existing_keys(path) == {("m7_pta", 2, "ood_elastoplastic")}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["method"] == "m7_pta"
    assert rows[0]["seed"] == "2"


def test_summarize_paired_elastoplastic_combines_original_and_extra_csvs(tmp_path):
    from tools import pre_submission_audit as audit

    original = tmp_path / "original.csv"
    extra = tmp_path / "extra.csv"
    header = "method,seed,split,mean_transfer\n"
    original.write_text(
        header
        + "m1_reactive,42,ood_elastoplastic,0.0\n"
        + "m7_pta,42,ood_elastoplastic,0.8\n"
        + "m1_reactive,0,ood_elastoplastic,0.4\n"
        + "m7_pta,0,ood_elastoplastic,0.1\n",
        encoding="utf-8",
    )
    extra.write_text(
        header
        + "m1_reactive,2,ood_elastoplastic,0.5\n"
        + "m7_pta,2,ood_elastoplastic,0.6\n"
        + "m1_reactive,3,ood_elastoplastic,0.2\n"
        + "m7_pta,3,ood_elastoplastic,0.9\n",
        encoding="utf-8",
    )

    summary = audit.summarize_paired_elastoplastic([original, extra])

    assert summary["n_pairs"] == 4
    assert summary["positive_pairs"] == 3
    assert summary["mean_delta_pp"] == pytest.approx(32.5)
    assert summary["pairs"][0] == {
        "seed": 0,
        "m1_transfer": 0.4,
        "m7_transfer": 0.1,
        "delta": -0.30000000000000004,
    }


def test_parse_args_for_summary_mode_does_not_load_eval_module(monkeypatch):
    from tools import pre_submission_audit as audit

    def fail_if_loaded():
        raise AssertionError("eval module should not be loaded for summary parsing")

    monkeypatch.setattr(audit, "_load_eval_module", fail_if_loaded)

    args = audit.parse_args(["--mode", "summarize-five-seed"])

    assert args.mode == "summarize-five-seed"


def test_audit_mode_names_are_stable_protocol_labels():
    from tools import pre_submission_audit as audit

    assert audit.RANDOM_STRESS_MODE_NAME == "random_eval_encoder_stress"
    assert audit.MATCHED_ENCODER_MODE_NAME == "matched_encoder_checkpoint_audit"


def test_parse_args_accepts_matched_encoder_audit_mode():
    from tools import pre_submission_audit as audit

    args = audit.parse_args(["--mode", "matched-encoder-audit", "--method", "m7_pta", "--seed", "42"])

    assert args.mode == "matched-encoder-audit"
    assert args.method == "m7_pta"
    assert args.seed == 42


def test_encoder_gate_rejects_all_failed_zero_transfer_rows():
    from tools import pre_submission_audit as audit

    rows = [
        {"mean_transfer": 0.0, "n_failed_episodes": 3},
        {"mean_transfer": 0.0, "n_failed_episodes": 3},
        {"mean_transfer": 0.0, "n_failed_episodes": 3},
    ]

    gate = audit.encoder_sensitivity_gate(rows, max_transfer_range_pp=5.0)

    assert gate == {
        "passes": False,
        "transfer_range_pp": 0.0,
        "total_failed_episodes": 9,
        "reasons": ["encoder sensitivity eval had failed episodes"],
    }


class _FakeEvalEnv:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakePPO:
    loaded = []

    @classmethod
    def load(cls, path, device="auto"):
        cls.loaded.append((path, device))
        return {"model_path": path, "device": device}


def _metrics(**overrides):
    metrics = {
        "mean_reward": 10.0,
        "std_reward": 0.0,
        "mean_transfer": 0.7,
        "std_transfer": 0.0,
        "mean_spill": 0.2,
        "std_spill": 0.0,
        "success_rate": 1.0,
        "n_failed_episodes": 0,
    }
    metrics.update(overrides)
    return metrics


def test_encoder_sensitivity_uses_random_stress_resolver_and_payload_mode(monkeypatch, tmp_path):
    from tools import pre_submission_audit as audit

    checkpoint = tmp_path / "best_model.zip"
    checkpoint.write_text("policy", encoding="utf-8")
    resolver_calls = []
    make_env_calls = []

    def resolve_m7_belief_encoder(policy_path, ablation, encoder_mode, encoder_seed, expected):
        resolver_calls.append((policy_path, ablation, encoder_mode, encoder_seed, expected))
        return f"encoder-{encoder_seed}", {
            "encoder_mode": encoder_mode,
            "encoder_seed": str(encoder_seed),
            "encoder_artifact": "",
            "encoder_sha256": "",
            "policy_checkpoint": "checkpoints/m7_pta_seed42/best/best_model.zip",
            "policy_sha256": "policy-sha",
            "protocol": audit.RANDOM_STRESS_MODE_NAME,
        }

    def make_eval_env(**kwargs):
        make_env_calls.append(kwargs)
        return _FakeEvalEnv()

    fake_eval = types.SimpleNamespace(
        SPLITS={"ood_elastoplastic": {"name": "split"}},
        resolve_checkpoint_path=lambda project_root, pattern, seed: checkpoint,
        resolve_m7_belief_encoder=resolve_m7_belief_encoder,
        make_eval_env=make_eval_env,
        evaluate_one=lambda model, env, n_episodes, deterministic=True: _metrics(mean_transfer=0.55),
    )
    monkeypatch.setattr(audit, "_load_eval_module", lambda: fake_eval)
    monkeypatch.setattr(audit, "_load_ppo", lambda: _FakePPO)

    args = audit.parse_args(
        [
            "--mode",
            "encoder-sensitivity",
            "--method",
            "m7_pta",
            "--seed",
            "42",
            "--split",
            "ood_elastoplastic",
            "--encoder-seeds",
            "11",
            "--output-dir",
            str(tmp_path),
        ]
    )

    audit.run_encoder_sensitivity(args)

    assert resolver_calls == [
        (
            checkpoint,
            "none",
            "random-stress",
            11,
            {"method": "m7_pta", "seed": 42, "ablation": "none", "latent_dim": 16, "n_probes": 3},
        )
    ]
    assert make_env_calls[0]["belief_encoder"] == "encoder-11"
    payload = json.loads((tmp_path / "audit_encoder_m7_pta_s42_ood_elastoplastic.json").read_text())
    assert payload["mode"] == audit.RANDOM_STRESS_MODE_NAME
    assert payload["rows"][0]["encoder_mode"] == "random-stress"
    assert payload["rows"][0]["protocol"] == audit.RANDOM_STRESS_MODE_NAME


def test_matched_encoder_audit_uses_matched_resolver_and_writes_named_payload(monkeypatch, tmp_path):
    from tools import pre_submission_audit as audit

    checkpoint = tmp_path / "best_model.zip"
    checkpoint.write_text("policy", encoding="utf-8")
    resolver_calls = []
    make_env_calls = []

    def resolve_m7_belief_encoder(policy_path, ablation, encoder_mode, encoder_seed, expected):
        resolver_calls.append((policy_path, ablation, encoder_mode, encoder_seed, expected))
        return "matched-encoder", {
            "encoder_mode": "matched",
            "encoder_seed": "",
            "encoder_artifact": "checkpoints/m7_pta_seed42/best/belief_encoder.pt",
            "encoder_sha256": "encoder-sha",
            "policy_checkpoint": "checkpoints/m7_pta_seed42/best/best_model.zip",
            "policy_sha256": "policy-sha",
            "protocol": "matched_encoder_v1",
        }

    def make_eval_env(**kwargs):
        make_env_calls.append(kwargs)
        return _FakeEvalEnv()

    fake_eval = types.SimpleNamespace(
        SPLITS={"ood_elastoplastic": {"name": "split"}},
        resolve_checkpoint_path=lambda project_root, pattern, seed: checkpoint,
        resolve_m7_belief_encoder=resolve_m7_belief_encoder,
        make_eval_env=make_eval_env,
        evaluate_one=lambda model, env, n_episodes, deterministic=True: _metrics(mean_transfer=0.88),
    )
    monkeypatch.setattr(audit, "_load_eval_module", lambda: fake_eval)
    monkeypatch.setattr(audit, "_load_ppo", lambda: _FakePPO)

    args = audit.parse_args(
        [
            "--mode",
            "matched-encoder-audit",
            "--method",
            "m7_pta",
            "--seed",
            "42",
            "--split",
            "ood_elastoplastic",
            "--output-dir",
            str(tmp_path),
        ]
    )

    audit.run_matched_encoder_audit(args)

    assert resolver_calls == [
        (
            checkpoint,
            "none",
            "matched",
            None,
            {"method": "m7_pta", "seed": 42, "ablation": "none", "latent_dim": 16, "n_probes": 3},
        )
    ]
    assert make_env_calls[0]["belief_encoder"] == "matched-encoder"
    payload_path = tmp_path / "audit_matched_encoder_m7_pta_s42_ood_elastoplastic.json"
    payload = json.loads(payload_path.read_text())
    assert payload["mode"] == audit.MATCHED_ENCODER_MODE_NAME
    assert payload["encoder_mode"] == "matched"
    assert payload["encoder_sha256"] == "encoder-sha"
    assert payload["mean_transfer"] == 0.88
    assert payload["passes"] is True
    assert payload["total_failed_episodes"] == 0
    assert payload["reasons"] == []
