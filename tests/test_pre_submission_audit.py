import csv
import math

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
