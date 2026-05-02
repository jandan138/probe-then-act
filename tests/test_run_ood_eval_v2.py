import sys
import types

import pytest


class _FakeModel:
    def predict(self, obs, deterministic=True):
        return 0, None


class _FakeEnv:
    def __init__(self):
        self._done = False

    def reset(self):
        self._done = False
        return 0, {}

    def step(self, action):
        if self._done:
            raise RuntimeError("step called after done")
        self._done = True
        return (
            0,
            123.0,
            True,
            False,
            {
                "success_rate": 1.0,
                "transfer_efficiency": 0.42,
                "spill_ratio": 0.11,
            },
        )


class _OneCrashEnv:
    def __init__(self):
        self._episode = -1
        self._done = False

    def reset(self):
        self._episode += 1
        self._done = False
        return 0, {}

    def step(self, action):
        if self._episode == 0:
            raise RuntimeError("simulator produced nan")
        if self._done:
            raise RuntimeError("step called after done")
        self._done = True
        return (
            0,
            100.0,
            True,
            False,
            {
                "success_rate": 1.0,
                "transfer_efficiency": 0.5,
                "spill_ratio": 0.2,
            },
        )


class _ResetCrashEnv(_OneCrashEnv):
    def reset(self):
        self._episode += 1
        if self._episode == 0:
            raise RuntimeError("Invalid constraint forces causing 'nan'")
        self._done = False
        return 0, {}


class _BuggyEnv:
    def reset(self):
        return 0, {}

    def step(self, action):
        raise RuntimeError("programming bug")


class _BuggyNanSubstringEnv:
    def reset(self):
        return 0, {}

    def step(self, action):
        raise RuntimeError("simulator produced banana split bug")


def _result_row(**overrides):
    row = {
        "method": "m1_reactive",
        "seed": 42,
        "split": "id_sand",
        "encoder_mode": "policy-only",
        "encoder_seed": "",
        "encoder_sha256": "",
        "policy_sha256": "",
        "protocol": "policy_only",
        "mean_reward": 1.0,
        "std_reward": 0.0,
        "mean_transfer": 0.2,
        "std_transfer": 0.0,
        "mean_spill": 0.1,
        "std_spill": 0.0,
        "success_rate": 1.0,
        "n_failed_episodes": 0,
    }
    row.update(overrides)
    return row


def test_evaluate_one_reads_task_metric_keys():
    from pta.scripts.run_ood_eval_v2 import evaluate_one

    metrics = evaluate_one(_FakeModel(), _FakeEnv(), n_episodes=3)

    assert metrics["success_rate"] == 1.0
    assert metrics["mean_transfer"] == 0.42
    assert metrics["mean_spill"] == 0.11


def test_evaluate_one_counts_crashed_episode_as_failure_and_continues():
    from pta.scripts.run_ood_eval_v2 import evaluate_one

    metrics = evaluate_one(_FakeModel(), _OneCrashEnv(), n_episodes=2)

    assert metrics["success_rate"] == 0.5
    assert metrics["mean_transfer"] == 0.25
    assert metrics["mean_spill"] == 0.6
    assert metrics["n_failed_episodes"] == 1


def test_evaluate_one_counts_reset_nan_as_failure_and_continues():
    from pta.scripts.run_ood_eval_v2 import evaluate_one

    metrics = evaluate_one(_FakeModel(), _ResetCrashEnv(), n_episodes=2)

    assert metrics["success_rate"] == 0.5
    assert metrics["n_failed_episodes"] == 1


def test_evaluate_one_reraises_non_nan_errors():
    from pta.scripts.run_ood_eval_v2 import evaluate_one

    with pytest.raises(RuntimeError, match="programming bug"):
        evaluate_one(_FakeModel(), _BuggyEnv(), n_episodes=1)


def test_evaluate_one_reraises_non_simulator_errors_containing_nan():
    from pta.scripts.run_ood_eval_v2 import evaluate_one

    with pytest.raises(RuntimeError, match="banana split bug"):
        evaluate_one(_FakeModel(), _BuggyNanSubstringEnv(), n_episodes=1)


def test_aggregate_results_reports_failed_episode_counts():
    from pta.scripts.run_ood_eval_v2 import aggregate_results

    rows = [
        {
            "method": "m1_reactive",
            "seed": 0,
            "split": "ood_snow",
            "mean_reward": 1.0,
            "mean_transfer": 0.2,
            "mean_spill": 0.1,
            "success_rate": 0.0,
            "n_failed_episodes": 1,
        },
        {
            "method": "m1_reactive",
            "seed": 1,
            "split": "ood_snow",
            "mean_reward": 3.0,
            "mean_transfer": 0.4,
            "mean_spill": 0.3,
            "success_rate": 1.0,
            "n_failed_episodes": 0,
        },
    ]

    agg_rows = aggregate_results(rows)

    assert agg_rows[0]["n_failed_episodes_sum"] == 1
    assert agg_rows[0]["n_failed_episodes_mean"] == 0.5


def test_result_key_uses_encoder_protocol_identity_fields():
    from pta.scripts.run_ood_eval_v2 import result_key

    row = {
        "method": "m7_pta",
        "seed": 42,
        "split": "ood_snow",
        "encoder_mode": "matched",
        "encoder_seed": "",
        "encoder_sha256": "encoder-digest",
        "policy_sha256": "policy-digest",
        "protocol": "matched_encoder_v1",
    }

    assert result_key(row) == (
        "m7_pta",
        42,
        "ood_snow",
        "matched",
        "",
        "encoder-digest",
        "policy-digest",
        "matched_encoder_v1",
    )


def test_parse_args_defaults_to_matched_encoder_without_random_seed():
    from pta.scripts.run_ood_eval_v2 import parse_args

    args = parse_args([])

    assert args.m7_encoder_mode == "matched"
    assert args.m7_random_encoder_seed is None


def test_resolve_m7_matched_rejects_policy_only_checkpoint(tmp_path):
    from pta.scripts.run_ood_eval_v2 import resolve_m7_belief_encoder

    policy_path = tmp_path / "best_model.zip"
    policy_path.write_text("policy only", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing matched encoder artifact"):
        resolve_m7_belief_encoder(
            policy_path,
            ablation="none",
            encoder_mode="matched",
            encoder_seed=None,
            expected={"method": "m7_pta", "seed": 42, "ablation": "none"},
        )


@pytest.mark.parametrize("ablation", ["no_probe", "no_belief"])
def test_resolve_m7_ablations_use_zero_z_identity_without_sidecar(tmp_path, ablation):
    from pta.scripts.run_ood_eval_v2 import resolve_m7_belief_encoder

    policy_path = tmp_path / "m7_pta_final.zip"
    policy_path.write_text("policy only", encoding="utf-8")

    encoder, identity = resolve_m7_belief_encoder(
        policy_path,
        ablation=ablation,
        encoder_mode="matched",
        encoder_seed=None,
        expected={"method": "m7_pta", "seed": 42, "ablation": ablation},
    )

    assert encoder is None
    assert identity["encoder_mode"] == "zero-z"
    assert identity["protocol"] == "ablation_zero_z"
    assert identity["encoder_seed"] == ""
    assert identity["encoder_sha256"] == ""


def test_resolve_m7_random_stress_requires_seed_and_seeds_rngs(monkeypatch, tmp_path):
    from pta.scripts import run_ood_eval_v2

    policy_path = tmp_path / "best_model.zip"
    policy_path.write_text("policy", encoding="utf-8")
    calls = []

    monkeypatch.setattr(
        run_ood_eval_v2.random,
        "seed",
        lambda seed: calls.append(("random", seed)),
    )
    monkeypatch.setattr(
        run_ood_eval_v2.np.random,
        "seed",
        lambda seed: calls.append(("numpy", seed)),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        types.SimpleNamespace(manual_seed=lambda seed: calls.append(("torch", seed))),
    )

    with pytest.raises(ValueError, match="requires --m7-random-encoder-seed"):
        run_ood_eval_v2.resolve_m7_belief_encoder(
            policy_path,
            ablation="none",
            encoder_mode="random-stress",
            encoder_seed=None,
            expected={},
        )

    encoder, identity = run_ood_eval_v2.resolve_m7_belief_encoder(
        policy_path,
        ablation="none",
        encoder_mode="random-stress",
        encoder_seed=123,
        expected={},
    )

    assert encoder is None
    assert identity["encoder_mode"] == "random-stress"
    assert identity["encoder_seed"] == "123"
    assert identity["protocol"] == "random_stress"
    assert ("random", 123) in calls
    assert ("numpy", 123) in calls
    assert ("torch", 123) in calls


def test_make_eval_env_passes_loaded_encoder_through_probe_wrapper(monkeypatch):
    from pta.scripts import run_ood_eval_v2

    captured = {}
    loaded_encoder = object()

    class FakeBaseEnv:
        def __init__(self, task_config, scene_config):
            self.task_config = task_config
            self.scene_config = scene_config
            self.observation_space = types.SimpleNamespace(shape=(8,))

        def reset(self, seed=None):
            captured["reset_seed"] = seed
            return 0, {}

    class FakeJointResidualWrapper:
        def __init__(self, env, residual_scale, trajectory):
            self.env = env
            self.residual_scale = residual_scale
            self.trajectory = trajectory
            self.observation_space = env.observation_space

    class FakeProbePhaseWrapper:
        def __init__(self, env, **kwargs):
            captured["probe_kwargs"] = kwargs
            self.env = env

        def reset(self, seed=None):
            return self.env.env.reset(seed=seed)

    monkeypatch.setattr(run_ood_eval_v2, "_load_genesis_gym_wrapper", lambda: FakeBaseEnv)
    monkeypatch.setattr(
        run_ood_eval_v2,
        "_load_joint_residual_wrapper",
        lambda: FakeJointResidualWrapper,
    )
    monkeypatch.setattr(
        run_ood_eval_v2,
        "_load_probe_phase_wrapper",
        lambda: FakeProbePhaseWrapper,
    )

    env = run_ood_eval_v2.make_eval_env(
        split_config={"particle_material": "sand", "particle_params": {}},
        use_privileged=False,
        use_m7_env=True,
        belief_encoder=loaded_encoder,
        seed=99,
    )

    assert isinstance(env, FakeProbePhaseWrapper)
    assert captured["probe_kwargs"]["belief_encoder"] is loaded_encoder
    assert captured["reset_seed"] == 99


def test_resolve_checkpoint_path_prefers_explicit_final_pattern(tmp_path):
    from pta.scripts.run_ood_eval_v2 import resolve_checkpoint_path

    run_dir = tmp_path / "checkpoints" / "m7_pta_noprobe_seed0"
    (run_dir / "best").mkdir(parents=True)
    (run_dir / "best" / "best_model.zip").write_text("best", encoding="utf-8")
    (run_dir / "m7_pta_final.zip").write_text("final", encoding="utf-8")

    resolved = resolve_checkpoint_path(
        tmp_path,
        "checkpoints/m7_pta_noprobe_seed{seed}/m7_pta_final",
        seed=0,
    )

    assert resolved == run_dir / "m7_pta_final.zip"


def test_append_result_row_writes_header_once(tmp_path):
    from pta.scripts.run_ood_eval_v2 import RESULT_FIELDNAMES, append_result_row

    path = tmp_path / "ood_eval_per_seed.csv"
    row = _result_row()

    append_result_row(path, row)
    append_result_row(path, {**row, "seed": 0})

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0].split(",") == RESULT_FIELDNAMES
    assert len(lines) == 3


def test_write_result_rows_replaces_existing_file(tmp_path):
    from pta.scripts.run_ood_eval_v2 import write_result_rows

    path = tmp_path / "ood_eval_per_seed.csv"
    path.write_text("stale\n", encoding="utf-8")

    write_result_rows(path, [_result_row(seed=42), _result_row(seed=0)])

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    assert "stale" not in path.read_text(encoding="utf-8")


def test_load_completed_rows_reads_existing_csv(tmp_path):
    from pta.scripts.run_ood_eval_v2 import append_result_row, load_completed_rows

    path = tmp_path / "ood_eval_per_seed.csv"
    row = _result_row()
    append_result_row(path, row)

    rows, keys = load_completed_rows(path, resume=True)

    assert rows == [row]
    assert keys == {
        (
            "m1_reactive",
            42,
            "id_sand",
            "policy-only",
            "",
            "",
            "",
            "policy_only",
        )
    }


def test_load_completed_rows_ignores_file_when_resume_disabled(tmp_path):
    from pta.scripts.run_ood_eval_v2 import append_result_row, load_completed_rows

    path = tmp_path / "ood_eval_per_seed.csv"
    append_result_row(path, _result_row())

    rows, keys = load_completed_rows(path, resume=False)

    assert rows == []
    assert keys == set()


def test_load_completed_rows_skips_nonfinite_rows(tmp_path):
    from pta.scripts.run_ood_eval_v2 import append_result_row, load_completed_rows

    path = tmp_path / "ood_eval_per_seed.csv"
    append_result_row(path, _result_row(mean_reward=float("nan")))

    rows, keys = load_completed_rows(path, resume=True)

    assert rows == []
    assert keys == set()


def test_load_completed_rows_skips_malformed_rows(tmp_path):
    from pta.scripts.run_ood_eval_v2 import load_completed_rows

    path = tmp_path / "ood_eval_per_seed.csv"
    path.write_text("method,seed,split\nm1_reactive,42,id_sand\n", encoding="utf-8")

    rows, keys = load_completed_rows(path, resume=True)

    assert rows == []
    assert keys == set()


def test_load_completed_rows_skips_invalid_failed_episode_counts(tmp_path):
    from pta.scripts.run_ood_eval_v2 import append_result_row, load_completed_rows

    path = tmp_path / "ood_eval_per_seed.csv"
    append_result_row(path, _result_row(n_failed_episodes=float("inf")))
    append_result_row(path, _result_row(seed=0, n_failed_episodes=1.5))

    rows, keys = load_completed_rows(path, resume=True)

    assert rows == []
    assert keys == set()


def test_write_aggregate_results_creates_main_results_csv(tmp_path):
    from pta.scripts.run_ood_eval_v2 import write_aggregate_results

    path = tmp_path / "main_results.csv"

    agg_rows = write_aggregate_results(path, [_result_row(mean_reward=2.0)])

    assert path.exists()
    assert agg_rows[0]["method"] == "m1_reactive"
    assert agg_rows[0]["n_failed_episodes_sum"] == 0


def test_csv_writers_use_lf_line_endings(tmp_path):
    from pta.scripts.run_ood_eval_v2 import (
        append_result_row,
        write_aggregate_results,
        write_result_rows,
    )

    appended_path = tmp_path / "appended.csv"
    replaced_path = tmp_path / "replaced.csv"
    aggregate_path = tmp_path / "aggregate.csv"
    row = _result_row()

    append_result_row(appended_path, row)
    write_result_rows(replaced_path, [row])
    write_aggregate_results(aggregate_path, [row])

    for path in [appended_path, replaced_path, aggregate_path]:
        assert b"\r\n" not in path.read_bytes()


def test_prepare_result_files_removes_existing_outputs_when_resume_disabled(tmp_path):
    from pta.scripts.run_ood_eval_v2 import prepare_result_files

    per_seed_path = tmp_path / "ood_eval_per_seed.csv"
    main_results_path = tmp_path / "main_results.csv"
    per_seed_path.write_text("old per-seed", encoding="utf-8")
    main_results_path.write_text("old aggregate", encoding="utf-8")

    prepare_result_files(per_seed_path, main_results_path, resume=False)

    assert not per_seed_path.exists()
    assert not main_results_path.exists()


def test_prepare_result_files_keeps_existing_outputs_when_resume_enabled(tmp_path):
    from pta.scripts.run_ood_eval_v2 import prepare_result_files

    per_seed_path = tmp_path / "ood_eval_per_seed.csv"
    main_results_path = tmp_path / "main_results.csv"
    per_seed_path.write_text("old per-seed", encoding="utf-8")
    main_results_path.write_text("old aggregate", encoding="utf-8")

    prepare_result_files(per_seed_path, main_results_path, resume=True)

    assert per_seed_path.exists()
    assert main_results_path.exists()


def test_parse_args_resumes_by_default(monkeypatch):
    from pta.scripts.run_ood_eval_v2 import parse_args

    monkeypatch.setattr(sys, "argv", ["run_ood_eval_v2.py"])
    args = parse_args()

    assert args.resume is True


def test_parse_args_uses_hotfix_residual_scale_default(monkeypatch):
    from pta.scripts.run_ood_eval_v2 import parse_args

    monkeypatch.setattr(sys, "argv", ["run_ood_eval_v2.py"])
    args = parse_args()

    assert args.residual_scale == 0.05


def test_optional_ablation_methods_cover_three_approved_seeds():
    from pta.scripts import run_ood_eval_v2

    assert run_ood_eval_v2.METHODS["m7_noprobe"]["seeds"] == [42, 0, 1]
    assert run_ood_eval_v2.METHODS["m7_nobelief"]["seeds"] == [42, 0, 1]
