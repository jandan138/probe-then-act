from pathlib import Path

import pytest

from pta.scripts import resume_m7


def test_latest_step_checkpoint_selects_highest_numeric_step(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    for name in [
        "m7_pta_50000_steps.zip",
        "m7_pta_400000_steps.zip",
        "m7_pta_150000_steps.zip",
        "best_model.zip",
        "m7_pta_final.zip",
    ]:
        (checkpoint_dir / name).write_text("checkpoint", encoding="utf-8")

    selected = resume_m7.latest_step_checkpoint(checkpoint_dir)

    assert selected == checkpoint_dir / "m7_pta_400000_steps.zip"


def test_latest_step_checkpoint_rejects_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError, match="No step checkpoints"):
        resume_m7.latest_step_checkpoint(tmp_path)


def test_remaining_timesteps_uses_target_minus_loaded_steps():
    assert resume_m7.remaining_timesteps(current_timesteps=400000, target_timesteps=500000) == 100000
    assert resume_m7.remaining_timesteps(current_timesteps=500000, target_timesteps=500000) == 0


def test_remaining_timesteps_rejects_target_before_checkpoint():
    with pytest.raises(ValueError, match="already past target"):
        resume_m7.remaining_timesteps(current_timesteps=550000, target_timesteps=500000)


def test_default_run_name_matches_train_m7_conventions():
    assert resume_m7.default_run_name("no_probe", 0) == "m7_pta_noprobe_seed0"
    assert resume_m7.default_run_name("no_belief", 42) == "m7_pta_nobelief_seed42"
    assert resume_m7.default_run_name("none", 1) == "m7_pta_seed1"
