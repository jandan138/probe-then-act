from pathlib import Path


def test_detect_run_completion_from_final_checkpoint(tmp_path):
    from pta.scripts.cron_aris_orchestrator import detect_run_completion

    run_dir = tmp_path / "checkpoints" / "m1_reactive_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "scoop_transfer_teacher_final.zip").write_text("ok")

    result = detect_run_completion(
        checkpoint_dir=run_dir,
        final_name="scoop_transfer_teacher_final.zip",
    )

    assert result.completed is True
    assert result.final_checkpoint == run_dir / "scoop_transfer_teacher_final.zip"


def test_detect_active_process_from_ps_output():
    from pta.scripts.cron_aris_orchestrator import parse_ps_output

    output = "21934 40730 python pta/scripts/train_baselines.py --method m8 --seed 42"
    processes = parse_ps_output(output)

    assert processes[0]["pid"] == 21934
    assert "train_baselines.py" in processes[0]["cmd"]


def test_choose_latest_resume_checkpoint(tmp_path):
    from pta.scripts.cron_aris_orchestrator import choose_latest_resume_checkpoint

    ckpt_dir = tmp_path / "checkpoints" / "m8_teacher_seed42"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "scoop_transfer_teacher_50000_steps.zip").write_text("a")
    (ckpt_dir / "scoop_transfer_teacher_final.zip").write_text("b")

    result = choose_latest_resume_checkpoint(ckpt_dir)

    assert result.name == "scoop_transfer_teacher_final.zip"
