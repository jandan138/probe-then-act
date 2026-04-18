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


def test_parse_ps_output_skips_header_malformed_and_empty_lines():
    from pta.scripts.cron_aris_orchestrator import parse_ps_output

    output = """\
PID ELAPSED CMD
21934 40730 python pta/scripts/train_baselines.py --method m8 --seed 42
malformed

22001 15 python pta/scripts/train_m7.py --seed 0
"""

    processes = parse_ps_output(output)

    assert processes == [
        {
            "pid": 21934,
            "elapsed": 40730,
            "cmd": "python pta/scripts/train_baselines.py --method m8 --seed 42",
        },
        {
            "pid": 22001,
            "elapsed": 15,
            "cmd": "python pta/scripts/train_m7.py --seed 0",
        },
    ]


def test_parse_ps_output_returns_empty_list_for_empty_input():
    from pta.scripts.cron_aris_orchestrator import parse_ps_output

    assert parse_ps_output("") == []


def test_choose_latest_resume_checkpoint_prefers_final_checkpoint(tmp_path):
    from pta.scripts.cron_aris_orchestrator import choose_latest_resume_checkpoint

    ckpt_dir = tmp_path / "checkpoints" / "m8_teacher_seed42"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "scoop_transfer_teacher_50000_steps.zip").write_text("a")
    (ckpt_dir / "scoop_transfer_teacher_final.zip").write_text("b")

    result = choose_latest_resume_checkpoint(ckpt_dir)

    assert result.name == "scoop_transfer_teacher_final.zip"


def test_choose_latest_resume_checkpoint_uses_highest_step_number(tmp_path):
    from pta.scripts.cron_aris_orchestrator import choose_latest_resume_checkpoint

    ckpt_dir = tmp_path / "checkpoints" / "m8_teacher_seed0"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "scoop_transfer_teacher_9_steps.zip").write_text("a")
    (ckpt_dir / "scoop_transfer_teacher_100_steps.zip").write_text("b")
    (ckpt_dir / "scoop_transfer_teacher_1000_steps.zip").write_text("c")

    result = choose_latest_resume_checkpoint(ckpt_dir)

    assert result.name == "scoop_transfer_teacher_1000_steps.zip"


def test_choose_latest_resume_checkpoint_returns_none_for_empty_directory(tmp_path):
    from pta.scripts.cron_aris_orchestrator import choose_latest_resume_checkpoint

    ckpt_dir = tmp_path / "checkpoints" / "m8_teacher_seed1"
    ckpt_dir.mkdir(parents=True)

    assert choose_latest_resume_checkpoint(ckpt_dir) is None


def test_decide_next_step_returns_running_when_process_active():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": True, "completed": False},
        "m1": {"running": False, "completed_seeds": []},
        "m7": {"running": False, "completed_seeds": []},
        "ood_eval": {"completed": False},
    }

    decision = decide_next_step(state)

    assert decision["action"] == "wait"
    assert decision["stage"] == "m8"


def test_decide_next_step_launches_m8_resume_when_not_completed():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": False, "completed": False},
        "m1": {"running": False, "completed_seeds": []},
        "m7": {"running": False, "completed_seeds": []},
        "ood_eval": {"completed": False},
    }

    assert decide_next_step(state) == {"action": "launch_m8_resume"}


def test_decide_next_step_prioritizes_m8_resume_over_running_m1():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": False, "completed": False},
        "m1": {"running": True, "completed_seeds": []},
        "m7": {"running": False, "completed_seeds": []},
        "ood_eval": {"completed": False},
    }

    assert decide_next_step(state) == {"action": "launch_m8_resume"}


def test_decide_next_step_waits_on_running_m1_after_m8_completion():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": False, "completed": True},
        "m1": {"running": True, "completed_seeds": []},
        "m7": {"running": False, "completed_seeds": []},
        "ood_eval": {"completed": False},
    }

    assert decide_next_step(state) == {"action": "wait", "stage": "m1"}


def test_decide_next_step_launches_first_missing_m1_seed():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": False, "completed": True},
        "m1": {"running": False, "completed_seeds": [42]},
        "m7": {"running": False, "completed_seeds": []},
        "ood_eval": {"completed": False},
    }

    decision = decide_next_step(state)

    assert decision == {"action": "launch_m1", "seed": 0}


def test_decide_next_step_prioritizes_missing_m1_over_running_m7():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": False, "completed": True},
        "m1": {"running": False, "completed_seeds": [42]},
        "m7": {"running": True, "completed_seeds": []},
        "ood_eval": {"completed": False},
    }

    assert decide_next_step(state) == {"action": "launch_m1", "seed": 0}


def test_decide_next_step_launches_first_missing_m7_seed():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": False, "completed": True},
        "m1": {"running": False, "completed_seeds": [42, 0, 1]},
        "m7": {"running": False, "completed_seeds": [42]},
        "ood_eval": {"completed": False},
    }

    assert decide_next_step(state) == {"action": "launch_m7", "seed": 0}


def test_decide_next_step_waits_on_running_m7_after_m1_completion():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": False, "completed": True},
        "m1": {"running": False, "completed_seeds": [42, 0, 1]},
        "m7": {"running": True, "completed_seeds": []},
        "ood_eval": {"completed": False},
    }

    assert decide_next_step(state) == {"action": "wait", "stage": "m7"}


def test_decide_next_step_runs_ood_eval_after_training_stages_complete():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": False, "completed": True},
        "m1": {"running": False, "completed_seeds": [42, 0, 1]},
        "m7": {"running": False, "completed_seeds": [42, 0, 1]},
        "ood_eval": {"completed": False},
    }

    assert decide_next_step(state) == {"action": "run_ood_eval"}


def test_decide_next_step_hands_off_after_ood_eval():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": False, "completed": True},
        "m1": {"running": False, "completed_seeds": [42, 0, 1]},
        "m7": {"running": False, "completed_seeds": [42, 0, 1]},
        "ood_eval": {"completed": True},
    }

    decision = decide_next_step(state)

    assert decision == {"action": "handoff_aris"}


def test_first_missing_seed_returns_none_for_empty_expected():
    from pta.scripts.cron_aris_orchestrator import _first_missing_seed

    assert _first_missing_seed(done=[42, 0, 1], expected=[]) is None


def test_first_missing_seed_ignores_extra_completed_seeds():
    from pta.scripts.cron_aris_orchestrator import _first_missing_seed

    assert _first_missing_seed(done=[99, 42, 7], expected=[42, 0, 1]) == 0
