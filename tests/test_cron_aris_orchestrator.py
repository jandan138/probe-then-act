import json
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


def test_write_handoff_files_creates_machine_and_human_records(tmp_path):
    from pta.scripts.cron_aris_orchestrator import write_handoff_files

    state = {
        "m8": {"completed": True},
        "m1": {"completed_seeds": [42, 0, 1]},
        "m7": {"completed_seeds": [42, 0, 1]},
        "ood_eval": {"completed": True},
        "aris": {"blocked": False},
    }

    records = write_handoff_files(tmp_path, state)

    assert json.loads(records["json"].read_text(encoding="utf-8")) == {
        "aris": {"blocked": False, "ready": True},
        "m1": {"completed_seeds": [42, 0, 1]},
        "m7": {"completed_seeds": [42, 0, 1]},
        "m8": {"completed": True},
        "ood_eval": {"completed": True},
    }
    assert records["summary"].read_text(encoding="utf-8") == (
        "# ARIS Handoff Ready\n\n"
        "- M8 complete: True\n"
        "- M1 seeds: [42, 0, 1]\n"
        "- M7 seeds: [42, 0, 1]\n"
        "- OOD eval complete: True\n"
    )


def test_decide_next_step_stays_blocked_when_handoff_failed():
    from pta.scripts.cron_aris_orchestrator import decide_next_step

    state = {
        "m8": {"running": False, "completed": True},
        "m1": {"running": False, "completed_seeds": [42, 0, 1]},
        "m7": {"running": False, "completed_seeds": [42, 0, 1]},
        "ood_eval": {"completed": True},
        "aris": {"blocked": True},
    }

    decision = decide_next_step(state)

    assert decision == {"action": "blocked", "stage": "aris"}


def test_first_missing_seed_returns_none_for_empty_expected():
    from pta.scripts.cron_aris_orchestrator import _first_missing_seed

    assert _first_missing_seed(done=[42, 0, 1], expected=[]) is None


def test_first_missing_seed_ignores_extra_completed_seeds():
    from pta.scripts.cron_aris_orchestrator import _first_missing_seed

    assert _first_missing_seed(done=[99, 42, 7], expected=[42, 0, 1]) == 0


def test_build_m1_command_uses_hotfix_scale():
    from pta.scripts.cron_aris_orchestrator import build_command

    command = build_command({"action": "launch_m1", "seed": 42})

    assert "train_baselines.py --method m1 --seed 42" in command
    assert "--residual-scale 0.05" in command


def test_build_m7_command_uses_hotfix_scale():
    from pta.scripts.cron_aris_orchestrator import build_command

    command = build_command({"action": "launch_m7", "seed": 0})

    assert "train_m7.py --seed 0" in command
    assert "--residual-scale 0.05" in command


def test_build_ood_eval_command_uses_corrected_script_defaults():
    from pta.scripts.cron_aris_orchestrator import build_command

    command = build_command({"action": "run_ood_eval"})

    assert "run_ood_eval_v2.py" in command
    assert "--residual-scale 0.05" in command


def test_launch_detached_uses_direct_process_launch(tmp_path, monkeypatch):
    from pta.scripts.cron_aris_orchestrator import launch_detached

    recorded: dict[str, object] = {}

    class DummyProcess:
        pid = 4321

    def fake_popen(args, cwd, stdout, stderr, stdin, start_new_session):
        recorded["args"] = args
        recorded["cwd"] = cwd
        recorded["stdout_name"] = stdout.name
        recorded["stderr_name"] = stderr.name
        recorded["stdin"] = stdin
        recorded["start_new_session"] = start_new_session
        return DummyProcess()

    monkeypatch.setattr(
        "pta.scripts.cron_aris_orchestrator.subprocess.Popen", fake_popen
    )

    log_path = tmp_path / "logs" / "orchestrator.log"
    pid = launch_detached("python pta/scripts/train_m7.py --seed 0", log_path, tmp_path)

    assert pid == 4321
    assert log_path.parent.is_dir()
    assert recorded["cwd"] == tmp_path
    assert recorded["stdout_name"] == str(log_path)
    assert recorded["stderr_name"] == str(log_path)
    assert recorded["args"] == ["python", "pta/scripts/train_m7.py", "--seed", "0"]
    assert recorded["stdin"] is __import__("subprocess").DEVNULL
    assert recorded["start_new_session"] is True


def test_launch_detached_returns_spawned_process_pid_not_wrapper_pid(
    tmp_path, monkeypatch
):
    from pta.scripts.cron_aris_orchestrator import launch_detached

    class DummyProcess:
        def __init__(self, pid):
            self.pid = pid

    wrapper_pid = 111
    child_pid = 9876

    def fake_popen(args, cwd, stdout, stderr, stdin, start_new_session):
        assert args == [
            "python",
            "pta/scripts/train_baselines.py",
            "--method",
            "m1",
            "--seed",
            "42",
        ]
        return DummyProcess(child_pid)

    monkeypatch.setattr(
        "pta.scripts.cron_aris_orchestrator.subprocess.Popen", fake_popen
    )

    log_path = tmp_path / "logs" / "orchestrator.log"
    pid = launch_detached(
        "python pta/scripts/train_baselines.py --method m1 --seed 42",
        log_path,
        tmp_path,
    )

    assert pid == child_pid
    assert pid != wrapper_pid


def test_save_and_load_state_roundtrip(tmp_path):
    from pta.scripts.cron_aris_orchestrator import load_state, save_state

    state_path = tmp_path / "aris_state.json"
    state = {"stage": "m1", "m1": {"completed_seeds": [42]}}
    save_state(state_path, state)

    loaded = load_state(state_path)

    assert loaded == state


def test_load_state_defaults_when_missing(tmp_path):
    from pta.scripts.cron_aris_orchestrator import load_state

    loaded = load_state(tmp_path / "missing.json")

    assert loaded["stage"] == "bootstrap"
    assert loaded["m1"]["completed_seeds"] == []


def test_reconcile_state_marks_m1_seed_complete_from_final_zip(tmp_path):
    from pta.scripts.cron_aris_orchestrator import reconcile_state

    project_root = tmp_path
    run_dir = project_root / "checkpoints" / "m1_reactive_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "scoop_transfer_teacher_final.zip").write_text("ok")

    state = reconcile_state(project_root=project_root, ps_output="")

    assert 42 in state["m1"]["completed_seeds"]


def test_reconcile_state_marks_m8_complete_from_final_zip(tmp_path):
    from pta.scripts.cron_aris_orchestrator import reconcile_state

    run_dir = tmp_path / "checkpoints" / "m8_teacher_seed42"
    run_dir.mkdir(parents=True)
    (run_dir / "scoop_transfer_teacher_final.zip").write_text("ok")

    state = reconcile_state(project_root=tmp_path, ps_output="")

    assert state["m8"]["completed"] is True


def test_reconcile_state_marks_m7_seed_complete_from_final_artifact(tmp_path):
    from pta.scripts.cron_aris_orchestrator import reconcile_state

    run_dir = tmp_path / "checkpoints" / "m7_pta_seed0"
    run_dir.mkdir(parents=True)
    (run_dir / "m7_pta_final.zip").write_text("ok")

    state = reconcile_state(project_root=tmp_path, ps_output="")

    assert 0 in state["m7"]["completed_seeds"]


def test_reconcile_state_marks_running_stages_from_ps_output(tmp_path):
    from pta.scripts.cron_aris_orchestrator import reconcile_state

    ps_output = "\n".join(
        [
            "101 5 python pta/scripts/train_baselines.py --method m8 --seed 42",
            "102 6 python pta/scripts/train_baselines.py --method m1 --seed 0",
            "103 7 python pta/scripts/train_m7.py --seed 1",
        ]
    )

    state = reconcile_state(project_root=tmp_path, ps_output=ps_output)

    assert state["m8"]["running"] is True
    assert state["m1"]["running"] is True
    assert state["m7"]["running"] is True


def test_reconcile_state_does_not_mark_ood_eval_complete_from_main_results_alone(
    tmp_path,
):
    from pta.scripts.cron_aris_orchestrator import reconcile_state

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "main_results.csv").write_text("method,split\n")

    state = reconcile_state(project_root=tmp_path, ps_output="")

    assert state["ood_eval"]["completed"] is False


def test_reconcile_state_marks_ood_eval_complete_from_corrected_outputs(tmp_path):
    from pta.scripts.cron_aris_orchestrator import reconcile_state

    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True)
    (results_dir / "main_results.csv").write_text("method,split\n")
    (results_dir / "ood_eval_per_seed.csv").write_text("method,seed,split\n")

    state = reconcile_state(project_root=tmp_path, ps_output="")

    assert state["ood_eval"]["completed"] is True


def test_cron_aris_orchestrator_has_cli_entrypoint_guard():
    script_path = (
        Path(__file__).resolve().parents[1] / "pta/scripts/cron_aris_orchestrator.py"
    )
    source = script_path.read_text(encoding="utf-8")

    assert 'if __name__ == "__main__":' in source
    assert "raise SystemExit(main())" in source
