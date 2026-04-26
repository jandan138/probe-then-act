import json
import os
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TASK = REPO_ROOT / "pta" / "scripts" / "dlc" / "run_task.sh"
LAUNCH_JOB = REPO_ROOT / "pta" / "scripts" / "dlc" / "launch_job.sh"
SUBMIT_SWEEP = REPO_ROOT / "pta" / "scripts" / "dlc" / "submit_ablation_sweep.sh"


def _base_env(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "PTA_CODE_ROOT": str(REPO_ROOT),
            "GENESIS_ROOT": str(tmp_path / "Genesis"),
            "GENESIS_VENV": str(tmp_path / "Genesis" / ".venv"),
            "DLC_RESULTS_ROOT": str(tmp_path / "dlc"),
            "DLC_SKIP_PREFLIGHT": "1",
            "DLC_DRY_RUN": "1",
            "DLC_RUN_ID": "test_run",
        }
    )
    return env


def _run_task(tmp_path, *args):
    return subprocess.run(
        ["bash", str(RUN_TASK), *args],
        cwd=REPO_ROOT,
        env=_base_env(tmp_path),
        text=True,
        capture_output=True,
    )


def _record(tmp_path):
    return json.loads((tmp_path / "dlc" / "runs" / "test_run.json").read_text())


def _write_fake_python(path, body):
    path.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    path.chmod(0o755)


def test_run_task_train_ablation_dry_run_builds_expected_command(tmp_path):
    result = _run_task(tmp_path, "train_ablation", "no_probe", "42")

    assert result.returncode == 0, result.stderr
    assert "python -u pta/scripts/train_m7.py --ablation no_probe --seed 42" in result.stdout
    record = _record(tmp_path)
    assert record["mode"] == "train_ablation"
    assert record["exit_code"] == 0
    assert record["checkpoint_hint"] == (
        "checkpoints/m7_pta_noprobe_seed42/best/best_model.zip"
    )


def test_run_task_eval_ood_dry_run_uses_ablation_default(tmp_path):
    result = _run_task(tmp_path, "eval_ood")

    assert result.returncode == 0, result.stderr
    assert (
        "python -u pta/scripts/run_ood_eval_v2.py --residual-scale 0.05 "
        "--methods m7_noprobe m7_nobelief"
    ) in result.stdout
    record = _record(tmp_path)
    assert record["mode"] == "eval_ood"
    assert record["result_hint"] == "results/main_results.csv"


def test_run_task_writes_record_when_only_python3_is_on_path(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ["date", "mkdir", "python3"]:
        target = shutil.which(tool)
        assert target is not None
        (bin_dir / tool).symlink_to(target)
    env = _base_env(tmp_path)
    env["PATH"] = str(bin_dir)

    result = subprocess.run(
        ["/bin/bash", str(RUN_TASK), "eval_ood"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    record = _record(tmp_path)
    assert record["mode"] == "eval_ood"
    assert record["exit_code"] == 0


def test_run_task_writes_record_for_early_failure_with_python3_fallback(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for tool in ["date", "mkdir", "python3"]:
        target = shutil.which(tool)
        assert target is not None
        (bin_dir / tool).symlink_to(target)
    env = _base_env(tmp_path)
    env["PATH"] = str(bin_dir)
    env["PTA_CODE_ROOT"] = str(tmp_path / "missing-code-root")

    result = subprocess.run(
        ["/bin/bash", str(RUN_TASK), "eval_ood"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "PTA_CODE_ROOT not found" in result.stderr
    record = _record(tmp_path)
    assert record["mode"] == "eval_ood"
    assert record["exit_code"] == result.returncode


def test_run_task_train_ablation_propagates_python_failure_even_with_old_checkpoint(
    tmp_path,
):
    code_root = tmp_path / "code"
    old_checkpoint = (
        code_root
        / "checkpoints"
        / "m7_pta_noprobe_seed42"
        / "best"
        / "best_model.zip"
    )
    old_checkpoint.parent.mkdir(parents=True)
    old_checkpoint.write_text("old", encoding="utf-8")
    fake_python = tmp_path / "fake-python"
    real_python = shutil.which("python3")
    assert real_python is not None
    _write_fake_python(
        fake_python,
        f"""\
if [ "$1" = "-" ]; then
    exec {real_python} -
fi
case "$*" in
    *"pta/scripts/train_m7.py"*) exit 17 ;;
esac
exec {real_python} "$@"
""",
    )
    env = _base_env(tmp_path)
    env["PTA_CODE_ROOT"] = str(code_root)
    env["PYTHON_BIN"] = str(fake_python)
    env["DLC_DRY_RUN"] = "0"

    result = subprocess.run(
        ["/bin/bash", str(RUN_TASK), "train_ablation", "no_probe", "42"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 17
    record = _record(tmp_path)
    assert record["mode"] == "train_ablation"
    assert record["exit_code"] == 17


def test_run_task_smoke_propagates_cuda_probe_failure(tmp_path):
    fake_python = tmp_path / "fake-python"
    real_python = shutil.which("python3")
    assert real_python is not None
    _write_fake_python(
        fake_python,
        f"""\
if [ "$1" = "-" ]; then
    script=$(cat)
    case "$script" in
        *cuda_available*) exit 19 ;;
    esac
    printf '%s\\n' "$script" | exec {real_python} -
fi
case "$*" in
    *"pta/scripts/train_m7.py --help"*) exit 0 ;;
esac
exec {real_python} "$@"
""",
    )
    env = _base_env(tmp_path)
    env["DLC_DRY_RUN"] = "0"
    env["PYTHON_BIN"] = str(fake_python)

    result = subprocess.run(
        ["/bin/bash", str(RUN_TASK), "smoke_env"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 19
    record = _record(tmp_path)
    assert record["mode"] == "smoke_env"
    assert record["exit_code"] == 19


def test_run_task_rejects_invalid_ablation_before_python_launch(tmp_path):
    result = _run_task(tmp_path, "train_ablation", "bad_variant", "42")

    assert result.returncode != 0
    assert "unsupported ablation variant" in result.stderr
    record = _record(tmp_path)
    assert record["mode"] == "train_ablation"
    assert record["exit_code"] == result.returncode


def test_run_task_custom_rejects_agent_and_cron_commands(tmp_path):
    result = _run_task(tmp_path, "custom", "opencode", "-s", "some-session")

    assert result.returncode != 0
    assert "disallowed custom command" in result.stderr
    record = _record(tmp_path)
    assert record["mode"] == "custom"
    assert record["exit_code"] == result.returncode


def test_run_task_custom_rejects_nested_agent_commands(tmp_path):
    result = _run_task(tmp_path, "custom", "bash", "-lc", "opencode -s some-session")

    assert result.returncode != 0
    assert "disallowed custom command" in result.stderr
    record = _record(tmp_path)
    assert record["mode"] == "custom"
    assert record["exit_code"] == result.returncode


def test_run_task_custom_rejects_aris_module_and_uppercase_agent_commands(tmp_path):
    disallowed_commands = [
        ("python", "-m", "pta.scripts.cron_aris_orchestrator"),
        ("bash", "-lc", "Claude --version"),
        ("bash", "-lc", "Codex exec something"),
        ("bash", "-lc", "/tmp/Auto-claude-code-research-in-sleep/run.sh"),
        ("bash", "-lc", "run ARIS handoff"),
    ]

    for index, command in enumerate(disallowed_commands):
        env = _base_env(tmp_path / str(index))
        result = subprocess.run(
            ["bash", str(RUN_TASK), "custom", *command],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )

        assert result.returncode != 0, command
        assert "disallowed custom command" in result.stderr
        record = json.loads(
            (
                tmp_path
                / str(index)
                / "dlc"
                / "runs"
                / "test_run.json"
            ).read_text()
        )
        assert record["mode"] == "custom"
        assert record["exit_code"] == result.returncode


def test_launch_job_dry_run_points_to_pta_worker_path(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "DLC_DRY_RUN": "1",
            "PTA_CODE_ROOT": "/cpfs/shared/simulation/zhuzihou/dev/probe-then-act",
            "DLC_GPU_COUNT": "1",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(LAUNCH_JOB),
            "pta_ablation_no_probe_s42",
            "0",
            "6",
            "d-a,d-b",
            "train_ablation no_probe 42",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--worker_gpu=1" in result.stdout
    assert (
        "bash /cpfs/shared/simulation/zhuzihou/dev/probe-then-act/"
        "pta/scripts/dlc/run_task.sh train_ablation no_probe 42"
    ) in result.stdout


def test_launch_job_dry_run_preserves_multi_token_command_args(tmp_path):
    env = os.environ.copy()
    env.update(
        {
            "DLC_DRY_RUN": "1",
            "PTA_CODE_ROOT": "/cpfs/shared/simulation/zhuzihou/dev/probe-then-act",
            "DLC_GPU_COUNT": "1",
        }
    )

    result = subprocess.run(
        [
            "bash",
            str(LAUNCH_JOB),
            "pta_ood_ablation",
            "0",
            "1",
            "d-a,d-b",
            "eval_ood",
            "--residual-scale",
            "0.05",
            "--methods",
            "m7_noprobe",
            "m7_nobelief",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    assert (
        "pta/scripts/dlc/run_task.sh eval_ood --residual-scale 0.05 "
        "--methods m7_noprobe m7_nobelief"
    ) in result.stdout


def test_submit_ablation_sweep_uses_python3_when_python_is_absent(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python3 = bin_dir / "python3"
    _write_fake_python(
        fake_python3,
        """\
printf '%s\\n' "$*" > "$PYTHON_CAPTURE"
exit 0
""",
    )
    env = os.environ.copy()
    env["PATH"] = str(bin_dir)
    env["PYTHON_CAPTURE"] = str(tmp_path / "python_args.txt")

    result = subprocess.run(
        ["/bin/bash", str(SUBMIT_SWEEP), "--dry-run"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    captured = (tmp_path / "python_args.txt").read_text(encoding="utf-8")
    assert "pta/scripts/dlc/submit_jobs.py" in captured
    assert "--suite ablation" in captured
