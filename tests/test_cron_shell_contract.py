from pathlib import Path
import subprocess


def test_cron_wrapper_exports_required_env_vars():
    wrapper = Path("pta/scripts/run_cron_aris_orchestrator.sh").read_text(
        encoding="utf-8"
    )

    assert "source /home/zhuzihou/dev/Genesis/.venv/bin/activate" in wrapper
    assert "export PYOPENGL_PLATFORM=osmesa" in wrapper
    assert "export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}" in wrapper
    assert (
        "export PYTHONPATH=/home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d:${PYTHONPATH:-}"
        in wrapper
    )
    assert (
        "cd /home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d" in wrapper
    )
    assert "python pta/scripts/cron_aris_orchestrator.py" in wrapper


def test_install_script_prints_exact_90_minute_schedule():
    output = subprocess.run(
        ["bash", "pta/scripts/install_cron_aris_orchestrator.sh"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output.stdout.splitlines() == [
        "0 */3 * * * /home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d/pta/scripts/run_cron_aris_orchestrator.sh",
        "30 1-22/3 * * * /home/zhuzihou/dev/probe-then-act/.worktrees/aris-resume-stage-d/pta/scripts/run_cron_aris_orchestrator.sh",
    ]
