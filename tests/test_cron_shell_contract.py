from pathlib import Path


def test_cron_wrapper_exports_required_env_vars():
    wrapper = Path("pta/scripts/run_cron_aris_orchestrator.sh").read_text(
        encoding="utf-8"
    )

    assert "source /home/zhuzihou/dev/Genesis/.venv/bin/activate" in wrapper
    assert "export PYOPENGL_PLATFORM=osmesa" in wrapper
    assert "python pta/scripts/cron_aris_orchestrator.py" in wrapper


def test_install_script_prints_90_minute_schedule():
    install_script = Path("pta/scripts/install_cron_aris_orchestrator.sh").read_text(
        encoding="utf-8"
    )

    assert "*/90" not in install_script
    assert "0 */3 * * *" in install_script or "30 1-23/3 * * *" in install_script
