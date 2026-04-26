import json
from pathlib import Path

import pytest

from pta.scripts.dlc import submit_jobs


def test_ablation_suite_builds_one_job_per_variant_seed():
    jobs = submit_jobs.build_job_specs(
        suite="ablation",
        name="pta_ablation",
        variants=["no_probe", "no_belief"],
        seeds=[42, 0, 1],
        gpu_count=1,
        data_sources="d-a,d-b",
        skips=[],
    )

    assert [job.job_name for job in jobs] == [
        "pta_ablation_no_probe_s42",
        "pta_ablation_no_probe_s0",
        "pta_ablation_no_probe_s1",
        "pta_ablation_no_belief_s42",
        "pta_ablation_no_belief_s0",
        "pta_ablation_no_belief_s1",
    ]
    assert jobs[0].command_args == "train_ablation no_probe 42"
    assert jobs[-1].command_args == "train_ablation no_belief 1"
    assert all(job.gpu_count == 1 for job in jobs)
    assert all(job.data_sources == "d-a,d-b" for job in jobs)


def test_ablation_suite_can_skip_local_in_progress_variant_seed():
    jobs = submit_jobs.build_job_specs(
        suite="ablation",
        name="pta_ablation",
        variants=["no_probe", "no_belief"],
        seeds=[42, 0, 1],
        gpu_count=1,
        data_sources=None,
        skips=["no_probe:42"],
    )

    assert [job.job_name for job in jobs] == [
        "pta_ablation_no_probe_s0",
        "pta_ablation_no_probe_s1",
        "pta_ablation_no_belief_s42",
        "pta_ablation_no_belief_s0",
        "pta_ablation_no_belief_s1",
    ]
    assert [job.chunk_id for job in jobs] == [0, 1, 2, 3, 4]
    assert all(job.chunk_total == 5 for job in jobs)


def test_ood_ablation_suite_builds_single_eval_job():
    jobs = submit_jobs.build_job_specs(
        suite="ood-ablation",
        name="pta_ood_ablation",
        variants=[],
        seeds=[],
        gpu_count=1,
        data_sources=None,
        skips=[],
    )

    assert len(jobs) == 1
    assert jobs[0].job_name == "pta_ood_ablation"
    assert jobs[0].command_args == (
        "eval_ood --residual-scale 0.05 --methods m7_noprobe m7_nobelief"
    )


def test_smoke_suite_builds_single_smoke_job():
    jobs = submit_jobs.build_job_specs(
        suite="smoke",
        name="pta_smoke",
        variants=[],
        seeds=[],
        gpu_count=1,
        data_sources=None,
        skips=[],
    )

    assert len(jobs) == 1
    assert jobs[0].command_args == "smoke_env"


def test_rejects_unknown_ablation_variant():
    with pytest.raises(ValueError, match="unsupported variant"):
        submit_jobs.build_job_specs(
            suite="ablation",
            name="pta_ablation",
            variants=["bad_variant"],
            seeds=[42],
            gpu_count=1,
            data_sources=None,
            skips=[],
        )


def test_rejects_unknown_skip_variant():
    with pytest.raises(ValueError, match="unsupported skip variant"):
        submit_jobs.build_job_specs(
            suite="ablation",
            name="pta_ablation",
            variants=["no_probe"],
            seeds=[42],
            gpu_count=1,
            data_sources=None,
            skips=["bad_variant:42"],
        )


def test_dry_run_does_not_call_subprocess_and_writes_manifest(tmp_path, monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("subprocess should not be called in dry-run")

    monkeypatch.setattr(submit_jobs.subprocess, "run", fake_run)
    jobs = submit_jobs.build_job_specs(
        suite="smoke",
        name="pta_smoke",
        variants=[],
        seeds=[],
        gpu_count=1,
        data_sources="d-a",
        skips=[],
    )

    submit_jobs.submit_specs(
        jobs,
        repo_root=tmp_path,
        launch_script=Path("pta/scripts/dlc/launch_job.sh"),
        dry_run=True,
        manifest_path=tmp_path / "jobs.jsonl",
    )

    assert calls == []
    rows = [
        json.loads(line)
        for line in (tmp_path / "jobs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows[0]["job_name"] == "pta_smoke"
    assert rows[0]["suite"] == "smoke"
    assert rows[0]["command_args"] == "smoke_env"
    assert rows[0]["dry_run"] is True
