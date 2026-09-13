from __future__ import annotations

import inspect
import time

from src.jobs import PIPELINE_FLAGS, JobConflict, get_job, start_job


def test_start_job_runs_command_and_records_success(tmp_path, monkeypatch):
    monkeypatch.setattr("src.jobs.JOBS_DIR", tmp_path)
    job = start_job("echo", extra_args=["pipeline-ok"], command=["/bin/echo", "pipeline-ok"])
    assert job["status"] in {"queued", "running", "succeeded"}
    deadline = time.time() + 5
    current = get_job(job["id"])
    while current["status"] in {"queued", "running"} and time.time() < deadline:
        time.sleep(0.05)
        current = get_job(job["id"])
    assert current["status"] == "succeeded"
    assert "pipeline-ok" in (current.get("log") or "")
    assert (tmp_path / f"{job['id']}.log").exists()


def test_rejects_second_job_while_one_is_running(tmp_path, monkeypatch):
    monkeypatch.setattr("src.jobs.JOBS_DIR", tmp_path)
    first = start_job("sleep", command=["/bin/sleep", "2"])
    try:
        start_job("echo", command=["/bin/echo", "nope"])
        raised = False
    except JobConflict:
        raised = True
    assert raised is True
    current = get_job(first["id"])
    deadline = time.time() + 5
    while current["status"] == "running" and time.time() < deadline:
        time.sleep(0.05)
        current = get_job(first["id"])


def test_train_job_command_has_no_holdout_flag(tmp_path, monkeypatch):
    monkeypatch.setattr("src.jobs.JOBS_DIR", tmp_path)
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        return _FakeProc()

    monkeypatch.setattr("src.jobs.subprocess.Popen", fake_popen)
    start_job("train")
    command = captured["command"]
    assert "--holdout-season" not in command
    assert "holdout_season" not in inspect.signature(start_job).parameters
    assert PIPELINE_FLAGS["train"] == ["--skip-collect", "--skip-weather"]


class _FakeProc:
    pid = 12345

    def wait(self):
        return 0

