from __future__ import annotations

import time

from src.jobs import JobConflict, get_job, start_job


def test_collect_job_sets_cfbd_allow_calls(tmp_path, monkeypatch):
    monkeypatch.setattr("src.jobs.JOBS_DIR", tmp_path)
    captured = {}

    class _Proc:
        pid = 4242

        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        captured["env"] = kwargs.get("env") or {}
        captured["command"] = command
        return _Proc()

    monkeypatch.setattr("src.jobs.subprocess.Popen", fake_popen)
    job = start_job("collect")
    assert captured["env"].get("CFBD_ALLOW_CALLS") == "1"
    assert job["kind"] == "collect"


def test_scores_job_does_not_set_cfbd_allow_calls(tmp_path, monkeypatch):
    monkeypatch.setattr("src.jobs.JOBS_DIR", tmp_path)
    captured = {}

    class _Proc:
        pid = 4243

        def wait(self):
            return 0

    def fake_popen(command, **kwargs):
        captured["env"] = kwargs.get("env") or {}
        return _Proc()

    monkeypatch.setattr("src.jobs.subprocess.Popen", fake_popen)
    monkeypatch.delenv("CFBD_ALLOW_CALLS", raising=False)
    start_job("scores")
    assert "CFBD_ALLOW_CALLS" not in captured["env"]


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
