"""Background collect / train jobs. One at a time."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from src.config import MODELS_DIR, ROOT

JOBS_DIR = MODELS_DIR / "jobs"

PIPELINE_FLAGS = {
    "collect": ["--skip-weather", "--skip-train"],
    "weather": ["--skip-collect", "--skip-train"],
    "train": ["--skip-collect", "--skip-weather"],
    "pipeline": [],
}


class JobConflict(RuntimeError):
    """A collect/train job is already running."""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _read(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if not text.strip():
        raise json.JSONDecodeError("empty job file", text, 0)
    return json.loads(text)


def _write(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    try:
        tmp.replace(path)
    except FileNotFoundError:
        tmp.unlink(missing_ok=True)


def _log_tail(job_id: str, limit: int = 200) -> str:
    log_path = JOBS_DIR / f"{job_id}.log"
    if not log_path.exists():
        return ""
    lines = log_path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def _reap_stale() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for path in JOBS_DIR.glob("*.json"):
        try:
            data = _read(path)
        except json.JSONDecodeError:
            continue
        if data.get("status") not in {"queued", "running"}:
            continue
        started = float(data.get("started_at") or 0)
        if time.time() - started < 8:
            continue
        pid = data.get("pid")
        if pid and _alive(int(pid)):
            continue
        data["status"] = "failed"
        data["error"] = "process exited unexpectedly"
        data["finished_at"] = time.time()
        _write(path, data)


def running_job() -> dict[str, Any] | None:
    _reap_stale()
    for path in sorted(JOBS_DIR.glob("*.json")):
        try:
            data = _read(path)
        except json.JSONDecodeError:
            continue
        if data.get("status") in {"queued", "running"}:
            return data
    return None


def _watch(job_id: str, proc: subprocess.Popen, log_handle) -> None:
    try:
        code = proc.wait()
    finally:
        log_handle.close()
    path = JOBS_DIR / f"{job_id}.json"
    data = None
    for _ in range(10):
        try:
            data = _read(path)
            break
        except (json.JSONDecodeError, FileNotFoundError, OSError):
            time.sleep(0.05)
    if data is None:
        return
    data["returncode"] = code
    data["finished_at"] = time.time()
    data["status"] = "succeeded" if code == 0 else "failed"
    if code != 0:
        data["error"] = f"exit {code}"
    try:
        _write(path, data)
    except FileNotFoundError:
        return
    if code == 0:
        try:
            from src.simulate import invalidate_cache

            invalidate_cache()
        except Exception:
            pass


def start_job(
    kind: str,
    extra_args: list[str] | None = None,
    command: list[str] | None = None,
) -> dict[str, Any]:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    active = running_job()
    if active:
        raise JobConflict(f"job {active['id']} is already {active['status']}")
    if command is None:
        if kind not in PIPELINE_FLAGS:
            raise ValueError(f"unknown job kind {kind!r}")
        command = [sys.executable, str(ROOT / "run_pipeline.py"), *PIPELINE_FLAGS[kind]]
        if extra_args:
            command.extend(extra_args)
    job_id = uuid.uuid4().hex[:12]
    log_path = JOBS_DIR / f"{job_id}.log"
    meta_path = JOBS_DIR / f"{job_id}.json"
    payload = {
        "id": job_id,
        "kind": kind,
        "command": command,
        "status": "queued",
        "started_at": time.time(),
        "finished_at": None,
        "pid": None,
        "returncode": None,
        "error": None,
    }
    _write(meta_path, payload)
    log_handle = log_path.open("w")
    proc = subprocess.Popen(
        command,
        cwd=str(ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    payload["pid"] = proc.pid
    payload["status"] = "running"
    _write(meta_path, payload)
    thread = threading.Thread(target=_watch, args=(job_id, proc, log_handle), daemon=True)
    thread.start()
    return get_job(job_id)


def get_job(job_id: str) -> dict[str, Any]:
    path = JOBS_DIR / f"{job_id}.json"
    if not path.exists():
        raise KeyError(job_id)
    _reap_stale()
    data = None
    for _ in range(8):
        try:
            data = _read(path)
            break
        except json.JSONDecodeError:
            time.sleep(0.02)
    if data is None:
        data = _read(path)
    data["log"] = _log_tail(job_id)
    return data


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted(JOBS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
        try:
            rows.append(get_job(path.stem))
        except (json.JSONDecodeError, KeyError):
            continue
    return rows
