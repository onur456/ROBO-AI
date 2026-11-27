"""One-machine job queue. The cap counts MuJoCo workers, not job cards."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
QUEUE_DIR = ROOT / "queue" / "jobs"
LOCK_PATH = ROOT / "queue" / "jobs.lock"
PID_PATH = ROOT / "queue" / "runner.pid"
CHUNK_SIZE = 10

_PROCS: dict[str, subprocess.Popen[str]] = {}
_LOGS: dict[str, object] = {}


def worker_cap() -> int:
    raw = os.environ.get("ROBO_MAX_MUJOCO_WORKERS")
    if raw:
        return max(1, int(raw))
    return max(1, os.cpu_count() or 1)


def workers_wanted(episodes: int, chunk_size: int = CHUNK_SIZE, cpu: int | None = None) -> int:
    """How many pool processes one batch would open if it had the machine to itself."""
    limit = worker_cap() if cpu is None else max(1, int(cpu))
    chunks = max(1, (int(episodes) + int(chunk_size) - 1) // int(chunk_size))
    return max(1, min(chunks, limit))


def assign_workers(wanted: int, in_use: int, cap: int) -> int:
    """Workers this job may take without pushing the machine over the cap."""
    free = int(cap) - int(in_use)
    if free <= 0 or int(wanted) <= 0:
        return 0
    return min(int(wanted), free)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pid_alive(pid: int | None) -> bool:
    if not pid or int(pid) <= 0:
        return False
    if os.name == "nt":
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False
        kernel.CloseHandle(handle)
        return True
    try:
        os.kill(int(pid), 0)
    except OSError:
        return False
    return True


@contextmanager
def queue_lock():
    LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle = open(LOCK_PATH, "a+b")
    try:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        handle.close()


def _job_path(job_id: str) -> Path:
    return QUEUE_DIR / f"{job_id}.json"


def save_job(job: dict) -> None:
    path = _job_path(str(job["id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def list_jobs() -> list[dict]:
    if not QUEUE_DIR.is_dir():
        return []
    jobs: list[dict] = []
    for path in QUEUE_DIR.glob("*.json"):
        try:
            jobs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    jobs.sort(key=lambda job: str(job.get("created_at") or ""))
    return jobs


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


def job_dir(job_id: str) -> Path:
    return QUEUE_DIR / job_id


def enqueue_record(job: dict, *, start_runner: bool = True) -> dict:
    with queue_lock():
        save_job(job)
    if start_runner:
        ensure_runner()
    return job


def runner_alive() -> bool:
    if not PID_PATH.is_file():
        return False
    try:
        pid = int(PID_PATH.read_text(encoding="utf-8").strip() or "0")
    except ValueError:
        return False
    return pid_alive(pid)


def ensure_runner() -> None:
    if runner_alive():
        return
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PID_PATH.is_file():
        PID_PATH.unlink(missing_ok=True)
    log_path = ROOT / "queue" / "runner.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "a", encoding="utf-8")
    flags = 0
    if os.name == "nt":
        flags = 0x08000000 | 0x00000200
    process = subprocess.Popen(
        [sys.executable, "-m", "job_queue"],
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=flags,
    )
    PID_PATH.write_text(str(process.pid), encoding="utf-8")


def _read_progress(path: str | None) -> tuple[int, int] | None:
    if not path:
        return None
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return int(payload.get("done") or 0), int(payload.get("total") or 0)


def _log_tail(path: str | None, limit: int = 40) -> str:
    if not path or not Path(path).is_file():
        return ""
    lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-limit:])


def _refresh_progress(job: dict) -> None:
    reading = _read_progress(job.get("progress_path"))
    if reading is None:
        return
    done, total = reading
    job["progress_done"] = done
    if total > 0:
        job["progress_total"] = total


def _finish(job: dict, code: int | None) -> None:
    report = Path(str(job.get("report_path") or ""))
    report_ready = report.is_file() and report.stat().st_size > 0
    succeeded = report_ready and (code in (None, 0))
    job["workers_assigned"] = 0
    job["finished_at"] = _now()
    if succeeded:
        job["state"] = "done"
        job["progress_done"] = int(job.get("progress_total") or job.get("episodes") or 0)
        job["error"] = None
        return
    job["state"] = "error"
    tail = _log_tail(job.get("log_path"))
    job["error"] = tail or f"step3 exited with code {code}"


def _launch(job: dict, grant: int) -> None:
    argv = list(job["argv"]) + [
        "--max-workers",
        str(int(grant)),
        "--progress",
        str(job["progress_path"]),
    ]
    log_path = Path(str(job["log_path"]))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        argv,
        cwd=ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
    )
    _PROCS[str(job["id"])] = process
    _LOGS[str(job["id"])] = log_handle
    job["state"] = "running"
    job["pid"] = int(process.pid)
    job["workers_assigned"] = int(grant)
    job["started_at"] = _now()


def pump_once() -> None:
    """Start queued jobs that fit under the worker cap and close finished ones."""
    with queue_lock():
        jobs = list_jobs()
        for job in jobs:
            if job.get("state") != "running":
                continue
            _refresh_progress(job)
            process = _PROCS.get(str(job["id"]))
            if process is not None:
                code = process.poll()
                alive = code is None
            else:
                code = None
                alive = pid_alive(job.get("pid"))
            if alive:
                save_job(job)
                continue
            _finish(job, code)
            _PROCS.pop(str(job["id"]), None)
            log_handle = _LOGS.pop(str(job["id"]), None)
            if log_handle is not None:
                log_handle.close()
            save_job(job)
        in_use = sum(int(job.get("workers_assigned") or 0) for job in jobs if job.get("state") == "running")
        cap = worker_cap()
        queued = [job for job in jobs if job.get("state") == "queued"]
        for job in queued:
            grant = assign_workers(int(job.get("workers_wanted") or 0), in_use, cap)
            if grant <= 0:
                continue
            try:
                _launch(job, grant)
            except OSError as exc:
                job["state"] = "error"
                job["workers_assigned"] = 0
                job["error"] = str(exc)
                job["finished_at"] = _now()
            else:
                in_use += grant
            save_job(job)


def serve() -> None:
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    while True:
        pump_once()
        time.sleep(0.4)


def blank_job(
    *,
    job_id: str,
    episodes: int,
    argv: list[str],
    report_path: Path,
    progress_path: Path,
    log_path: Path,
    label: str,
) -> dict:
    return {
        "id": job_id,
        "state": "queued",
        "created_at": _now(),
        "label": label,
        "episodes": int(episodes),
        "workers_wanted": workers_wanted(episodes),
        "workers_assigned": 0,
        "progress_done": 0,
        "progress_total": int(episodes),
        "report_path": str(report_path),
        "progress_path": str(progress_path),
        "log_path": str(log_path),
        "argv": argv,
        "error": None,
        "pid": None,
    }


if __name__ == "__main__":
    serve()
