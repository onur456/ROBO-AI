"""Queue states and the shared MuJoCo worker cap."""

from __future__ import annotations

import os
import sys
import textwrap
import time
import unittest
from pathlib import Path

import job_queue
import step3_batch_stress as s3


class WorkerBudgetTest(unittest.TestCase):
    def test_chunk_size_matches_the_batch_engine(self) -> None:
        self.assertEqual(job_queue.CHUNK_SIZE, s3.CHUNK_SIZE)

    def test_one_small_batch_takes_a_single_worker(self) -> None:
        self.assertEqual(job_queue.workers_wanted(10, chunk_size=10, cpu=8), 1)
        self.assertEqual(job_queue.workers_wanted(30, chunk_size=10, cpu=8), 3)
        self.assertEqual(job_queue.workers_wanted(1000, chunk_size=10, cpu=8), 8)

    def test_second_job_only_gets_the_free_workers(self) -> None:
        self.assertEqual(job_queue.assign_workers(8, in_use=8, cap=8), 0)
        self.assertEqual(job_queue.assign_workers(8, in_use=1, cap=8), 7)
        self.assertEqual(job_queue.assign_workers(4, in_use=0, cap=2), 2)

    def test_pool_cap_lowers_one_batch_without_changing_the_default(self) -> None:
        cpu = os.cpu_count() or 1
        previous = s3.MAX_POOL_WORKERS
        try:
            s3.MAX_POOL_WORKERS = None
            self.assertEqual(s3.pool_worker_count(100), min(100, cpu))
            s3.MAX_POOL_WORKERS = 2
            self.assertEqual(s3.pool_worker_count(100), min(2, cpu))
        finally:
            s3.MAX_POOL_WORKERS = previous


class QueueScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = job_queue.QUEUE_DIR
        self._lock = job_queue.LOCK_PATH
        self._procs = dict(job_queue._PROCS)
        self._env = os.environ.get("ROBO_MAX_MUJOCO_WORKERS")

    def tearDown(self) -> None:
        for process in job_queue._PROCS.values():
            if process.poll() is None:
                process.kill()
        for handle in job_queue._LOGS.values():
            handle.close()
        job_queue._PROCS.clear()
        job_queue._LOGS.clear()
        job_queue._PROCS.update(self._procs)
        job_queue.QUEUE_DIR = self._dir
        job_queue.LOCK_PATH = self._lock
        if self._env is None:
            os.environ.pop("ROBO_MAX_MUJOCO_WORKERS", None)
        else:
            os.environ["ROBO_MAX_MUJOCO_WORKERS"] = self._env

    def test_two_full_jobs_do_not_run_two_pools_at_once(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job_queue.QUEUE_DIR = root / "jobs"
            job_queue.LOCK_PATH = root / "jobs.lock"
            os.environ["ROBO_MAX_MUJOCO_WORKERS"] = "2"
            script = root / "fake_job.py"
            script.write_text(
                textwrap.dedent(
                    """
                    import argparse, json, time
                    from pathlib import Path
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--report", required=True)
                    parser.add_argument("--progress", required=True)
                    parser.add_argument("--max-workers", type=int, required=True)
                    parser.add_argument("--sleep", type=float, default=0.2)
                    args = parser.parse_args()
                    Path(args.progress).write_text(json.dumps({"done": 1, "total": 2}), encoding="utf-8")
                    time.sleep(args.sleep)
                    Path(args.progress).write_text(json.dumps({"done": 2, "total": 2}), encoding="utf-8")
                    Path(args.report).write_text(json.dumps({"episodes": [{}, {}]}), encoding="utf-8")
                    """
                ),
                encoding="utf-8",
            )

            def add(name: str) -> None:
                report = root / name / "report.json"
                progress = root / name / "progress.json"
                report.parent.mkdir(parents=True)
                job = job_queue.blank_job(
                    job_id=name,
                    episodes=100,
                    argv=[
                        sys.executable,
                        str(script),
                        "--report",
                        str(report),
                        "--sleep",
                        "0.8",
                    ],
                    report_path=report,
                    progress_path=progress,
                    log_path=root / name / "log.txt",
                    label=name,
                )
                job["workers_wanted"] = 2
                job_queue.enqueue_record(job, start_runner=False)

            add("aaaa")
            time.sleep(0.05)
            add("bbbb")
            job_queue.pump_once()
            states = {job["id"]: job for job in job_queue.list_jobs()}
            self.assertEqual(states["aaaa"]["state"], "running")
            self.assertEqual(states["aaaa"]["workers_assigned"], 2)
            self.assertEqual(states["bbbb"]["state"], "queued")
            self.assertEqual(states["bbbb"]["workers_assigned"], 0)
            self.assertLessEqual(
                sum(int(job["workers_assigned"]) for job in states.values()),
                2,
            )
            deadline = time.time() + 8
            while time.time() < deadline:
                job_queue.pump_once()
                states = {job["id"]: job for job in job_queue.list_jobs()}
                if states["aaaa"]["state"] == "done" and states["bbbb"]["state"] == "done":
                    break
                time.sleep(0.2)
            self.assertEqual(states["aaaa"]["state"], "done")
            self.assertEqual(states["bbbb"]["state"], "done")
            self.assertEqual(
                sum(int(job["workers_assigned"]) for job in states.values()),
                0,
            )


if __name__ == "__main__":
    unittest.main()
