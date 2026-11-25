"""Smoke test: load dashboard episodes without starting Streamlit."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.app import load_data


def _mini_batch() -> dict:
    return {
        "parameter_sweeps": {
            "scenario": "nominal",
            "parameters": [
                {
                    "id": "com_x",
                    "levels": [
                        {
                            "id": "zero",
                            "label": "baseline 0 m",
                            "outcomes": [
                                {
                                    "episode": 1,
                                    "seed": 42000127,
                                    "failed": False,
                                    "fail_reason": None,
                                },
                                {
                                    "episode": 2,
                                    "seed": 42000128,
                                    "failed": True,
                                    "fail_reason": "stance_loss",
                                },
                            ],
                        }
                    ],
                }
            ],
        }
    }


class DashboardSmokeTest(unittest.TestCase):
    def test_load_data_flattens_two_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "batch.json"
            path.write_text(json.dumps(_mini_batch()), encoding="utf-8")
            frame = load_data(str(path))

        self.assertIsInstance(frame, pd.DataFrame)
        for column in ("param_id", "level_label", "failed"):
            self.assertIn(column, frame.columns)
        self.assertEqual(len(frame), 2)
        self.assertEqual(list(frame["param_id"]), ["com_x", "com_x"])
        self.assertEqual(list(frame["level_label"]), ["baseline 0 m", "baseline 0 m"])
        self.assertEqual(list(frame["failed"]), [False, True])

    def test_load_data_reads_root_episodes(self) -> None:
        payload = {
            "scenario": "default",
            "pd": "baseline",
            "controller": "pd",
            "kp": 40.0,
            "kd": 3.0,
            "episodes": [
                {
                    "episode": 1,
                    "seed": 42000127,
                    "failed": False,
                    "friction": 0.8,
                    "lateral_force": -40.0,
                },
                {
                    "episode": 2,
                    "seed": 42000128,
                    "failed": True,
                    "fail_reason": "fall",
                    "friction": 0.2,
                    "lateral_force": 120.0,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "quick.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            frame = load_data(str(path))

        self.assertEqual(len(frame), 2)
        self.assertEqual(list(frame["param_id"]), ["quick_run", "quick_run"])
        self.assertEqual(list(frame["level_label"]), ["default", "default"])
        self.assertEqual(list(frame["friction"]), [0.8, 0.2])
        self.assertEqual(list(frame["lateral_force"]), [-40.0, 120.0])
        self.assertEqual(list(frame.columns[:2]), ["param_id", "level_label"])
        self.assertIn("friction", frame.columns)
        self.assertIn("lateral_force", frame.columns)


if __name__ == "__main__":
    unittest.main()
