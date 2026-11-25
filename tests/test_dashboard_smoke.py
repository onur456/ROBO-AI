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


if __name__ == "__main__":
    unittest.main()
