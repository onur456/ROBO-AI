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

from paired_compare import BETTER, NOISE, WORSE
from dashboard.app import (
    _verdict_color,
    apply_filters,
    auto_generate_config,
    comparison_table,
    failure_heatmap,
    load_data,
)
from dashboard.workspace import estimated_price, run_stress_backend


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

    def test_load_data_stacks_main_batch_and_ofat(self) -> None:
        payload = _mini_batch()
        payload["scenario"] = "default"
        payload["episodes"] = [
            {
                "episode": 1,
                "seed": 42000127,
                "failed": False,
                "friction": 0.5,
                "lateral_force": 10.0,
            },
            {
                "episode": 2,
                "seed": 42000128,
                "failed": True,
                "fail_reason": "fall",
                "friction": 0.1,
                "lateral_force": 200.0,
            },
        ]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "full.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            frame = load_data(str(path))

        self.assertEqual(len(frame), 4)
        main = frame[frame["param_id"] == "main_batch"]
        ofat = frame[frame["param_id"] == "com_x"]
        self.assertEqual(len(main), 2)
        self.assertEqual(list(main["level_label"]), ["baseline", "baseline"])
        self.assertEqual(list(ofat["level_label"]), ["baseline 0 m", "baseline 0 m"])
        self.assertEqual(list(main["friction"]), [0.5, 0.1])
        self.assertTrue(ofat["friction"].isna().all())

        only_ofat = apply_filters(frame, ["com_x"], "All")
        self.assertEqual(len(only_ofat), 2)
        self.assertTrue((only_ofat["param_id"] == "com_x").all())

        heat = failure_heatmap(frame)
        self.assertEqual(set(heat.index), {"main_batch", "com_x"})
        self.assertIn("baseline", heat.columns)
        self.assertIn("baseline 0 m", heat.columns)
        self.assertEqual(float(heat.loc["main_batch", "baseline"]), 50.0)
        self.assertEqual(float(heat.loc["com_x", "baseline 0 m"]), 50.0)

    def test_comparison_table_formats_pvalues_and_verdict_colors(self) -> None:
        table = comparison_table(
            {
                "rows": [
                    {
                        "scenario": "nominal",
                        "param": "com_x",
                        "label": "forward +0.03 m",
                        "n_shared": 20,
                        "n_0_to_1": 0,
                        "n_1_to_0": 12,
                        "delta_pp": -60.0,
                        "p_mcnemar": 4e-15,
                        "p_holm": 0.04523,
                        "direction": BETTER,
                    },
                    {
                        "param": "sensor_bias",
                        "label": "bias +0.05 rad",
                        "p_mcnemar": 0.04523,
                        "p_holm": 0.2,
                        "direction": WORSE,
                    },
                    {
                        "param": "torque_limit",
                        "label": "baseline 100% ctrlrange",
                        "p_mcnemar": None,
                        "p_holm": 1,
                        "direction": NOISE,
                    },
                ]
            }
        )
        self.assertEqual(table.loc[0, "p_mcnemar"], "< 0.001")
        self.assertEqual(table.loc[0, "p_holm"], "0.0452")
        self.assertEqual(table.loc[1, "p_mcnemar"], "0.0452")
        self.assertEqual(table.loc[1, "p_holm"], "0.2000")
        self.assertIsNone(table.loc[2, "p_mcnemar"])
        self.assertEqual(table.loc[2, "p_holm"], "1.0000")
        html = table.style.map(_verdict_color, subset=["Verdict"]).to_html()
        self.assertIn("#3DDC97", html)
        self.assertIn("#FF5C5C", html)
        self.assertIn("#9AA0A6", html)

    def test_auto_generate_config_reads_joint_names(self) -> None:
        xml = """<mujoco model="arm">
          <worldbody>
            <body>
              <joint name="shoulder"/>
              <body>
                <joint name="elbow" type="hinge"/>
                <freejoint name="ignored"/>
              </body>
            </body>
          </worldbody>
        </mujoco>"""
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "arm.xml"
            path.write_text(xml, encoding="utf-8")
            config = auto_generate_config(path)
        self.assertEqual(config["joint_names"], ["shoulder", "elbow"])
        self.assertEqual(config["num_motors"], 2)

    def test_estimated_price_matches_the_two_anchors(self) -> None:
        self.assertAlmostEqual(estimated_price(100), 5.0)
        self.assertAlmostEqual(estimated_price(1000), 20.0)

    def test_run_stress_backend_writes_one_episode(self) -> None:
        report = Path(tempfile.mkdtemp()) / "urdf_batch.json"
        result = run_stress_backend(
            ROOT / "models" / "simple_arm.urdf",
            1,
            report_path=report,
            test_friction=False,
            test_impact=False,
        )
        self.assertEqual(result.returncode, 0, (result.stderr or result.stdout)[-800:])
        payload = json.loads(report.read_text(encoding="utf-8"))
        self.assertEqual(len(payload["episodes"]), 1)

    def test_uploaded_profile_records_robot_type(self) -> None:
        from dashboard.workspace import write_uploaded_profile

        with tempfile.TemporaryDirectory() as folder:
            urdf_profile = Path(folder) / "urdf.json"
            write_uploaded_profile(ROOT / "models" / "simple_arm.urdf", urdf_profile)
            urdf_raw = json.loads(urdf_profile.read_text(encoding="utf-8"))
            self.assertEqual(urdf_raw["robot_type"], "manipulator")

            fixed = Path(folder) / "fixed.xml"
            fixed.write_text(
                """
                <mujoco>
                  <worldbody>
                    <body name="link0">
                      <joint name="hinge" type="hinge"/>
                      <geom size="0.1"/>
                    </body>
                  </worldbody>
                  <actuator>
                    <motor name="m1" joint="hinge"/>
                  </actuator>
                </mujoco>
                """,
                encoding="utf-8",
            )
            fixed_profile = Path(folder) / "fixed.json"
            write_uploaded_profile(fixed, fixed_profile)
            fixed_raw = json.loads(fixed_profile.read_text(encoding="utf-8"))
            self.assertEqual(fixed_raw["robot_type"], "manipulator")

            floating = Path(folder) / "float.xml"
            floating.write_text(
                """
                <mujoco>
                  <worldbody>
                    <body name="base">
                      <freejoint/>
                      <geom size="0.1"/>
                      <body name="leg">
                        <joint name="hinge" type="hinge"/>
                        <geom size="0.05"/>
                      </body>
                    </body>
                  </worldbody>
                  <actuator>
                    <motor name="m1" joint="hinge"/>
                  </actuator>
                </mujoco>
                """,
                encoding="utf-8",
            )
            float_profile = Path(folder) / "float.json"
            write_uploaded_profile(floating, float_profile)
            float_raw = json.loads(float_profile.read_text(encoding="utf-8"))
            self.assertEqual(float_raw["robot_type"], "legged")

    def test_bad_model_is_reported_before_the_batch(self) -> None:
        from dashboard.workspace import write_uploaded_profile

        with tempfile.TemporaryDirectory() as folder:
            broken = Path(folder) / "broken.xml"
            broken.write_text("<not-mujoco>", encoding="utf-8")
            with self.assertRaises(ValueError):
                write_uploaded_profile(broken, Path(folder) / "profile.json")


if __name__ == "__main__":
    unittest.main()
