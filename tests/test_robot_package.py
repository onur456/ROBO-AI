"""A package is rejected before any batch when meshes, actuators, or task fields are wrong."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from robot_package import PackageError, build_package, validate_profile
from robot_profile import RobotProfile

ROOT = Path(__file__).resolve().parents[1]


class RobotPackageTest(unittest.TestCase):
    def test_legacy_go2_and_arm_pass(self) -> None:
        go2 = RobotProfile(ROOT / "configs" / "go2.json")
        self.assertEqual(go2.package_version, 0)
        self.assertEqual(go2.task, "stance")
        validate_profile(go2)
        arm = RobotProfile(ROOT / "configs" / "robot_arm.json")
        self.assertEqual(arm.task, "track_ee")
        validate_profile(arm)

    def test_missing_mesh_is_reported_before_compile(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "arm.xml"
            model.write_text(
                """
                <mujoco>
                  <compiler meshdir="assets"/>
                  <asset><mesh name="link" file="link.obj"/></asset>
                  <worldbody>
                    <body>
                      <joint name="hinge" type="hinge"/>
                      <geom type="mesh" mesh="link"/>
                    </body>
                  </worldbody>
                  <actuator><motor name="m1" joint="hinge"/></actuator>
                </mujoco>
                """,
                encoding="utf-8",
            )
            with self.assertRaises(PackageError) as caught:
                build_package(model, "track_ee", ee_tracking_limit_m=0.05, saturation_frac_limit=0.9)
            self.assertIn("link.obj", str(caught.exception))

    def test_stance_on_a_fixed_model_fails(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "fixed.xml"
            model.write_text(
                """
                <mujoco>
                  <worldbody>
                    <body>
                      <joint name="hinge" type="hinge"/>
                      <geom size="0.1"/>
                    </body>
                  </worldbody>
                  <actuator><motor name="m1" joint="hinge"/></actuator>
                </mujoco>
                """,
                encoding="utf-8",
            )
            with self.assertRaises(PackageError) as caught:
                build_package(model, "stance", target_height=0.3)
            self.assertIn("floating base", str(caught.exception))

    def test_version_one_track_ee_requires_its_limits(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            model = Path(folder) / "fixed.xml"
            model.write_text(
                """
                <mujoco>
                  <worldbody>
                    <body>
                      <joint name="hinge" type="hinge"/>
                      <geom size="0.1"/>
                    </body>
                  </worldbody>
                  <actuator><motor name="m1" joint="hinge"/></actuator>
                </mujoco>
                """,
                encoding="utf-8",
            )
            payload = build_package(model, "track_ee")
            self.assertEqual(payload["package_version"], 1)
            self.assertIn("ee_tracking_limit_m", payload)
            raw_path = Path(folder) / "pack.json"
            payload.pop("ee_tracking_limit_m")
            payload["model_path"] = str(model)
            raw_path.write_text(json.dumps(payload), encoding="utf-8")
            profile = RobotProfile(raw_path)
            with self.assertRaises(PackageError) as caught:
                validate_profile(profile)
            self.assertIn("ee_tracking_limit_m", str(caught.exception))

    def test_bad_package_exits_before_a_report(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            folder_path = Path(folder)
            model = folder_path / "broken.xml"
            model.write_text(
                """
                <mujoco>
                  <asset><mesh name="link" file="missing.obj"/></asset>
                  <worldbody>
                    <body>
                      <joint name="hinge" type="hinge"/>
                      <geom type="mesh" mesh="link"/>
                    </body>
                  </worldbody>
                  <actuator><motor name="m1" joint="hinge"/></actuator>
                </mujoco>
                """,
                encoding="utf-8",
            )
            config = folder_path / "pack.json"
            config.write_text(
                json.dumps(
                    {
                        "package_version": 1,
                        "model_path": str(model),
                        "task": "track_ee",
                        "robot_type": "manipulator",
                        "joint_names": ["m1"],
                        "num_motors": 1,
                        "target_height": 0.2,
                        "fall_height": 0.2,
                        "contact_z": 0.12,
                        "tilt_fail_deg": 45,
                        "ee_tracking_limit_m": 0.05,
                        "saturation_frac_limit": 0.9,
                    }
                ),
                encoding="utf-8",
            )
            report = folder_path / "should_not_exist.json"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "step3_batch_stress.py"),
                    "--mode",
                    "quick",
                    "--episodes",
                    "1",
                    "--config",
                    str(config),
                    "--report",
                    str(report),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("Package check failed", result.stdout)
            self.assertIn("missing.obj", result.stdout)
            self.assertFalse(report.is_file())


if __name__ == "__main__":
    unittest.main()
