"""URDF profiles fill joint names, and motor-less files gain position actuators."""

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco

from robot_profile import RobotProfile
from step3_batch_stress import auto_inject_actuators, load_robot_model

ROOT = Path(__file__).resolve().parents[1]


class UrdfProfileTest(unittest.TestCase):
    def test_profile_reads_motion_joints_from_urdf(self) -> None:
        profile = RobotProfile(ROOT / "configs" / "my_urdf_robot.json")
        self.assertTrue(profile.model_path.lower().endswith("simple_arm.urdf"))
        self.assertEqual(profile.joint_names, ["joint1", "joint2"])
        self.assertEqual(profile.num_motors, 2)

    def test_auto_inject_adds_position_actuators(self) -> None:
        path = str(ROOT / "models" / "simple_arm.urdf")
        bare = mujoco.MjModel.from_xml_path(path)
        self.assertEqual(int(bare.nu), 0)
        injected = auto_inject_actuators(bare, path)
        self.assertEqual(int(injected.nu), 2)
        names = [
            mujoco.mj_id2name(injected, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(injected.nu)
        ]
        self.assertEqual(names, ["joint1", "joint2"])
        loaded = load_robot_model(path, announce=False)
        self.assertEqual(int(loaded.nu), 2)


if __name__ == "__main__":
    unittest.main()
