"""Legged and manipulator profiles do not share Go2 stance thresholds."""

from __future__ import annotations

import unittest
from pathlib import Path

import mujoco

import step3_batch_stress as s3
from robot_profile import RobotProfile

ROOT = Path(__file__).resolve().parents[1]
GO2_Z = 0.23840614105565083
GO2_SLIP = 0.06211325560010261


class RobotTypeProfileTest(unittest.TestCase):
    def tearDown(self) -> None:
        s3.apply_robot_profile(RobotProfile(s3.DEFAULT_CONFIG))
        s3.load_outcome_thresholds()

    def test_go2_uses_the_global_calibration_file(self) -> None:
        profile = RobotProfile(ROOT / "configs" / "go2.json")
        self.assertEqual(profile.robot_type, "legged")
        self.assertEqual(profile.impulse_mode, "base")
        self.assertTrue(profile.use_global_outcome_thresholds)
        derivation, source = s3.resolve_outcome_thresholds(profile)
        self.assertTrue(str(source).endswith("outcome_thresholds.json"))
        self.assertAlmostEqual(float(derivation["z_threshold"]), GO2_Z)
        self.assertAlmostEqual(float(derivation["slip_threshold"]), GO2_SLIP)
        self.assertAlmostEqual(float(derivation["z_ref"]), 0.2440828391927459)

    def test_arm_keeps_its_own_height_and_skips_the_go2_file(self) -> None:
        profile = RobotProfile(ROOT / "configs" / "robot_arm.json")
        self.assertEqual(profile.robot_type, "manipulator")
        self.assertEqual(profile.impulse_mode, "none")
        self.assertFalse(profile.use_global_outcome_thresholds)
        self.assertAlmostEqual(profile.ee_tracking_limit_m, 0.05)
        self.assertAlmostEqual(profile.saturation_frac_limit, 0.9)
        self.assertIsNone(profile.tau_over_limit_limit)
        derivation, source = s3.resolve_outcome_thresholds(profile)
        self.assertIn("robot_arm.json", str(source))
        self.assertNotAlmostEqual(float(derivation["z_threshold"]), GO2_Z)
        self.assertNotAlmostEqual(float(derivation["slip_threshold"]), GO2_SLIP)
        self.assertAlmostEqual(float(derivation["z_ref"]), 0.625)
        self.assertAlmostEqual(float(derivation["z_threshold"]), 0.4)
        s3.apply_robot_profile(profile)
        loaded = s3.load_outcome_thresholds()
        self.assertAlmostEqual(float(loaded["z_threshold"]), 0.4)
        self.assertAlmostEqual(float(s3.STANCE_THRESH["z_threshold"]), 0.4)
        self.assertNotAlmostEqual(float(s3.STANCE_THRESH["z_threshold"]), GO2_Z)

    def test_humanoid_does_not_inherit_go2_slip(self) -> None:
        profile = RobotProfile(ROOT / "configs" / "humanoid.json")
        self.assertEqual(profile.robot_type, "legged")
        self.assertFalse(profile.use_global_outcome_thresholds)
        derivation, source = s3.resolve_outcome_thresholds(profile)
        self.assertIn("humanoid.json", str(source))
        self.assertAlmostEqual(float(derivation["z_threshold"]), 0.7)
        self.assertAlmostEqual(float(derivation["slip_threshold"]), 0.15)
        self.assertAlmostEqual(float(derivation["z_ref"]), 0.79)

    def test_urdf_arm_falls_back_to_fraction_of_its_own_height(self) -> None:
        profile = RobotProfile(ROOT / "configs" / "my_urdf_robot.json")
        self.assertEqual(profile.robot_type, "manipulator")
        self.assertEqual(profile.impulse_mode, "none")
        self.assertFalse(profile.use_global_outcome_thresholds)
        derivation, _source = s3.resolve_outcome_thresholds(profile)
        self.assertAlmostEqual(float(derivation["z_ref"]), 0.5)
        self.assertAlmostEqual(float(derivation["z_threshold"]), 0.85 * 0.5)
        self.assertIsNone(derivation["slip_threshold"])
        self.assertNotAlmostEqual(float(derivation["z_threshold"]), GO2_Z)

    def test_panda_has_no_foot_geoms_for_slip(self) -> None:
        profile = RobotProfile(ROOT / "configs" / "robot_arm.json")
        s3.apply_robot_profile(profile)
        model = mujoco.MjModel.from_xml_path(profile.model_path)
        self.assertEqual(s3.find_foot_geom_ids(model), [])

    def test_outcome_rules_diverge_for_slip_tracking_and_torque(self) -> None:
        go2 = s3.classify_episode_outcome(
            fall=False,
            z_after_settle=0.244,
            foot_slip=0.08,
            n_feet=4,
            z_thr=GO2_Z,
            slip_thr=GO2_SLIP,
            z_ref_use=0.2440828391927459,
            robot_type="legged",
            saturation_frac=1.0,
            saturation_frac_limit=0.9,
        )
        self.assertTrue(go2["stance_loss"])
        self.assertEqual(go2["fail_reason"], "stance_loss")
        self.assertFalse(go2["torque_fail"])
        self.assertTrue(go2["slip_checked"])

        arm_slip = s3.classify_episode_outcome(
            fall=False,
            z_after_settle=0.62,
            foot_slip=0.08,
            n_feet=0,
            z_thr=0.4,
            slip_thr=GO2_SLIP,
            z_ref_use=0.625,
            robot_type="manipulator",
            ee_tracking_error=0.01,
            ee_limit=0.05,
            saturation_frac=0.66,
            saturation_frac_limit=0.9,
        )
        self.assertFalse(arm_slip["stance_loss"])
        self.assertFalse(arm_slip["slip_checked"])
        self.assertFalse(arm_slip["failed"])
        self.assertIsNone(arm_slip["fail_reason"])

        arm_ee = s3.classify_episode_outcome(
            fall=False,
            z_after_settle=0.62,
            foot_slip=0.0,
            n_feet=0,
            z_thr=0.4,
            slip_thr=0.15,
            z_ref_use=0.625,
            robot_type="manipulator",
            ee_tracking_error=0.12,
            ee_limit=0.05,
            saturation_frac=0.1,
            saturation_frac_limit=0.9,
        )
        self.assertEqual(arm_ee["fail_reason"], "ee_tracking_error")
        self.assertTrue(arm_ee["ee_fail"])
        self.assertTrue(arm_ee["failed"])

        arm_low = s3.classify_episode_outcome(
            fall=False,
            z_after_settle=0.2,
            foot_slip=1.0,
            n_feet=4,
            z_thr=0.4,
            slip_thr=0.15,
            z_ref_use=0.625,
            robot_type="manipulator",
            ee_tracking_error=0.01,
            ee_limit=0.05,
            saturation_frac=0.66,
            saturation_frac_limit=0.9,
            max_tau_over_limit=49.0,
        )
        self.assertFalse(arm_low["stance_loss"])
        self.assertFalse(arm_low["failed"])
        self.assertIsNone(arm_low["fail_reason"])
        self.assertFalse(arm_low["slip_checked"])

        arm_tau = s3.classify_episode_outcome(
            fall=False,
            z_after_settle=0.62,
            foot_slip=0.0,
            n_feet=0,
            z_thr=0.4,
            slip_thr=0.15,
            z_ref_use=0.625,
            robot_type="manipulator",
            ee_tracking_error=0.01,
            ee_limit=0.05,
            saturation_frac=0.95,
            saturation_frac_limit=0.9,
        )
        self.assertEqual(arm_tau["fail_reason"], "torque_saturation")
        self.assertTrue(arm_tau["torque_fail"])

        arm_peak = s3.classify_episode_outcome(
            fall=False,
            z_after_settle=0.62,
            foot_slip=0.0,
            n_feet=0,
            z_thr=0.4,
            slip_thr=0.15,
            z_ref_use=0.625,
            robot_type="manipulator",
            ee_tracking_error=0.01,
            ee_limit=0.05,
            saturation_frac=0.2,
            saturation_frac_limit=0.9,
            max_tau_over_limit=12.0,
            tau_over_limit_limit=5.0,
        )
        self.assertEqual(arm_peak["fail_reason"], "torque_saturation")

        legged_tau = s3.classify_episode_outcome(
            fall=False,
            z_after_settle=0.244,
            foot_slip=0.01,
            n_feet=4,
            z_thr=GO2_Z,
            slip_thr=GO2_SLIP,
            z_ref_use=0.2440828391927459,
            robot_type="legged",
            saturation_frac=1.0,
            saturation_frac_limit=0.9,
            max_tau_over_limit=49.0,
            tau_over_limit_limit=1.0,
        )
        self.assertFalse(legged_tau["failed"])
        self.assertIsNone(legged_tau["fail_reason"])


if __name__ == "__main__":
    unittest.main()
