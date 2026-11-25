"""Check that torque_limit=5% actually changes saturation and/or outcomes."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import mujoco

import step3_batch_stress as s3


def _mean(rows: list[dict], key: str) -> float:
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else 0.0


class TorqueLimitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(s3.MODEL_PATH)
        cls.data = mujoco.MjData(cls.model)
        floor_id = s3.find_floor_geom_id(cls.model)
        foot_ids = s3.find_foot_geom_ids(cls.model)
        cls.torso_id = s3.find_torso_body_id(cls.model)
        cls.friction_geoms = [floor_id] + [g for g in foot_ids if g != floor_id]
        s3.reset_to_stance(cls.model, cls.data)
        cls.target_qpos = np.array(cls.data.qpos[7:], copy=True)
        s3.STOCK_CTRLRANGE["value"] = None
        s3.load_outcome_thresholds()

    def test_saturation_note_zero(self) -> None:
        self.assertEqual(s3.saturation_note(0.0), s3.LIMIT_INACTIVE_NOTE)
        self.assertEqual(s3.saturation_note(0.2), s3.LIMIT_ACTIVE_NOTE)

    def test_five_percent_changes_fail_or_saturation(self) -> None:
        kp = float(s3.PD_MODES["baseline"]["kp"])
        kd = float(s3.PD_MODES["baseline"]["kd"])
        full = s3.run_torque_scale_sample(
            self.model,
            self.data,
            self.target_qpos,
            self.friction_geoms,
            self.torso_id,
            kp,
            kd,
            1.0,
            n_episodes=3,
        )
        harsh = s3.run_torque_scale_sample(
            self.model,
            self.data,
            self.target_qpos,
            self.friction_geoms,
            self.torso_id,
            kp,
            kd,
            0.05,
            n_episodes=3,
        )
        self.assertEqual(len(full), 3)
        self.assertEqual(len(harsh), 3)
        for ep in full + harsh:
            self.assertIn("max_tau_over_limit", ep)
            self.assertIn("saturation_frac", ep)
            self.assertIn("fail", ep)
            self.assertIn("fall", ep)
            self.assertIn("stance_loss", ep)
            self.assertIn("limit_note", ep)

        sat_full = _mean(full, "saturation_frac")
        sat_harsh = _mean(harsh, "saturation_frac")
        fail_full = _mean(full, "fail")
        fail_harsh = _mean(harsh, "fail")
        fall_full = _mean(full, "fall")
        fall_harsh = _mean(harsh, "fall")

        self.assertLessEqual(sat_full, 1e-12)
        self.assertEqual(full[0]["limit_note"], s3.LIMIT_INACTIVE_NOTE)
        self.assertGreater(sat_harsh, 0.5)
        self.assertTrue(
            fail_harsh > fail_full or sat_harsh > sat_full + 0.1,
            msg=(
                f"5% did not differ from 100%: "
                f"fail {fail_full:.2f}->{fail_harsh:.2f}, "
                f"fall {fall_full:.2f}->{fall_harsh:.2f}, "
                f"sat {sat_full:.3f}->{sat_harsh:.3f}"
            ),
        )


if __name__ == "__main__":
    unittest.main()
