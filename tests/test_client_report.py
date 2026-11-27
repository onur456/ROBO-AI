"""Client wording stays on top of the existing Wilson interval."""

from __future__ import annotations

import unittest

from client_report import client_summary, client_text
from step3_batch_stress import wilson_ci


def _row(**kwargs) -> dict:
    base = {
        "episode": 1,
        "seed": 42000127,
        "failed": False,
        "fail_reason": None,
        "friction": 0.8,
        "lateral_force": 10.0,
        "scenario": "default",
        "param_id": "main_batch",
        "level_label": "baseline",
        "pd": "baseline",
        "controller": "pd",
    }
    base.update(kwargs)
    return base


class ClientReportTest(unittest.TestCase):
    def test_empty_run(self) -> None:
        summary = client_summary([])
        self.assertEqual(summary["n"], 0)
        self.assertIsNone(summary["lead_reason"])
        self.assertEqual(client_text(summary), "This run has no episodes yet.")

    def test_no_failures_quotes_the_wilson_interval(self) -> None:
        rows = [_row(episode=index, seed=42000126 + index) for index in range(1, 3)]
        summary = client_summary(rows)
        lo, hi = wilson_ci(0, 2)
        self.assertEqual(summary["ci95"], [lo, hi])
        text = client_text(summary)
        self.assertIn("No failures in 2 episodes.", text)
        self.assertIn(f"[{lo * 100:.1f}%, {hi * 100:.1f}%]", text)

    def test_fall_leads_and_the_example_is_the_harder_hit(self) -> None:
        rows = [
            _row(episode=1, seed=1, failed=True, fail_reason="fall", lateral_force=20.0),
            _row(episode=2, seed=2, failed=True, fail_reason="fall", lateral_force=-350.0, friction=0.14),
            _row(episode=3, seed=3, failed=True, fail_reason="stance_loss", lateral_force=400.0),
            _row(episode=4, seed=4),
        ]
        summary = client_summary(rows)
        lo, hi = wilson_ci(3, 4)
        self.assertEqual(summary["lead_reason"], "fall")
        self.assertEqual(summary["reasons"], {"fall": 2, "stance_loss": 1})
        self.assertEqual(summary["ci95"], [lo, hi])
        self.assertEqual(summary["example"]["seed"], 2)
        self.assertAlmostEqual(summary["example"]["lateral_force"], -350.0)
        text = client_text(summary)
        self.assertIn("The robot fell.", text)
        self.assertIn("lateral force -350.0 N", text)
        self.assertIn("friction 0.140", text)
        self.assertIn("seed 2", text)

    def test_a_tie_prefers_a_fall_over_stance_loss(self) -> None:
        rows = [
            _row(episode=1, seed=1, failed=True, fail_reason="stance_loss"),
            _row(episode=2, seed=2, failed=True, fail_reason="fall"),
        ]
        self.assertEqual(client_summary(rows)["lead_reason"], "fall")

    def test_end_effector_error_is_named_for_the_client(self) -> None:
        rows = [
            _row(
                episode=1,
                seed=9,
                failed=True,
                fail_reason="ee_tracking_error",
                lateral_force=0.0,
                param_id="quick_run",
            )
        ]
        text = client_text(client_summary(rows))
        self.assertIn("The end effector missed the target.", text)
        self.assertNotIn("lost its stance", text)

    def test_saturation_is_named_for_the_client(self) -> None:
        rows = [_row(failed=True, fail_reason="torque_saturation", lateral_force=1.0)]
        text = client_text(client_summary(rows))
        self.assertIn("The motors stayed saturated.", text)


if __name__ == "__main__":
    unittest.main()
