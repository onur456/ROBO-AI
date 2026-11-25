"""Exact McNemar and Holm adjustment without a MuJoCo run."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import step3_batch_stress as s3


def _outcomes(flags: list[bool]) -> list[dict]:
    return [
        {
            "episode": i + 1,
            "seed": s3.episode_seed(i + 1),
            "failed": bool(f),
            "fail_fall": bool(f),
            "fail_stance_loss": False,
            "fail_reason": "fall" if f else None,
        }
        for i, f in enumerate(flags)
    ]


class OfatStatsTest(unittest.TestCase):
    def test_wilson_ci_bounds(self) -> None:
        lo, hi = s3.wilson_ci(0, 100)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLess(hi, 0.05)
        lo2, hi2 = s3.wilson_ci(100, 100)
        self.assertGreater(lo2, 0.95)
        self.assertLessEqual(hi2, 1.0)

    def test_mcnemar_all_concordant(self) -> None:
        flags = [False] * 20 + [True] * 10
        res = s3.mcnemar_exact_paired(_outcomes(flags), _outcomes(flags))
        self.assertEqual(res["n_discordant"], 0)
        self.assertEqual(res["pvalue"], 1.0)
        self.assertEqual(res["n_paired"], 30)

    def test_mcnemar_detects_paired_shift(self) -> None:
        base = [False] * 30
        treat = [False] * 20 + [True] * 10
        res = s3.mcnemar_exact_paired(_outcomes(base), _outcomes(treat))
        self.assertEqual(res["n10"], 10)
        self.assertEqual(res["n01"], 0)
        self.assertLess(res["pvalue"], 0.01)

    def test_holm_family_over_mcnemar_pvalues(self) -> None:
        pvals = [0.001, 0.04, 0.20, 0.80]
        p_adj, sig = s3.holm_adjust(pvals, alpha=0.05)
        self.assertEqual(len(p_adj), 4)
        self.assertTrue(sig[0])
        self.assertFalse(sig[2])
        self.assertFalse(sig[3])
        self.assertGreaterEqual(p_adj[0], pvals[0])
        self.assertGreaterEqual(p_adj[1], pvals[1])

    def test_outcome_record_keeps_seed_and_fail(self) -> None:
        rec = s3.outcome_record(
            {
                "episode": 3,
                "seed": 123,
                "failed": True,
                "fail_fall": True,
                "fail_stance_loss": False,
                "fail_reason": "fall",
            }
        )
        self.assertEqual(rec["episode"], 3)
        self.assertEqual(rec["seed"], 123)
        self.assertTrue(rec["failed"])
        self.assertTrue(rec["fail_fall"])


if __name__ == "__main__":
    unittest.main()
