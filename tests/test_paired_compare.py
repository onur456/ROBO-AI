"""Paired run comparison by seed: order, McNemar, Holm, partial overlap."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import step3_batch_stress as s3
from paired_compare import BETTER, NOISE, WORSE, compare_runs


def _rows(pairs: list[tuple[int, bool]]) -> list[dict]:
    return [{"episode": i + 1, "seed": seed, "failed": failed} for i, (seed, failed) in enumerate(pairs)]


def _run(rows: list[dict], scenario: str = "nominal", param: str = "pd", level: str = "base") -> dict:
    return {
        "parameter_sweeps_by_scenario": {
            scenario: {
                "scenario": scenario,
                "parameters": [
                    {
                        "id": param,
                        "levels": [{"id": level, "label": level, "outcomes": rows}],
                    }
                ],
            }
        }
    }


class PairedCompareTest(unittest.TestCase):
    def test_shuffle_matches_by_seed(self) -> None:
        before = _rows([(10, True), (20, False), (30, True), (40, False)])
        after = _rows([(40, True), (10, False), (30, True), (20, False)])
        shuffled = list(reversed(after))
        left = compare_runs(_run(before), _run(after))
        right = compare_runs(_run(before), _run(shuffled))
        a = left["rows"][0]
        b = right["rows"][0]
        self.assertEqual(a["n_shared"], 4)
        self.assertEqual(b["n_shared"], 4)
        self.assertEqual(a["n_0_to_1"], b["n_0_to_1"])
        self.assertEqual(a["n_1_to_0"], b["n_1_to_0"])
        self.assertEqual(a["before"]["n_failures"], 2)
        self.assertEqual(a["after"]["n_failures"], 2)
        self.assertEqual(a["n_1_to_0"], 1)
        self.assertEqual(a["n_0_to_1"], 1)
        self.assertEqual(a["warnings"], [])

    def test_discordant_mcnemar_and_holm(self) -> None:
        # Shift level: two 1->0 and one 0->1. Flat level: every pair agrees.
        shift_a = _rows([(i, i <= 4) for i in range(1, 9)])
        shift_b = _rows(
            [
                (1, True),
                (2, True),
                (3, False),
                (4, False),
                (5, False),
                (6, False),
                (7, True),
                (8, False),
            ]
        )
        flat_a = _rows([(i, i <= 2) for i in range(1, 7)])
        flat_b = _rows([(i, i <= 2) for i in range(1, 7)])
        payload_a = {
            "parameter_sweeps_by_scenario": {
                "nominal": {
                    "parameters": [
                        {
                            "id": "ctrl",
                            "levels": [
                                {"id": "shift", "label": "shift", "outcomes": shift_a},
                                {"id": "flat", "label": "flat", "outcomes": flat_a},
                            ],
                        }
                    ]
                }
            }
        }
        payload_b = {
            "parameter_sweeps_by_scenario": {
                "nominal": {
                    "parameters": [
                        {
                            "id": "ctrl",
                            "levels": [
                                {"id": "shift", "label": "shift", "outcomes": shift_b},
                                {"id": "flat", "label": "flat", "outcomes": list(reversed(flat_b))},
                            ],
                        }
                    ]
                }
            }
        }
        result = compare_runs(payload_a, payload_b)
        by_level = {row["level"]: row for row in result["rows"]}
        shift = by_level["shift"]
        flat = by_level["flat"]
        self.assertEqual(shift["n_0_to_1"], 1)
        self.assertEqual(shift["n_1_to_0"], 2)
        self.assertEqual(shift["before"]["n_failures"], 4)
        self.assertEqual(shift["after"]["n_failures"], 3)
        self.assertAlmostEqual(shift["delta_pp"], -12.5, places=6)
        direct = s3.mcnemar_exact_paired(
            [{"episode": seed, "failed": failed} for seed, failed in [
                (1, True), (2, True), (3, True), (4, True),
                (5, False), (6, False), (7, False), (8, False),
            ]],
            [{"episode": seed, "failed": failed} for seed, failed in [
                (1, True), (2, True), (3, False), (4, False),
                (5, False), (6, False), (7, True), (8, False),
            ]],
        )
        self.assertAlmostEqual(shift["p_mcnemar"], direct["pvalue"], places=12)
        self.assertEqual(direct["n10"], 1)
        self.assertEqual(direct["n01"], 2)
        self.assertEqual(flat["n_0_to_1"], 0)
        self.assertEqual(flat["n_1_to_0"], 0)
        self.assertEqual(flat["p_mcnemar"], 1.0)
        p_adj, reject = s3.holm_adjust([shift["p_mcnemar"], flat["p_mcnemar"]])
        got = sorted(row["p_holm"] for row in result["rows"])
        self.assertEqual(got, sorted(p_adj))
        self.assertEqual(
            [by_level["shift"]["significant"], by_level["flat"]["significant"]],
            [bool(reject[0]), bool(reject[1])],
        )
        self.assertEqual(flat["direction"], NOISE)
        self.assertIn(shift["direction"], {BETTER, NOISE, WORSE})
        if shift["significant"]:
            self.assertEqual(shift["direction"], BETTER)

    def test_partial_seeds_and_zero_discordant(self) -> None:
        before = _rows([(1, True), (2, False), (3, True)])
        after = _rows([(2, False), (3, True), (9, False)])
        result = compare_runs(_run(before), _run(after))
        row = result["rows"][0]
        self.assertEqual(row["n_shared"], 2)
        self.assertEqual(row["n_only_a"], 1)
        self.assertEqual(row["n_only_b"], 1)
        self.assertEqual(row["n_0_to_1"], 0)
        self.assertEqual(row["n_1_to_0"], 0)
        self.assertEqual(row["p_mcnemar"], 1.0)
        self.assertEqual(row["direction"], NOISE)
        self.assertTrue(any("only in A" in note for note in row["warnings"]))
        lo, hi = row["before"]["ci95"]
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        empty = compare_runs(_run(_rows([(1, False)])), _run(_rows([(2, True)])))
        self.assertEqual(empty["rows"][0]["n_shared"], 0)
        self.assertEqual(empty["rows"][0]["direction"], "no shared seeds")
        self.assertEqual(empty["n_tests"], 0)
        self.assertEqual(empty["strongest"], [])

    def test_quick_report_uses_episode_seeds(self) -> None:
        def pack(flags: list[bool]) -> dict:
            return {
                "seed": 42,
                "scenario": "moderate",
                "n_episodes": len(flags),
                "failures": [],
                "episodes": [
                    {
                        "episode": i + 1,
                        "seed": 1000 + i,
                        "failed": flag,
                        "z": [0.2, 0.21],
                    }
                    for i, flag in enumerate(flags)
                ],
            }

        result = compare_runs(
            pack([True, False, False, True]),
            pack([True, True, False, True]),
        )
        row = result["rows"][0]
        self.assertEqual(row["scenario"], "moderate")
        self.assertEqual(row["n_shared"], 4)
        self.assertEqual(row["n_only_a"], 0)
        self.assertEqual(row["n_only_b"], 0)
        self.assertEqual(row["before"]["n_failures"], 2)
        self.assertEqual(row["after"]["n_failures"], 3)
        self.assertEqual(row["n_0_to_1"], 1)
        self.assertEqual(row["n_1_to_0"], 0)
        self.assertEqual(result["warnings"], [])

    def test_heading_names_controller_and_z_ref_mismatch(self) -> None:
        heading = s3.main_batch_heading("stance_shift", "baseline", 40, 3)
        self.assertIn("controller stance_shift", heading)
        self.assertIn("PD baseline", heading)
        self.assertIn("kp=40", heading)
        self.assertIn("kd=3", heading)
        self.assertNotIn("(baseline, kp=40, kd=3)", heading)
        before = _run(_rows([(1, False), (2, True)]))
        after = _run(_rows([(1, True), (2, True)]))
        before["controller"] = "pd"
        after["controller"] = "stance_shift"
        before["pd"] = "baseline"
        after["pd"] = "baseline"
        before["kp"] = 40
        after["kp"] = 40
        before["kd"] = 3
        after["kd"] = 3
        before["stance_reference"] = {
            "controller": "pd",
            "pd": "baseline",
            "kp": 40,
            "kd": 3,
            "z_ref": 0.2441,
            "z_threshold": 0.2384,
            "slip_threshold": 0.0621,
        }
        after["stance_reference"] = {
            "controller": "stance_shift",
            "pd": "baseline",
            "kp": 40,
            "kd": 3,
            "z_ref": 0.2702,
            "z_threshold": 0.2384,
            "slip_threshold": 0.0621,
        }
        result = compare_runs(before, after)
        self.assertEqual(result["before_meta"]["controller"], "pd")
        self.assertEqual(result["after_meta"]["controller"], "stance_shift")
        self.assertAlmostEqual(result["before_meta"]["z_ref"], 0.2441)
        self.assertAlmostEqual(result["after_meta"]["z_ref"], 0.2702)
        self.assertTrue(any("z_ref differs" in note for note in result["warnings"]))
        self.assertFalse(any("stance_loss z threshold" in note for note in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
