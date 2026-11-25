"""Step 4: recommendations follow Holm only, with no templates and no "worst of all: baseline"."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import step4_ai_diagnose as s4


def _outcomes(n: int, k: int) -> list[dict]:
    return [
        {"episode": i + 1, "seed": 42 * 1_000_003 + i + 1, "failed": i < k}
        for i in range(n)
    ]


def _level(
    lid: str,
    label: str,
    rate: float,
    *,
    n: int = 50,
    significant: bool = False,
    p_holm=None,
    factor: dict | None = None,
    baseline: bool = False,
) -> dict:
    k = int(round(rate * n))
    lo, hi = s4.wilson_ci(k, n)
    return {
        "id": lid,
        "label": label,
        "failure_rate": rate,
        "n_failures": k,
        "n_episodes": n,
        "failure_rate_ci95": [lo, hi],
        "delta_vs_baseline": 0.0 if baseline else rate,
        "p_holm": None if baseline else p_holm,
        "p_mcnemar": None if baseline else (0.001 if significant else 0.4),
        "significant": False if baseline else significant,
        "factor": factor or {},
        "outcomes": _outcomes(n, k),
    }


def _param(pid: str, name: str, baseline: str, levels: list[dict]) -> dict:
    base = next(lv for lv in levels if lv["id"] == baseline)
    base_rate = float(base["failure_rate"])
    for lv in levels:
        lv["delta_vs_baseline"] = float(lv["failure_rate"]) - base_rate
    return {"id": pid, "name": name, "baseline": baseline, "levels": levels}


def _payload(parameters: list[dict], scenario: str = "nominal") -> dict:
    sweeps = {
        "n_episodes": 50,
        "seed": 42,
        "scenario": scenario,
        "scenario_label": scenario,
        "multiple_testing": {"method": "holm", "alpha": 0.05, "n_tests": 24},
        "parameters": parameters,
    }
    return {
        "n_episodes": 0,
        "n_survived": 0,
        "failure_rate": 0.0,
        "failures": [],
        "parameter_sweeps_by_scenario": {scenario: sweeps},
    }


class Step4DiagnoseTest(unittest.TestCase):
    def test_templates_removed(self) -> None:
        self.assertFalse(hasattr(s4, "SWEEP_FIXES"))
        self.assertFalse(hasattr(s4, "controller_recommendations"))

    def test_wilson_zero_and_all(self) -> None:
        lo, hi = s4.wilson_ci(0, 50)
        self.assertGreaterEqual(lo, 0.0)
        self.assertLess(hi, 0.08)
        lo2, hi2 = s4.wilson_ci(50, 50)
        self.assertGreater(lo2, 0.90)
        self.assertLessEqual(hi2, 1.0)

    def test_no_baseline_as_worst_when_all_zero(self) -> None:
        payload = _payload(
            [
                _param(
                    "com_x",
                    "смещение COM по X",
                    "zero",
                    [
                        _level(
                            "zero",
                            "базовый 0 м",
                            0.0,
                            baseline=True,
                            factor={"kind": "com_offset", "offset": [0.0, 0.0, 0.0]},
                        ),
                        _level(
                            "pos",
                            "вперёд +0.03 м",
                            0.0,
                            p_holm=1.0,
                            factor={"kind": "com_offset", "offset": [0.03, 0.0, 0.0]},
                        ),
                    ],
                )
            ]
        )
        text = "\n".join(s4.format_sweep_sections(payload))
        self.assertIn("no significant harmful effect found", text)
        self.assertNotIn("worst of all", text.lower())
        self.assertNotIn("marked as fixed", text.lower())

    def test_improving_not_called_fix(self) -> None:
        payload = _payload(
            [
                _param(
                    "joint_damping",
                    "демпфирование суставов",
                    "base",
                    [
                        _level(
                            "base",
                            "базовое 1x",
                            0.40,
                            baseline=True,
                            factor={"kind": "damping", "scale": 1.0},
                        ),
                        _level(
                            "harsh",
                            "жёсткое 10x",
                            0.10,
                            significant=True,
                            p_holm=0.01,
                            factor={"kind": "damping", "scale": 10.0},
                        ),
                    ],
                )
            ]
        )
        diag = s4.build_diagnosis_payload(payload)
        self.assertEqual(diag["n_significant_harmful"], 0)
        self.assertEqual(diag["n_significant_improving"], 1)
        text = "\n".join(s4.format_recommendation_sections(diag))
        self.assertIn("improving effect", text)
        self.assertNotIn("marked as fixed", text.lower())
        self.assertIn("no significant harmful effect found", text)

    def test_nonsignificant_drop_is_not_improving(self) -> None:
        payload = _payload(
            [
                _param(
                    "joint_damping",
                    "демпфирование суставов",
                    "base",
                    [
                        _level(
                            "base",
                            "базовое 1x",
                            0.42,
                            baseline=True,
                            factor={"kind": "damping", "scale": 1.0},
                        ),
                        _level(
                            "harsh",
                            "жёсткое 10x",
                            0.24,
                            significant=False,
                            p_holm=0.074,
                            factor={"kind": "damping", "scale": 10.0},
                        ),
                    ],
                )
            ]
        )
        diag = s4.build_diagnosis_payload(payload)
        self.assertEqual(diag["n_significant_improving"], 0)
        self.assertEqual(diag["n_significant_harmful"], 0)

    def test_interpolate_noise_threshold(self) -> None:
        param = _param(
            "sensor_noise",
            "шум датчиков",
            "none",
            [
                _level(
                    "none",
                    "без шума",
                    0.0,
                    baseline=True,
                    factor={"kind": "noise", "q": 0.0, "v": 0.0},
                ),
                _level(
                    "medium",
                    "средний шум",
                    0.0,
                    p_holm=1.0,
                    factor={"kind": "noise", "q": 0.03, "v": 0.3},
                ),
                _level(
                    "strong",
                    "сильный шум",
                    0.98,
                    significant=True,
                    p_holm=1e-13,
                    factor={"kind": "noise", "q": 0.12, "v": 1.2},
                ),
            ],
        )
        crossings = s4.interpolate_threshold(param, 0.10)
        self.assertEqual(len(crossings), 1)
        self.assertEqual(crossings[0]["axis"], "q")
        expected = 0.03 + (0.10 - 0.0) / (0.98 - 0.0) * (0.12 - 0.03)
        self.assertAlmostEqual(crossings[0]["dose"], expected, places=6)
        self.assertEqual(crossings[0]["note"], "hypothesis")

    def test_top3_unique_params_and_verify_confirmed(self) -> None:
        payload = _payload(
            [
                _param(
                    "floor_contact",
                    "мягкость контакта пола",
                    "baseline",
                    [
                        _level(
                            "baseline",
                            "базовый solref",
                            0.0,
                            baseline=True,
                            factor={"kind": "contact", "preset": "baseline"},
                        ),
                        _level(
                            "soft",
                            "податливый пол",
                            1.0,
                            significant=True,
                            p_holm=1e-14,
                            factor={
                                "kind": "contact",
                                "preset": "soft",
                                "solref": [0.08, 0.5],
                            },
                        ),
                    ],
                ),
                _param(
                    "sensor_noise",
                    "шум датчиков",
                    "none",
                    [
                        _level(
                            "none",
                            "без шума",
                            0.0,
                            baseline=True,
                            factor={"kind": "noise", "q": 0.0, "v": 0.0},
                        ),
                        _level(
                            "medium",
                            "средний шум",
                            0.0,
                            p_holm=1.0,
                            factor={"kind": "noise", "q": 0.03, "v": 0.3},
                        ),
                        _level(
                            "strong",
                            "сильный шум",
                            0.98,
                            significant=True,
                            p_holm=1e-13,
                            factor={"kind": "noise", "q": 0.12, "v": 1.2},
                        ),
                    ],
                ),
                _param(
                    "sensor_bias",
                    "смещение нуля датчиков",
                    "none",
                    [
                        _level(
                            "none",
                            "без смещения",
                            0.0,
                            baseline=True,
                            factor={"kind": "bias", "q": 0.0, "drift": 0.0},
                        ),
                        _level(
                            "weak_pos",
                            "слабый +0.015 рад",
                            0.38,
                            significant=True,
                            p_holm=1e-4,
                            factor={"kind": "bias", "q": 0.015, "drift": 0.0},
                        ),
                        _level(
                            "offset",
                            "смещение нуля +0.05 рад",
                            1.0,
                            significant=True,
                            p_holm=1e-14,
                            factor={"kind": "bias", "q": 0.05, "drift": 0.0},
                        ),
                    ],
                ),
            ]
        )
        diag = s4.build_diagnosis_payload(payload)
        self.assertGreaterEqual(diag["n_significant_harmful"], 3)
        ver = diag["verifications"]
        self.assertEqual(len(ver), 3)
        params = [row["param_id"] for row in ver]
        self.assertEqual(len(set(params)), 3)
        self.assertIn("floor_contact", params)
        self.assertIn("sensor_noise", params)
        self.assertIn("sensor_bias", params)
        bias = next(row for row in ver if row["param_id"] == "sensor_bias")
        self.assertEqual(bias["before_id"], "offset")
        self.assertEqual(bias["after_id"], "weak_pos")
        self.assertEqual(bias["status"], "confirmed")
        noise = next(row for row in ver if row["param_id"] == "sensor_noise")
        self.assertEqual(noise["after_id"], "medium")
        self.assertEqual(noise["status"], "confirmed")
        floor = next(row for row in ver if row["param_id"] == "floor_contact")
        self.assertEqual(floor["after_id"], "baseline")
        self.assertEqual(floor["status"], "confirmed")
        text = s4.build_markdown(payload, [], diag)
        self.assertIn("parameter_sweeps_by_scenario.nominal.parameters[floor_contact].levels[soft]", text)
        self.assertNotIn("on the hardware", text)
        self.assertNotIn("go2.xml", text)
        self.assertNotIn("2 N*m*s/rad", text)
        self.assertNotIn("Add an orientation loop", text)

    def test_cite_mentions_outcomes_rows(self) -> None:
        level = _level(
            "soft",
            "податливый пол",
            1.0,
            significant=True,
            p_holm=1e-14,
            factor={"kind": "contact", "preset": "soft", "solref": [0.08, 0.5]},
        )
        cite = s4.cite_level(
            "parameter_sweeps_by_scenario.nominal",
            "nominal",
            "floor_contact",
            "soft",
            level,
            seed=42,
        )
        self.assertIn("parameter_sweeps_by_scenario.nominal.parameters[floor_contact].levels[soft]", cite)
        self.assertIn("outcomes of episodes 1-50", cite)
        self.assertIn("seed=42", cite)


if __name__ == "__main__":
    unittest.main()
