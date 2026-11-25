"""Парное сравнение двух прогонов по общим seed.

Доли и ДИ — wilson_ci, пары — mcnemar_exact_paired, семейство — holm_adjust
из step3_batch_stress. Свои формулы не дублируются.
"""

from __future__ import annotations

import json
from pathlib import Path

from step3_batch_stress import holm_adjust, mcnemar_exact_paired, wilson_ci

ALPHA = 0.05
WORSE = "significant worsening"
BETTER = "significant improvement"
NOISE = "within noise"
NO_PAIR = "no shared seeds"


def _as_failed(row: dict) -> bool:
    if row.get("failed") is not None:
        return bool(row.get("failed"))
    if row.get("fail") is not None:
        return bool(row.get("fail"))
    if row.get("fail_reason") in ("fall", "stance_loss"):
        return True
    return bool(row.get("fail_fall") or row.get("fail_stance_loss"))


def _seed_of(row: dict) -> int | None:
    if row.get("seed") is None:
        return None
    return int(row["seed"])


def index_by_seed(outcomes: list[dict]) -> tuple[dict[int, bool], dict]:
    """Первое вхождение seed. Повтор и строки без seed не затирают пару."""
    by: dict[int, bool] = {}
    duplicate: list[int] = []
    missing = 0
    for row in outcomes:
        seed = _seed_of(row)
        if seed is None:
            missing += 1
            continue
        if seed in by:
            duplicate.append(seed)
            continue
        by[seed] = _as_failed(row)
    return by, {"n_missing_seed": missing, "duplicate_seeds": duplicate}


def collect_units(payload) -> list[dict]:
    """Сопоставимые уровни: сценарий / фактор / уровень, либо один плоский список."""
    if isinstance(payload, list):
        return [
            {
                "key": ("run", "all", "all"),
                "scenario": "run",
                "param": "all",
                "level": "all",
                "label": "all episodes",
                "outcomes": payload,
            }
        ]
    if not isinstance(payload, dict):
        return []
    by = payload.get("parameter_sweeps_by_scenario") or {}
    if not by and isinstance(payload.get("parameter_sweeps"), dict):
        sweeps = payload["parameter_sweeps"]
        sid = str(sweeps.get("scenario") or "run")
        by = {sid: sweeps}
    units: list[dict] = []
    for sid, sweeps in by.items():
        if not isinstance(sweeps, dict):
            continue
        for param in sweeps.get("parameters") or []:
            pid = str(param.get("id") or "")
            for level in param.get("levels") or []:
                outcomes = level.get("outcomes") or []
                if not outcomes:
                    continue
                lid = str(level.get("id") or "")
                units.append(
                    {
                        "key": (str(sid), pid, lid),
                        "scenario": str(sid),
                        "param": pid,
                        "level": lid,
                        "label": level.get("label") or lid,
                        "outcomes": outcomes,
                    }
                )
    if units:
        return units
    outcomes = payload.get("outcomes") or payload.get("episodes") or []
    if not isinstance(outcomes, list) or not outcomes:
        return []
    label = str(payload.get("label") or payload.get("scenario") or "all episodes")
    return [
        {
            "key": ("run", "all", "all"),
            "scenario": str(payload.get("scenario") or "run"),
            "param": "all",
            "level": "all",
            "label": label,
            "outcomes": outcomes,
        }
    ]


def _rate_block(flags: list[bool]) -> dict:
    n = len(flags)
    k = sum(1 for flag in flags if flag)
    lo, hi = wilson_ci(k, n)
    rate = (k / n) if n else float("nan")
    return {
        "n": n,
        "n_failures": k,
        "rate": rate,
        "ci95": [lo, hi],
    }


def _pair_level(before: dict, after: dict) -> dict:
    a_map, a_notes = index_by_seed(before["outcomes"])
    b_map, b_notes = index_by_seed(after["outcomes"])
    shared = sorted(set(a_map) & set(b_map))
    only_a = sorted(set(a_map) - set(b_map))
    only_b = sorted(set(b_map) - set(a_map))
    a_flags = [a_map[seed] for seed in shared]
    b_flags = [b_map[seed] for seed in shared]
    base = [{"episode": seed, "failed": a_map[seed]} for seed in shared]
    treat = [{"episode": seed, "failed": b_map[seed]} for seed in shared]
    mc = mcnemar_exact_paired(base, treat) if shared else {
        "n_paired": 0,
        "n00": 0,
        "n01": 0,
        "n10": 0,
        "n11": 0,
        "n_discordant": 0,
        "pvalue": None,
    }
    # В таблице Шага 3: n10 = base жив, treat сбой (0->1); n01 = base сбой, treat жив (1->0).
    n_0_to_1 = int(mc["n10"])
    n_1_to_0 = int(mc["n01"])
    block_a = _rate_block(a_flags)
    block_b = _rate_block(b_flags)
    delta = (
        block_b["rate"] - block_a["rate"]
        if shared
        else float("nan")
    )
    warnings = []
    if only_a or only_b:
        warnings.append(
            f"seeds do not fully overlap: only in A {len(only_a)}, only in B {len(only_b)}"
        )
    if a_notes["duplicate_seeds"] or b_notes["duplicate_seeds"]:
        warnings.append(
            "duplicate seeds inside a run were skipped "
            f"(A {len(a_notes['duplicate_seeds'])}, B {len(b_notes['duplicate_seeds'])})"
        )
    if a_notes["n_missing_seed"] or b_notes["n_missing_seed"]:
        warnings.append(
            "rows without a seed were skipped "
            f"(A {a_notes['n_missing_seed']}, B {b_notes['n_missing_seed']})"
        )
    return {
        "scenario": before["scenario"],
        "param": before["param"],
        "level": before["level"],
        "label": before["label"],
        "n_shared": len(shared),
        "n_only_a": len(only_a),
        "n_only_b": len(only_b),
        "before": block_a,
        "after": block_b,
        "delta": delta,
        "delta_pp": delta * 100.0 if shared else float("nan"),
        "n_0_to_1": n_0_to_1,
        "n_1_to_0": n_1_to_0,
        "n_discordant": int(mc["n_discordant"]),
        "p_mcnemar": mc["pvalue"],
        "p_holm": None,
        "significant": False,
        "direction": NO_PAIR if not shared else NOISE,
        "warnings": warnings,
    }


def _pick(*values):
    for value in values:
        if value is not None:
            return value
    return None


def run_context(payload) -> dict:
    """Контроллер и эталон стойки, если они есть в JSON прогона."""
    if not isinstance(payload, dict):
        return {}
    stance = payload.get("stance_reference")
    if not isinstance(stance, dict):
        stance = {}
    config = payload.get("config")
    if not isinstance(config, dict):
        config = {}
    return {
        "controller": _pick(payload.get("controller"), stance.get("controller"), config.get("controller")),
        "pd": _pick(payload.get("pd"), stance.get("pd"), payload.get("mode")),
        "kp": _pick(payload.get("kp"), stance.get("kp"), config.get("kp")),
        "kd": _pick(payload.get("kd"), stance.get("kd"), config.get("kd")),
        "z_ref": _pick(stance.get("z_ref"), payload.get("stance_ref_z"), config.get("stance_ref_z")),
        "stance_loss_frac": _pick(
            stance.get("stance_loss_frac"),
            payload.get("stance_loss_frac"),
            config.get("stance_loss_frac"),
        ),
        "stance_loss_z_limit": stance.get("stance_loss_z_limit"),
        "z_threshold": stance.get("z_threshold"),
        "slip_threshold": stance.get("slip_threshold"),
    }


def _as_float(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def context_warnings(before: dict, after: dict) -> list[str]:
    notes = []
    z_a = _as_float(before.get("z_ref"))
    z_b = _as_float(after.get("z_ref"))
    if z_a is not None and z_b is not None and abs(z_a - z_b) > 1e-4:
        notes.append(f"z_ref differs: A {z_a:.4f} m, B {z_b:.4f} m")
    thr_a = _as_float(before.get("z_threshold"))
    thr_b = _as_float(after.get("z_threshold"))
    if thr_a is not None and thr_b is not None and abs(thr_a - thr_b) > 1e-4:
        notes.append(
            f"stance_loss z threshold differs: A z<{thr_a:.4f} m, B z<{thr_b:.4f} m"
        )
    slip_a = _as_float(before.get("slip_threshold"))
    slip_b = _as_float(after.get("slip_threshold"))
    if slip_a is not None and slip_b is not None and abs(slip_a - slip_b) > 1e-4:
        notes.append(
            f"stance_loss slip threshold differs: A slip>{slip_a:.4f} m, B slip>{slip_b:.4f} m"
        )
    return notes


def compare_runs(before_payload, after_payload) -> dict:
    before_units = {unit["key"]: unit for unit in collect_units(before_payload)}
    after_units = {unit["key"]: unit for unit in collect_units(after_payload)}
    keys = sorted(set(before_units) & set(after_units))
    only_before = sorted(set(before_units) - set(after_units))
    only_after = sorted(set(after_units) - set(before_units))
    rows = [_pair_level(before_units[key], after_units[key]) for key in keys]
    family_idx = [i for i, row in enumerate(rows) if row["p_mcnemar"] is not None]
    pvalues = [float(rows[i]["p_mcnemar"]) for i in family_idx]
    p_adj, reject = holm_adjust(pvalues, alpha=ALPHA)
    for i, padj, sig in zip(family_idx, p_adj, reject):
        row = rows[i]
        row["p_holm"] = padj
        row["significant"] = bool(sig)
        if not sig:
            row["direction"] = NOISE
        elif row["n_0_to_1"] > row["n_1_to_0"]:
            row["direction"] = WORSE
        elif row["n_1_to_0"] > row["n_0_to_1"]:
            row["direction"] = BETTER
        else:
            row["direction"] = NOISE
    unmatched = []
    if only_before or only_after:
        unmatched.append(
            "levels without a pair: "
            f"only in A {len(only_before)}, only in B {len(only_after)}"
        )
    seed_warn = [note for row in rows for note in row["warnings"]]
    before_meta = run_context(before_payload)
    after_meta = run_context(after_payload)
    strongest = sorted(
        [row for row in rows if row["significant"]],
        key=lambda row: abs(row["delta_pp"]),
        reverse=True,
    )
    return {
        "alpha": ALPHA,
        "n_levels": len(rows),
        "n_tests": len(family_idx),
        "before_meta": before_meta,
        "after_meta": after_meta,
        "rows": rows,
        "strongest": [
            {
                "scenario": row["scenario"],
                "param": row["param"],
                "level": row["level"],
                "label": row["label"],
                "direction": row["direction"],
                "delta_pp": row["delta_pp"],
                "p_holm": row["p_holm"],
                "n_0_to_1": row["n_0_to_1"],
                "n_1_to_0": row["n_1_to_0"],
            }
            for row in strongest[:5]
        ],
        "warnings": context_warnings(before_meta, after_meta) + unmatched + seed_warn,
    }


def _fmt_p(value) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4g}"


def _fmt_ci(block: dict) -> str:
    lo, hi = block["ci95"]
    if block["n"] <= 0:
        return "n/a"
    return f"[{lo * 100:.1f}, {hi * 100:.1f}]"


def _fmt_meta_num(value, unit: str) -> str:
    number = _as_float(value)
    if number is None:
        return "n/a"
    return f"{number:.4f} {unit}"


def _fmt_context(tag: str, meta: dict) -> str:
    if not meta:
        return f"{tag}: JSON has no controller or stance-reference metadata"
    ctrl = meta.get("controller") or "n/a"
    pd_name = meta.get("pd") or "n/a"
    kp = meta.get("kp")
    kd = meta.get("kd")
    kp_txt = f"{float(kp):g}" if _as_float(kp) is not None else "n/a"
    kd_txt = f"{float(kd):g}" if _as_float(kd) is not None else "n/a"
    z_thr = meta.get("z_threshold")
    slip = meta.get("slip_threshold")
    thr = "n/a"
    if _as_float(z_thr) is not None or _as_float(slip) is not None:
        z_part = f"z<{float(z_thr):.4f} m" if _as_float(z_thr) is not None else "z n/a"
        slip_part = f"slip>{float(slip):.4f} m" if _as_float(slip) is not None else "slip n/a"
        thr = f"{z_part}, {slip_part}"
    return (
        f"{tag}: controller {ctrl}, PD {pd_name}, kp={kp_txt}, kd={kd_txt}, "
        f"z_ref={_fmt_meta_num(meta.get('z_ref'), 'm')}, stance_loss threshold: {thr}"
    )


def format_report(result: dict) -> str:
    lines = [
        "Paired comparison on shared seeds (A before, B after).",
        _fmt_context("A", result.get("before_meta") or {}),
        _fmt_context("B", result.get("after_meta") or {}),
        f"Levels {result['n_levels']}, McNemar tests in the Holm family {result['n_tests']}, alpha={result['alpha']}.",
        "",
    ]
    if not result["rows"]:
        lines.append("No comparable levels.")
    for row in result["rows"]:
        lines.append(
            f"- {row['scenario']} / {row['param']} / {row['level']} ({row['label']}): "
            f"shared seeds {row['n_shared']}"
        )
        lines.append(
            f"    A {row['before']['n_failures']}/{row['before']['n']} "
            f"CI {_fmt_ci(row['before'])}  "
            f"B {row['after']['n_failures']}/{row['after']['n']} "
            f"CI {_fmt_ci(row['after'])}  "
            f"delta {row['delta_pp']:+.1f} pp"
        )
        lines.append(
            f"    0->1 {row['n_0_to_1']}  1->0 {row['n_1_to_0']}  "
            f"p_mcnemar={_fmt_p(row['p_mcnemar'])}  p_holm={_fmt_p(row['p_holm'])}  "
            f"{row['direction']}"
        )
    lines.append("")
    if result["strongest"]:
        lines.append("Strongest significant shifts:")
        for row in result["strongest"]:
            lines.append(
                f"- {row['scenario']} / {row['param']} / {row['level']}: "
                f"{row['direction']}, delta {row['delta_pp']:+.1f} pp, "
                f"0->1 {row['n_0_to_1']}, 1->0 {row['n_1_to_0']}, "
                f"p_holm={_fmt_p(row['p_holm'])}"
            )
    else:
        lines.append("No significant shifts after Holm.")
    if result["warnings"]:
        lines.append("")
        lines.append("Warnings:")
        for note in result["warnings"]:
            lines.append(f"- {note}")
    return "\n".join(lines)


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _forbidden(path: Path) -> bool:
    name = path.name.lower()
    return name in {"batch_failures.json", "final_diagnosis.md"}


def run_compare_files(before_path: str, after_path: str, out_path: str | None) -> dict:
    before = Path(before_path)
    after = Path(after_path)
    if not before.is_file():
        raise FileNotFoundError(f"run A file not found: {before}")
    if not after.is_file():
        raise FileNotFoundError(f"run B file not found: {after}")
    result = compare_runs(_load(before), _load(after))
    text = format_report(result)
    print(text)
    if out_path:
        dest = Path(out_path)
        if _forbidden(dest):
            raise ValueError(
                f"refusing to write {dest.name}: paired comparison does not overwrite "
                "batch_failures.json or final_diagnosis.md"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\njson: {dest}")
    return result
