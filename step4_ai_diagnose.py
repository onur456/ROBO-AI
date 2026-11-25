import json
import math
import os
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

FAILURES_PATH = Path("reports") / "batch_failures.json"
REPORT_PATH = Path("reports") / "final_diagnosis.md"
RECS_JSON_PATH = Path("reports") / "step4_recommendations.json"
GEMINI_MODEL = "gemini-2.0-flash"
FAIL_RATE_THRESHOLD = 0.10
VERIFY_TOP_K = 3
WILSON_Z = 1.959963984540054

ROLL_RATE_THRESHOLD = 2.5
SLIP_FRICTION_THRESHOLD = 0.15
DEFAULT_IMPULSE_START = 450
DEFAULT_IMPULSE_END = 500

CAT_OVERTURN = "Опрокидывание от бокового импульса"
CAT_SLIP = "Разъезжание лап на скользкой поверхности"
CAT_OTHER = "Прочее / смешанный сбой"

PHASE_BEFORE = "before_impact"
PHASE_DURING = "during_impact"
PHASE_AFTER = "after_impact"
PHASE_LABELS = {
    PHASE_BEFORE: "до удара (ещё до включения Fy)",
    PHASE_DURING: "во время удара (шаги 450..500)",
    PHASE_AFTER: "после удара",
}


def get_api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or None


def window_metrics(window: list[dict]) -> dict:
    if not window:
        return {
            "max_abs_roll_rate": 0.0,
            "max_body_speed": 0.0,
            "z_first": None,
            "z_last": None,
        }
    qvel = np.asarray([row["qvel_head"] for row in window], dtype=float)
    z = np.asarray([row["z"] for row in window], dtype=float)
    roll_rate = np.abs(qvel[:, 3])
    body_speed = np.linalg.norm(qvel[:, :3], axis=1)
    return {
        "max_abs_roll_rate": float(np.max(roll_rate)),
        "max_body_speed": float(np.max(body_speed)),
        "z_first": float(z[0]),
        "z_last": float(z[-1]),
    }


def _as_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def episode_force(ep: dict) -> float:
    return _as_float(ep.get("lateral_force", ep.get("fy", ep.get("lateral_vel", 0.0))))


def episode_uses_force(ep: dict) -> bool:
    return ep.get("lateral_force", ep.get("fy")) is not None


def impulse_bounds(ep: dict, payload: dict | None = None) -> tuple[int, int]:
    src = payload or {}
    start = _as_int(
        ep.get("impulse_start", src.get("impulse_start", DEFAULT_IMPULSE_START)),
        DEFAULT_IMPULSE_START,
    )
    end = _as_int(
        ep.get("impulse_end", src.get("impulse_end", DEFAULT_IMPULSE_END)),
        DEFAULT_IMPULSE_END,
    )
    return start, end


def impact_phase_of(fail_step: int, start: int, end: int, stored: object = None) -> str:
    if stored in {PHASE_BEFORE, PHASE_DURING, PHASE_AFTER}:
        return str(stored)
    if fail_step < start:
        return PHASE_BEFORE
    if fail_step <= end:
        return PHASE_DURING
    return PHASE_AFTER


def _steps_word(n: int) -> str:
    count = abs(int(n))
    mod10 = count % 10
    mod100 = count % 100
    if mod10 == 1 and mod100 != 11:
        return "шаг"
    if 2 <= mod10 <= 4 and not 12 <= mod100 <= 14:
        return "шага"
    return "шагов"


def fail_vs_impulse(fail_step: int, start: int, end: int) -> str:
    if fail_step < start:
        delta = start - fail_step
        return f"до удара ({delta} {_steps_word(delta)} до начала {start}..{end})"
    if fail_step <= end:
        delta = fail_step - start
        return f"во время удара (+{delta} {_steps_word(delta)} от начала {start}..{end})"
    delta = fail_step - end
    return f"после удара (+{delta} {_steps_word(delta)} после конца {start}..{end})"


def format_push(ep: dict) -> str:
    value = _as_float(ep.get("lateral_force"))
    if ep.get("push_is_force"):
        return f"{value:+.1f} Н"
    return f"{value:+.3f} м/с"


def classify_failure(friction: float, max_abs_roll_rate: float) -> str:
    if max_abs_roll_rate > ROLL_RATE_THRESHOLD:
        return CAT_OVERTURN
    if friction < SLIP_FRICTION_THRESHOLD and max_abs_roll_rate <= ROLL_RATE_THRESHOLD:
        return CAT_SLIP
    return CAT_OTHER


def classify_episodes(payload: dict) -> list[dict]:
    classified = []
    for ep in payload.get("failures") or []:
        metrics = window_metrics(ep.get("window") or [])
        friction = _as_float(ep.get("friction"))
        force = episode_force(ep)
        uses_force = episode_uses_force(ep)
        fail_step = _as_int(ep.get("fail_step"))
        impulse_start, impulse_end = impulse_bounds(ep, payload)
        phase = impact_phase_of(
            fail_step, impulse_start, impulse_end, ep.get("impact_phase")
        )
        category = classify_failure(friction, metrics["max_abs_roll_rate"])
        classified.append(
            {
                "episode": ep.get("episode"),
                "friction": friction,
                "lateral_force": force,
                "push_is_force": uses_force,
                "fail_step": fail_step,
                "fail_time": _as_float(ep.get("fail_time")),
                "final_z": _as_float(ep.get("final_z")),
                "impulse_start": impulse_start,
                "impulse_end": impulse_end,
                "impact_phase": phase,
                "fail_vs_impulse": fail_vs_impulse(fail_step, impulse_start, impulse_end),
                "fail_reason": ep.get("fail_reason"),
                "category": category,
                **metrics,
            }
        )
    return classified


def is_before_impact(ep: dict) -> bool:
    if ep.get("impact_phase") == PHASE_BEFORE:
        return True
    if ep.get("impact_phase") in {PHASE_DURING, PHASE_AFTER}:
        return False
    start = _as_int(ep.get("impulse_start"), DEFAULT_IMPULSE_START)
    return _as_int(ep.get("fail_step")) < start


def category_stats(episodes: list[dict]) -> list[tuple[str, int, int, float]]:
    counts = Counter(ep["category"] for ep in episodes)
    before = Counter(ep["category"] for ep in episodes if is_before_impact(ep))
    total = len(episodes)
    rows = []
    for name in (CAT_OVERTURN, CAT_SLIP, CAT_OTHER):
        n = counts.get(name, 0)
        n_before = before.get(name, 0)
        pct = (100.0 * n / total) if total else 0.0
        rows.append((name, n, n_before, pct))
    return rows


def impact_window_stats(payload: dict, episodes: list[dict]) -> dict:
    start = _as_int(payload.get("impulse_start"), DEFAULT_IMPULSE_START)
    n_all = payload.get("n_episodes")
    n_survived = payload.get("n_survived")
    n_before = sum(1 for ep in episodes if is_before_impact(ep))
    n_at_or_after = len(episodes) - n_before
    if isinstance(n_survived, (int, float)):
        n_reached = int(n_survived) + n_at_or_after
    elif isinstance(n_all, (int, float)):
        n_reached = max(0, int(n_all) - n_before)
    else:
        n_reached = n_at_or_after
    share = (n_at_or_after / n_reached) if n_reached else 0.0
    return {
        "impulse_start": start,
        "n_before": n_before,
        "n_at_or_after": n_at_or_after,
        "n_reached": n_reached,
        "n_survived": n_survived,
        "share_among_reached": share,
    }


def phase_stats(episodes: list[dict]) -> list[tuple[str, str, int, float]]:
    counts = Counter(ep.get("impact_phase") for ep in episodes)
    total = len(episodes)
    rows = []
    for phase, label in PHASE_LABELS.items():
        n = counts.get(phase, 0)
        pct = (100.0 * n / total) if total else 0.0
        rows.append((phase, label, n, pct))
    return rows


def mixed_comment(ep: dict) -> str:
    """Пояснение без изменения правил классификации: что выглядит преобладающим."""
    roll = _as_float(ep.get("max_abs_roll_rate"))
    friction = _as_float(ep.get("friction"))
    force = abs(_as_float(ep.get("lateral_force")))
    speed = _as_float(ep.get("max_body_speed"))
    z_first = ep.get("z_first")
    z_last = ep.get("z_last")
    z_drop = 0.0
    if z_first is not None and z_last is not None:
        z_drop = float(z_first) - float(z_last)
    before = is_before_impact(ep)
    near_overturn = roll >= 1.2 or (force >= 200.0 and roll >= 0.8)
    sag_like = friction < 0.40 and roll <= 1.2 and (before or z_drop >= 0.03 or speed < 0.35)
    if sag_like and not near_overturn:
        return (
            "Преобладает низкое или среднее трение с просадкой корпуса "
            "(крен слабый, это ближе к разъезжанию, но μ ≥ 0.15, поэтому нет отдельной категории)."
        )
    if near_overturn:
        return (
            "Преобладает боковая сила/крен "
            "(почти опрокидывание, но |qvel[3]| не превысил порог 2.5 рад/с)."
        )
    return (
        "Смешаны боковой импульс и просадка на среднем трении, явного лидера нет."
    )


def explain_example(ep: dict) -> str:
    push = format_push(ep)
    timing = ep.get("fail_vs_impulse", "")
    if ep["category"] == CAT_OVERTURN:
        return (
            f"Сильный боковой импульс ({push}) раскрутил корпус: "
            f"макс. |qvel[3]| = {ep['max_abs_roll_rate']:.2f} рад/с (> {ROLL_RATE_THRESHOLD}). "
            f"Сбой {timing}. "
            "Суставной PD держит стойку ног, но не гасит крен, поэтому робот опрокидывается."
        )
    if ep["category"] == CAT_SLIP:
        return (
            f"Трение пола низкое ({ep['friction']:.3f} < {SLIP_FRICTION_THRESHOLD}), "
            f"крен умеренный (|qvel[3]| = {ep['max_abs_roll_rate']:.2f} рад/с). "
            f"Сбой {timing}. "
            "Опоры проскальзывают, ноги разъезжаются, и высота корпуса падает без резкого опрокидывания."
        )
    return (
        f"Смешанный случай: friction={ep['friction']:.3f}, "
        f"толчок {push}, "
        f"|qvel[3]|={ep['max_abs_roll_rate']:.2f} рад/с, "
        f"скорость корпуса {ep['max_body_speed']:.2f} м/с. "
        f"Сбой {timing}. {mixed_comment(ep)}"
    )


def _sort_category(name: str, items: list[dict]) -> list[dict]:
    if name == CAT_OVERTURN:
        return sorted(
            items,
            key=lambda ep: (ep["max_abs_roll_rate"], abs(ep["lateral_force"])),
            reverse=True,
        )
    if name == CAT_SLIP:
        return sorted(items, key=lambda ep: (ep["friction"], -ep["max_body_speed"]))
    return sorted(
        items,
        key=lambda ep: (abs(ep["lateral_force"]), ep["max_body_speed"]),
        reverse=True,
    )


def pick_examples(episodes: list[dict], limit: int = 3) -> list[dict]:
    if not episodes:
        return []

    queues: dict[str, deque[dict]] = {}
    for name in (CAT_OVERTURN, CAT_SLIP, CAT_OTHER):
        items = [ep for ep in episodes if ep["category"] == name]
        if items:
            queues[name] = deque(_sort_category(name, items))
    category_order = list(queues)

    picked: list[dict] = []
    used_ids: set = set()

    def take(name: str) -> dict | None:
        queue = queues.get(name)
        if not queue:
            return None
        while queue:
            ep = queue.popleft()
            ep_id = ep.get("episode")
            if ep_id in used_ids:
                continue
            used_ids.add(ep_id)
            return ep
        return None

    for name in category_order:
        if len(picked) >= limit:
            break
        example = take(name)
        if example is not None:
            picked.append(example)

    while len(picked) < limit:
        last_cat = picked[-1]["category"] if picked else None
        preferred = [name for name in category_order if name != last_cat]
        fallback = [name for name in category_order if name == last_cat]
        example = None
        for name in preferred + fallback:
            example = take(name)
            if example is not None:
                break
        if example is None:
            break
        picked.append(example)

    return picked[:limit]


def wilson_ci(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / den
    return (max(0.0, center - half), min(1.0, center + half))


def sweep_level_rate(level: dict) -> float:
    return _as_float(level.get("failure_rate"))


def level_counts(level: dict) -> tuple[int, int]:
    n = _as_int(level.get("n_episodes"))
    k = _as_int(level.get("n_failures"))
    if n <= 0:
        outcomes = level.get("outcomes") or []
        if outcomes:
            n = len(outcomes)
            k = sum(1 for row in outcomes if row.get("failed"))
    return k, n


def level_ci(level: dict) -> tuple[float, float]:
    stored = level.get("failure_rate_ci95")
    if isinstance(stored, (list, tuple)) and len(stored) == 2:
        try:
            return (float(stored[0]), float(stored[1]))
        except (TypeError, ValueError):
            pass
    k, n = level_counts(level)
    return wilson_ci(k, n)


def format_pct(rate: float) -> str:
    if rate != rate:  # NaN
        return "н/д"
    return f"{100.0 * rate:.1f}%"


def format_ci(ci: tuple[float, float]) -> str:
    lo, hi = ci
    if lo != lo or hi != hi:
        return "н/д"
    return f"[{100.0 * lo:.1f}, {100.0 * hi:.1f}]%"


def format_p(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return "—"


def delta_vs_baseline(level: dict, base_rate: float) -> float:
    if level.get("delta_vs_baseline") is not None:
        return _as_float(level.get("delta_vs_baseline"))
    return sweep_level_rate(level) - base_rate


def effect_kind(level: dict, baseline_id: str | None, base_rate: float) -> str:
    if baseline_id is not None and level.get("id") == baseline_id:
        return "baseline"
    if not level.get("significant"):
        return "noise"
    delta = delta_vs_baseline(level, base_rate)
    if delta > 1e-12:
        return "harmful"
    if delta < -1e-12:
        return "improving"
    return "noise"


def effect_direction_ru(kind: str) -> str:
    if kind == "harmful":
        return "усиление сбоев"
    if kind == "improving":
        return "ослабление сбоев"
    if kind == "baseline":
        return "норма"
    return "в пределах шума"


def axis_samples(level: dict, param_id: str | None = None) -> list[tuple[str, float]]:
    """Скаляры уровня из payload фактора. Не подставляет зашитые xml-числа."""
    _ = param_id
    factor = level.get("factor") or {}
    kind = str(factor.get("kind") or "")
    if kind == "com_offset":
        off = list(factor.get("offset") or [0.0, 0.0, 0.0])
        while len(off) < 3:
            off.append(0.0)
        names = ("x", "y", "z")
        if param_id == "com_x":
            return [("x", float(off[0]))]
        if param_id == "com_y":
            return [("y", float(off[1]))]
        if param_id == "com_z":
            return [("z", float(off[2]))]
        return [(names[i], float(off[i])) for i in range(3)]
    if kind == "torque_limit":
        return [("scale", float(factor.get("scale", 1.0)))]
    if kind == "delay":
        return [("steps", float(factor.get("steps", 0)))]
    if kind == "damping":
        return [("scale", float(factor.get("scale", 1.0)))]
    if kind == "contact":
        solref = factor.get("solref")
        if isinstance(solref, (list, tuple)) and solref:
            return [("solref0", float(solref[0]))]
        return []
    if kind == "noise":
        return [("q", float(factor.get("q") or 0.0))]
    if kind == "ctrl_hold":
        return [("steps", float(factor.get("steps", 1)))]
    if kind == "bias":
        q = float(factor.get("q") or 0.0)
        drift = float(factor.get("drift") or 0.0)
        out: list[tuple[str, float]] = []
        if abs(drift) <= 1e-15:
            out.append(("q", q))
        if abs(q) <= 1e-15:
            out.append(("drift", drift))
        return out
    return []


def cite_level(
    root: str,
    scenario_id: str,
    param_id: str,
    level_id: str,
    level: dict | None = None,
    seed=None,
) -> str:
    if root.startswith("parameter_sweeps_by_scenario"):
        path = f"{root}.parameters[{param_id}].levels[{level_id}]"
    elif root == "parameter_sweeps":
        path = f"parameter_sweeps.parameters[{param_id}].levels[{level_id}]"
    else:
        path = f"{root}.parameters[{param_id}].levels[{level_id}]"
    bits = [f"`{path}`"]
    if seed is not None:
        bits.append(f"seed={seed}")
    if level:
        k, n = level_counts(level)
        bits.append(f"{k}/{n} сбоев")
        outcomes = level.get("outcomes") or []
        if outcomes:
            episodes = [_as_int(row.get("episode")) for row in outcomes]
            bits.append(f"outcomes эпизодов {min(episodes)}–{max(episodes)}")
    return ", ".join(bits)


def iter_sweep_blocks(payload: dict):
    by_scenario = payload.get("parameter_sweeps_by_scenario") or {}
    if by_scenario:
        order = ("nominal", "moderate", "extreme", "default")
        seen: set[str] = set()
        for sid in list(order) + list(by_scenario):
            if sid in seen or sid not in by_scenario:
                continue
            seen.add(sid)
            yield sid, by_scenario[sid], f"parameter_sweeps_by_scenario.{sid}"
        return
    sweeps = payload.get("parameter_sweeps") or {}
    if sweeps:
        sid = str(sweeps.get("scenario") or "default")
        yield sid, sweeps, "parameter_sweeps"


def interpolate_threshold(
    param: dict,
    threshold: float = FAIL_RATE_THRESHOLD,
) -> list[dict]:
    param_id = str(param.get("id") or "")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for level in param.get("levels") or []:
        rate = sweep_level_rate(level)
        for axis, dose in axis_samples(level, param_id):
            grouped[axis].append(
                {
                    "id": level.get("id"),
                    "label": level.get("label") or level.get("id"),
                    "dose": float(dose),
                    "rate": rate,
                    "n_failures": level_counts(level)[0],
                    "n_episodes": level_counts(level)[1],
                }
            )
    crossings: list[dict] = []
    for axis, points in grouped.items():
        points = sorted(points, key=lambda row: (row["dose"], row["rate"]))
        for left, right in zip(points, points[1:]):
            if left["id"] == right["id"]:
                continue
            r0, r1 = left["rate"], right["rate"]
            d0, d1 = left["dose"], right["dose"]
            if abs(d1 - d0) < 1e-15:
                continue
            lo_r, hi_r = (r0, r1) if r0 <= r1 else (r1, r0)
            if not (lo_r < threshold <= hi_r):
                continue
            t = (threshold - r0) / (r1 - r0)
            dose_star = d0 + t * (d1 - d0)
            crossings.append(
                {
                    "axis": axis,
                    "dose": dose_star,
                    "threshold": threshold,
                    "left_id": left["id"],
                    "left_label": left["label"],
                    "left_dose": d0,
                    "left_rate": r0,
                    "right_id": right["id"],
                    "right_label": right["label"],
                    "right_dose": d1,
                    "right_rate": r1,
                    "measured": False,
                    "note": "гипотеза",
                }
            )
    return crossings


def mcnemar_exact_paired(base_outcomes: list[dict], treat_outcomes: list[dict]) -> dict:
    """Точный Макнемар на парных исходах. Только для проверки Шага 4, статистику Шага 3 не меняет."""
    from statsmodels.stats.contingency_tables import mcnemar

    base_by = {int(o["episode"]): bool(o["failed"]) for o in base_outcomes}
    treat_by = {int(o["episode"]): bool(o["failed"]) for o in treat_outcomes}
    keys = sorted(set(base_by) & set(treat_by))
    n00 = n01 = n10 = n11 = 0
    for key in keys:
        b = base_by[key]
        t = treat_by[key]
        if b and t:
            n11 += 1
        elif b and not t:
            n01 += 1
        elif (not b) and t:
            n10 += 1
        else:
            n00 += 1
    table = np.array([[n00, n01], [n10, n11]], dtype=int)
    n_disc = n01 + n10
    if n_disc == 0 or len(keys) == 0:
        pvalue = 1.0
    else:
        pvalue = float(mcnemar(table, exact=True).pvalue)
    return {
        "test": "mcnemar_exact",
        "n_paired": len(keys),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "n_discordant": n_disc,
        "pvalue": pvalue,
    }


def safer_neighbor(param: dict, harmful: dict) -> tuple[dict | None, str]:
    levels = list(param.get("levels") or [])
    if not levels:
        return None, "none"
    baseline_id = param.get("baseline")
    base = next((lv for lv in levels if lv.get("id") == baseline_id), levels[0])
    param_id = str(param.get("id") or "")
    harmful_axes = dict(axis_samples(harmful, param_id))
    if not harmful_axes:
        return base, "baseline"
    axis, hdose = max(harmful_axes.items(), key=lambda kv: abs(kv[1]))
    base_axes = dict(axis_samples(base, param_id))
    if axis not in base_axes:
        return base, "baseline"
    bdose = base_axes[axis]
    denom = hdose - bdose
    if abs(denom) < 1e-15:
        return base, "baseline"
    candidates: list[tuple[float, dict]] = []
    for level in levels:
        if level.get("id") == harmful.get("id"):
            continue
        axes = dict(axis_samples(level, param_id))
        if axis not in axes:
            continue
        t = (axes[axis] - bdose) / denom
        if -1e-12 <= t < 1.0 - 1e-12:
            candidates.append((t, level))
    if not candidates:
        return base, "baseline"
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], "neighbor"


def collect_significant_findings(payload: dict, threshold: float = FAIL_RATE_THRESHOLD) -> list[dict]:
    findings: list[dict] = []
    for sid, sweeps, root in iter_sweep_blocks(payload):
        seed = sweeps.get("seed")
        n_tests = (sweeps.get("multiple_testing") or {}).get("n_tests")
        for param in sweeps.get("parameters") or []:
            param_id = str(param.get("id") or "")
            param_name = param.get("name") or param_id
            baseline_id = param.get("baseline")
            levels = list(param.get("levels") or [])
            if not levels:
                continue
            base = next((lv for lv in levels if lv.get("id") == baseline_id), levels[0])
            base_rate = sweep_level_rate(base)
            crossings = interpolate_threshold(param, threshold)
            for level in levels:
                kind = effect_kind(level, baseline_id, base_rate)
                if kind not in {"harmful", "improving"}:
                    continue
                k, n = level_counts(level)
                ci = level_ci(level)
                level_axes = {axis for axis, _ in axis_samples(level, param_id)}
                level_crossings = [row for row in crossings if row["axis"] in level_axes]
                findings.append(
                    {
                        "scenario_id": sid,
                        "scenario_label": sweeps.get("scenario_label") or sid,
                        "root": root,
                        "seed": seed,
                        "holm_n_tests": n_tests,
                        "param_id": param_id,
                        "param_name": param_name,
                        "baseline_id": baseline_id,
                        "baseline_label": base.get("label") or baseline_id,
                        "baseline_rate": base_rate,
                        "baseline_ci": list(level_ci(base)),
                        "level_id": level.get("id"),
                        "level_label": level.get("label") or level.get("id"),
                        "kind": kind,
                        "direction": effect_direction_ru(kind),
                        "rate": sweep_level_rate(level),
                        "n_failures": k,
                        "n_episodes": n,
                        "ci95": list(ci),
                        "delta": delta_vs_baseline(level, base_rate),
                        "p_mcnemar": level.get("p_mcnemar"),
                        "p_holm": level.get("p_holm"),
                        "cite": cite_level(root, sid, param_id, str(level.get("id")), level, seed),
                        "baseline_cite": cite_level(
                            root, sid, param_id, str(base.get("id")), base, seed
                        ),
                        "crossings": level_crossings,
                        "factor": level.get("factor"),
                    }
                )
    return findings


def finding_sort_key(item: dict) -> tuple:
    return (
        -abs(_as_float(item.get("delta"))),
        -_as_float(item.get("rate")),
        _as_float(item.get("p_holm"), 1.0),
        0 if item.get("scenario_id") == "nominal" else 1,
        str(item.get("param_id") or ""),
        str(item.get("level_id") or ""),
    )


def select_top_weaknesses(findings: list[dict], k: int = VERIFY_TOP_K) -> list[dict]:
    harmful = [item for item in findings if item.get("kind") == "harmful"]
    harmful.sort(key=finding_sort_key)
    picked: list[dict] = []
    seen_params: set[str] = set()
    for item in harmful:
        pid = str(item.get("param_id") or "")
        if pid in seen_params:
            continue
        picked.append(item)
        seen_params.add(pid)
        if len(picked) >= k:
            return picked
    for item in harmful:
        if item in picked:
            continue
        picked.append(item)
        if len(picked) >= k:
            break
    return picked


def lookup_param_level(payload: dict, finding: dict) -> tuple[dict | None, dict | None]:
    for sid, sweeps, root in iter_sweep_blocks(payload):
        if sid != finding.get("scenario_id") or root != finding.get("root"):
            continue
        for param in sweeps.get("parameters") or []:
            if param.get("id") != finding.get("param_id"):
                continue
            level = next(
                (lv for lv in (param.get("levels") or []) if lv.get("id") == finding.get("level_id")),
                None,
            )
            return param, level
    return None, None


def verify_status(
    before: dict,
    after: dict,
    finding: dict,
    mcnemar: dict | None,
) -> str:
    before_rate = sweep_level_rate(before)
    after_rate = sweep_level_rate(after)
    if after_rate >= before_rate - 1e-12:
        return "не подтверждено"
    after_is_baseline = after.get("id") == finding.get("baseline_id")
    if after_is_baseline and finding.get("kind") == "harmful" and finding.get("p_holm") is not None:
        if _as_float(finding.get("p_holm"), 1.0) <= 0.05:
            return "подтверждено"
    before_ci = level_ci(before)
    after_ci = level_ci(after)
    if after_ci[1] < before_ci[0]:
        return "подтверждено"
    p_mc = None if mcnemar is None else mcnemar.get("pvalue")
    if p_mc is not None and float(p_mc) < 0.05:
        return "подтверждено"
    return "гипотеза"


def verify_top_weaknesses(payload: dict, findings: list[dict], k: int = VERIFY_TOP_K) -> list[dict]:
    rows: list[dict] = []
    for finding in select_top_weaknesses(findings, k):
        param, before = lookup_param_level(payload, finding)
        if param is None or before is None:
            rows.append(
                {
                    "finding": finding,
                    "status": "гипотеза",
                    "reason": "в JSON нет строк уровня для проверки",
                }
            )
            continue
        after, how = safer_neighbor(param, before)
        if after is None:
            rows.append(
                {
                    "finding": finding,
                    "before": before,
                    "status": "гипотеза",
                    "reason": "нет измеренного уровня в безопасную сторону",
                }
            )
            continue
        mc = None
        before_out = before.get("outcomes") or []
        after_out = after.get("outcomes") or []
        if before_out and after_out:
            mc = mcnemar_exact_paired(after_out, before_out)
        status = verify_status(before, after, finding, mc)
        after_k, after_n = level_counts(after)
        before_k, before_n = level_counts(before)
        rows.append(
            {
                "finding": finding,
                "before_id": before.get("id"),
                "before_label": before.get("label") or before.get("id"),
                "before_rate": sweep_level_rate(before),
                "before_ci": list(level_ci(before)),
                "before_n_failures": before_k,
                "before_n_episodes": before_n,
                "before_cite": finding["cite"],
                "after_id": after.get("id"),
                "after_label": after.get("label") or after.get("id"),
                "after_rate": sweep_level_rate(after),
                "after_ci": list(level_ci(after)),
                "after_n_failures": after_k,
                "after_n_episodes": after_n,
                "after_cite": cite_level(
                    finding["root"],
                    finding["scenario_id"],
                    finding["param_id"],
                    str(after.get("id")),
                    after,
                    finding.get("seed"),
                ),
                "move": how,
                "mcnemar": mc,
                "status": status,
            }
        )
    return rows


def format_crossings(crossings: list[dict], threshold: float) -> str:
    if not crossings:
        return (
            f"точка, где доля сбоев превышает {format_pct(threshold)}, "
            "по измеренным уровням с общей осью не интерполируется (гипотеза)"
        )
    parts = []
    for cr in crossings:
        parts.append(
            f"порог {format_pct(threshold)} на оси `{cr['axis']}`≈{cr['dose']:.4g} "
            f"между «{cr['left_label']}» ({format_pct(cr['left_rate'])}) и "
            f"«{cr['right_label']}» ({format_pct(cr['right_rate'])}) — гипотеза, "
            "промежуточный уровень не измерялся"
        )
    return "; ".join(parts)


def scenario_heading(payload: dict) -> tuple[str | None, str | None]:
    raw = payload.get("scenario")
    if raw is None or str(raw).strip() == "":
        return None, None
    label = payload.get("scenario_label")
    return str(raw), None if label is None else str(label)


def failure_criterion_label(payload: dict) -> str | None:
    def from_value(value: object) -> str | None:
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, dict):
            fail = value.get("fail")
            if isinstance(fail, str) and fail.strip():
                return fail.strip()
        return None

    found = from_value(payload.get("failure_criterion"))
    if found:
        return found
    for _sid, sweeps, _root in iter_sweep_blocks(payload):
        found = from_value(sweeps.get("failure_criterion"))
        if found:
            return found
    return None


def scenario_blurb(payload: dict) -> str:
    criterion = failure_criterion_label(payload)
    criterion_txt = (
        f"`{criterion}`"
        if criterion
        else "поле `failure_criterion` отсутствует"
    )
    blocks = list(iter_sweep_blocks(payload))
    if blocks:
        names = ", ".join(f"`{sid}`" for sid, _sweeps, _root in blocks)
        return (
            "Автоматический разбор `reports/batch_failures.json`. "
            f"Критерий сбоя: {criterion_txt}. "
            f"Сценарии OFAT в файле: {names}."
        )
    sid, label = scenario_heading(payload)
    if sid is None:
        scenario_txt = "поле `scenario` в корне отсутствует"
    else:
        scenario_txt = f"сценарий `{sid}`" + (f" — {label}" if label else "")
    return (
        "Автоматический разбор `reports/batch_failures.json`. "
        f"Критерий сбоя: {criterion_txt}. {scenario_txt}."
    )


def baseline_fail_reason_lines(payload: dict) -> list[str]:
    """Один базовый уровень на сценарий: эпизоды факторов не складываются."""
    lines: list[str] = []
    for sid, sweeps, root in iter_sweep_blocks(payload):
        params = list(sweeps.get("parameters") or [])
        if not params:
            continue
        param = params[0]
        baseline_id = param.get("baseline")
        levels = list(param.get("levels") or [])
        level = next((lv for lv in levels if lv.get("id") == baseline_id), None)
        if level is None:
            continue
        reasons = level.get("fail_reasons")
        if isinstance(reasons, dict) and reasons:
            reason_txt = ", ".join(
                f"{key}={_as_int(reasons[key])}" for key in sorted(reasons)
            )
        elif isinstance(reasons, dict):
            reason_txt = "fail_reasons пуст"
        else:
            reason_txt = "поле fail_reasons отсутствует"
        k, n = level_counts(level)
        param_id = param.get("id")
        level_id = level.get("id")
        lines.append(
            f"- `{sid}`: {reason_txt}, сбоев {k}/{n} "
            f"(базовый уровень `{param_id}`/`{level_id}`, "
            f"`{root}.parameters[{param_id}].levels[{level_id}].fail_reasons`)."
        )
    return lines


def header_stat_lines(payload: dict) -> list[str]:
    """Шапка только из полей JSON. Корень не подменяет сценарии OFAT."""
    lines: list[str] = []
    blocks = list(iter_sweep_blocks(payload))
    sweep_ids = {sid for sid, _sweeps, _root in blocks}
    if blocks:
        lines.append("- сценарии OFAT:")
        for sid, sweeps, _root in blocks:
            label = sweeps.get("scenario_label")
            label_txt = f" — {label}" if label else ""
            n_ep = sweeps.get("n_episodes")
            n_txt = (
                str(int(n_ep))
                if isinstance(n_ep, (int, float))
                else "поле `n_episodes` отсутствует"
            )
            lines.append(f"  - `{sid}`{label_txt}: эпизодов на уровень {n_txt}")
    root_scenario, root_label = scenario_heading(payload)
    if not blocks:
        if root_scenario is None:
            lines.append("- сценарий: поле `scenario` отсутствует")
        else:
            label_txt = f" — {root_label}" if root_label else ""
            lines.append(f"- сценарий: `{root_scenario}`{label_txt}")
        n_all = payload.get("n_episodes")
        if n_all is None:
            lines.append("- всего прогонов: поле `n_episodes` отсутствует")
        else:
            lines.append(f"- всего прогонов: {n_all}")
    elif root_scenario is not None and root_scenario not in sweep_ids:
        lines.append(
            "- в корне JSON есть отдельная запись сценария, она не входит в блоки OFAT "
            "и в эту шапку как основной сценарий не подставляется"
        )
    kp = payload.get("kp")
    kd = payload.get("kd")
    if (kp is None or kd is None) and blocks:
        kp = blocks[0][1].get("kp", kp)
        kd = blocks[0][1].get("kd", kd)
    if kp is not None or kd is not None:
        lines.append(f"- PD: kp={kp}, kd={kd}")
    if payload.get("impulse_start") is not None and payload.get("impulse_end") is not None:
        lines.append(
            f"- окно удара: шаги {payload['impulse_start']}..{payload['impulse_end']}"
        )
    n_fail_list = len(payload.get("failures") or [])
    lines.append(f"- эпизодов в списке `failures`: {n_fail_list}")
    if payload.get("n_sag") is not None:
        sag = payload.get("n_sag")
        sag_rate = payload.get("sag_rate")
        rate_txt = (
            f" ({float(sag_rate) * 100:.1f}%)"
            if isinstance(sag_rate, (int, float))
            else ""
        )
        lines.append(f"- просадка z<0.20: {sag}{rate_txt}")
    return lines


def stored_fail_reason_lines(episodes: list[dict]) -> list[str]:
    if not episodes:
        return ["- список `failures` пуст, разбивки fall/stance_loss по нему нет."]
    counts: Counter[str] = Counter()
    missing = 0
    for ep in episodes:
        reason = ep.get("fail_reason")
        if reason is None or str(reason).strip() == "":
            missing += 1
        else:
            counts[str(reason)] += 1
    lines: list[str] = []
    for name in ("fall", "stance_loss"):
        if name in counts:
            lines.append(f"- `{name}`: {counts[name]}")
    for name in sorted(counts):
        if name not in {"fall", "stance_loss"}:
            lines.append(f"- `{name}`: {counts[name]} (как записано в `fail_reason`)")
    if missing:
        lines.append(
            f"- без поля `fail_reason`: {missing}. "
            "Причина по окну телеметрии не назначается."
        )
    if not lines:
        lines.append("- в эпизодах нет распознанного `fail_reason`.")
    return lines


def format_level_line(
    level: dict,
    baseline_id: str | None,
    base_rate: float,
    cite: str,
) -> str:
    kind = effect_kind(level, baseline_id, base_rate)
    k, n = level_counts(level)
    rate = sweep_level_rate(level)
    delta = delta_vs_baseline(level, base_rate)
    ci = level_ci(level)
    mark = " — норма" if kind == "baseline" else ""
    extra = ""
    if kind != "baseline":
        extra = (
            f", Δ {delta * 100:+.1f} п.п. ({effect_direction_ru(kind)}), "
            f"p_holm={format_p(level.get('p_holm'))}"
        )
        if kind == "harmful":
            extra += " — значимо, вредный эффект"
        elif kind == "improving":
            extra += " — значимо, улучшающий эффект"
        else:
            extra += " — в пределах шума"
    return (
        f"- **{level.get('label', level.get('id'))}** (`{level.get('id')}`): "
        f"доля {format_pct(rate)} ({k}/{n}), Уилсон 95% CI {format_ci(ci)}"
        f"{extra}{mark}. Источник: {cite}"
    )


def format_one_sweep(
    sweeps: dict,
    heading: str | None = None,
    root: str | None = None,
    threshold: float = FAIL_RATE_THRESHOLD,
) -> list[str]:
    parameters = sweeps.get("parameters") or []
    if not parameters:
        return []

    n_ep = _as_int(sweeps.get("n_episodes"), 0)
    sid = str(sweeps.get("scenario") or "default")
    slabel = sweeps.get("scenario_label") or sid
    seed = sweeps.get("seed")
    root = root or "parameter_sweeps"
    n_tests = (sweeps.get("multiple_testing") or {}).get("n_tests")
    title = heading or "One-factor-at-a-time по параметрам"
    holm_note = (
        f"Выводы ниже по значимости — только после поправки Холма "
        f"(n_tests={n_tests}). "
        if n_tests is not None
        else "Выводы ниже по значимости — только после поправки Холма. "
    )
    lines = [
        "",
        f"## {title}",
        "",
        f"Батч {n_ep} эпизодов на каждый уровень, seed={seed}, "
        f"сценарий `{sid}` — {slabel}. "
        "Доля сбоев, 95% CI Уилсона и p_holm vs норма того же фактора. "
        + holm_note
        + "Направление: усиление сбоев (доля выше нормы) или ослабление сбоев (доля ниже нормы).",
        "",
    ]
    any_harmful = False
    for param in parameters:
        param_id = str(param.get("id") or "")
        name = param.get("name") or param_id
        baseline_id = param.get("baseline")
        levels = list(param.get("levels") or [])
        if not levels:
            continue
        baseline = next((lv for lv in levels if lv.get("id") == baseline_id), levels[0])
        base_rate = sweep_level_rate(baseline)
        lines.append(f"### {name} (`{param_id}`)")
        lines.append("")
        harmful = []
        improving = []
        for level in levels:
            cite = cite_level(root, sid, param_id, str(level.get("id")), level, seed)
            lines.append(format_level_line(level, baseline_id, base_rate, cite))
            kind = effect_kind(level, baseline_id, base_rate)
            if kind == "harmful":
                harmful.append(level)
            elif kind == "improving":
                improving.append(level)
        lines.append("")
        if harmful:
            any_harmful = True
            crossings = interpolate_threshold(param, threshold)
            lines.append("Значимые вредные эффекты (Холм):")
            for level in harmful:
                k, n = level_counts(level)
                ci = level_ci(level)
                lines.append(
                    f"- **{level.get('label', level.get('id'))}**: "
                    f"{effect_direction_ru('harmful')}, доля {format_pct(sweep_level_rate(level))} "
                    f"({k}/{n}), Уилсон 95% CI {format_ci(ci)}, "
                    f"p_holm={format_p(level.get('p_holm'))}, "
                    f"Δ {delta_vs_baseline(level, base_rate) * 100:+.1f} п.п. "
                    f"Источник: {cite_level(root, sid, param_id, str(level.get('id')), level, seed)}."
                )
            lines.append(f"- {format_crossings(crossings, threshold).capitalize()}.")
            lines.append("")
        else:
            lines.append("значимого вредного эффекта не найдено.")
            lines.append("")
        if improving:
            lines.append("Значимые улучшающие эффекты:")
            for level in improving:
                k, n = level_counts(level)
                ci = level_ci(level)
                lines.append(
                    f"- **{level.get('label', level.get('id'))}**: "
                    f"улучшающий эффект, доля {format_pct(sweep_level_rate(level))} "
                    f"({k}/{n}), Уилсон 95% CI {format_ci(ci)}, "
                    f"p_holm={format_p(level.get('p_holm'))}, "
                    f"Δ {delta_vs_baseline(level, base_rate) * 100:+.1f} п.п. "
                    f"Источник: {cite_level(root, sid, param_id, str(level.get('id')), level, seed)}."
                )
            lines.append("")
    if not any_harmful:
        lines += [
            "По этому сценарию значимого вредного эффекта не найдено.",
            "",
        ]
    return lines


def format_sweep_sections(payload: dict, threshold: float = FAIL_RATE_THRESHOLD) -> list[str]:
    blocks = list(iter_sweep_blocks(payload))
    if not blocks:
        return []
    lines: list[str] = []
    for sid, sweeps, root in blocks:
        label = sweeps.get("scenario_label") or sid
        role = {
            "nominal": "контроль",
            "moderate": "чувствительность",
        }.get(sid, sid)
        heading = f"OFAT: {sid} ({role}) — {label}" if sid in {"nominal", "moderate"} else f"OFAT: {sid} — {label}"
        lines += format_one_sweep(sweeps, heading, root=root, threshold=threshold)
    return lines


def slim_finding(item: dict) -> dict:
    return {
        "scenario_id": item.get("scenario_id"),
        "param_id": item.get("param_id"),
        "param_name": item.get("param_name"),
        "level_id": item.get("level_id"),
        "level_label": item.get("level_label"),
        "kind": item.get("kind"),
        "direction": item.get("direction"),
        "rate": item.get("rate"),
        "n_failures": item.get("n_failures"),
        "n_episodes": item.get("n_episodes"),
        "ci95": item.get("ci95"),
        "delta": item.get("delta"),
        "p_holm": item.get("p_holm"),
        "p_mcnemar": item.get("p_mcnemar"),
        "cite": item.get("cite"),
        "crossings": item.get("crossings"),
        "factor": item.get("factor"),
    }


def build_diagnosis_payload(
    payload: dict,
    threshold: float = FAIL_RATE_THRESHOLD,
    top_k: int = VERIFY_TOP_K,
) -> dict:
    findings = collect_significant_findings(payload, threshold)
    verifications = verify_top_weaknesses(payload, findings, top_k)
    harmful = [item for item in findings if item.get("kind") == "harmful"]
    improving = [item for item in findings if item.get("kind") == "improving"]
    recs: list[dict] = []
    if not harmful:
        recs.append(
            {
                "text": (
                    "значимого вредного эффекта не найдено: ни один ненулевой уровень OFAT "
                    "не имеет значимого усиления сбоев после поправки Холма."
                ),
                "cite": "parameter_sweeps_by_scenario.*.parameters[*].levels[*] (significant=false или Δ≤0)",
                "kind": "none",
            }
        )
    for item in harmful:
        recs.append(
            {
                "text": (
                    f"Фактор `{item['param_id']}` ({item['param_name']}), уровень "
                    f"«{item['level_label']}» (`{item['level_id']}`), сценарий `{item['scenario_id']}`: "
                    f"{item['direction']}, доля {format_pct(item['rate'])} "
                    f"({item['n_failures']}/{item['n_episodes']}), "
                    f"Уилсон 95% CI {format_ci(tuple(item['ci95']))}, "
                    f"p_holm={format_p(item['p_holm'])}, "
                    f"Δ {item['delta'] * 100:+.1f} п.п. к норме "
                    f"«{item['baseline_label']}» ({format_pct(item['baseline_rate'])}). "
                    f"{format_crossings(item.get('crossings') or [], threshold).capitalize()}."
                ),
                "cite": item["cite"],
                "kind": "harmful",
                "param_id": item["param_id"],
                "level_id": item["level_id"],
                "scenario_id": item["scenario_id"],
            }
        )
    for item in improving:
        recs.append(
            {
                "text": (
                    f"Фактор `{item['param_id']}` ({item['param_name']}), уровень "
                    f"«{item['level_label']}» (`{item['level_id']}`), сценарий `{item['scenario_id']}`: "
                    f"улучшающий эффект, доля {format_pct(item['rate'])} "
                    f"({item['n_failures']}/{item['n_episodes']}), "
                    f"Уилсон 95% CI {format_ci(tuple(item['ci95']))}, "
                    f"p_holm={format_p(item['p_holm'])}, "
                    f"Δ {item['delta'] * 100:+.1f} п.п. к норме."
                ),
                "cite": item["cite"],
                "kind": "improving",
                "param_id": item["param_id"],
                "level_id": item["level_id"],
                "scenario_id": item["scenario_id"],
            }
        )
    slim_verifications = []
    for row in verifications:
        if "before_id" not in row:
            slim_verifications.append(
                {
                    "status": row.get("status"),
                    "reason": row.get("reason"),
                    "finding": slim_finding(row["finding"]),
                }
            )
            continue
        slim_verifications.append(
            {
                "param_id": row["finding"]["param_id"],
                "scenario_id": row["finding"]["scenario_id"],
                "before_id": row["before_id"],
                "before_label": row["before_label"],
                "before_rate": row["before_rate"],
                "before_ci": row["before_ci"],
                "before_n_failures": row["before_n_failures"],
                "before_n_episodes": row["before_n_episodes"],
                "before_cite": row["before_cite"],
                "after_id": row["after_id"],
                "after_label": row["after_label"],
                "after_rate": row["after_rate"],
                "after_ci": row["after_ci"],
                "after_n_failures": row["after_n_failures"],
                "after_n_episodes": row["after_n_episodes"],
                "after_cite": row["after_cite"],
                "move": row["move"],
                "mcnemar_pvalue": None if not row.get("mcnemar") else row["mcnemar"].get("pvalue"),
                "status": row["status"],
            }
        )
    return {
        "threshold": threshold,
        "n_significant_harmful": len(harmful),
        "n_significant_improving": len(improving),
        "findings": [slim_finding(item) for item in findings],
        "recommendations": recs,
        "verifications": slim_verifications,
        "verifications_raw": verifications,
    }


def format_recommendation_sections(diag: dict, threshold: float = FAIL_RATE_THRESHOLD) -> list[str]:
    lines = [
        "",
        "## Рекомендации (только значимые эффекты после Холма)",
        "",
        "Шаблоны про железо и произвольные числа из кода не используются. "
        "Каждый пункт ссылается на уровень OFAT в `reports/batch_failures.json`.",
        "",
    ]
    recs = diag.get("recommendations") or []
    if not recs:
        lines.append("значимого вредного эффекта не найдено.")
        lines.append("")
    else:
        for rec in recs:
            lines.append(f"- {rec['text']}")
            lines.append(f"  Источник: {rec['cite']}.")
            lines.append("")
    lines += [
        "## Проверка топ-3 значимых слабостей (сдвиг в безопасную сторону)",
        "",
        "Для каждой слабости берётся значимый вредный уровень и соседний уже измеренный "
        "уровень OFAT ближе к норме (те же seed). До и после — доля сбоев и 95% CI Уилсона.",
        "",
    ]
    verifications = diag.get("verifications_raw") or []
    if not verifications:
        lines.append("значимого вредного эффекта не найдено — проверять нечего.")
        lines.append("")
        return lines
    for i, row in enumerate(verifications, start=1):
        finding = row["finding"]
        if "before_id" not in row:
            lines.append(
                f"{i}. `{finding.get('param_id')}` / `{finding.get('level_id')}`: "
                f"{row.get('status')} ({row.get('reason')}). "
                f"Источник: {finding.get('cite')}."
            )
            lines.append("")
            continue
        p_mc = None if not row.get("mcnemar") else row["mcnemar"].get("pvalue")
        lines.append(
            f"{i}. **{finding['param_name']}** (`{finding['param_id']}`), сценарий "
            f"`{finding['scenario_id']}`: «{row['before_label']}» → «{row['after_label']}» "
            f"({row['move']})."
        )
        lines.append(
            f"   - до: доля {format_pct(row['before_rate'])} "
            f"({row['before_n_failures']}/{row['before_n_episodes']}), "
            f"Уилсон 95% CI {format_ci(tuple(row['before_ci']))}. "
            f"Источник: {row['before_cite']}."
        )
        lines.append(
            f"   - после: доля {format_pct(row['after_rate'])} "
            f"({row['after_n_failures']}/{row['after_n_episodes']}), "
            f"Уилсон 95% CI {format_ci(tuple(row['after_ci']))}. "
            f"Источник: {row['after_cite']}."
        )
        lines.append(
            f"   - статус: **{row['status']}**"
            + (f" (точный Макнемар до/после p={format_p(p_mc)})" if p_mc is not None else "")
            + "."
        )
        lines.append(
            f"   - порог {format_pct(threshold)}: "
            f"{format_crossings(finding.get('crossings') or [], threshold)}."
        )
        lines.append("")
    return lines

def build_markdown(payload: dict, episodes: list[dict], diag: dict | None = None) -> str:
    n_fail = len(episodes)
    phases = phase_stats(episodes)
    impact = impact_window_stats(payload, episodes)
    examples = pick_examples(episodes)
    if diag is None:
        diag = build_diagnosis_payload(payload)

    criterion = failure_criterion_label(payload)
    criterion_txt = f"`{criterion}`" if criterion else "поле `failure_criterion` отсутствует"

    lines = [
        "# Диагностика сбоев Go2",
        "",
        scenario_blurb(payload),
        "",
        "## Общая статистика",
        "",
        *header_stat_lines(payload),
        "",
        "## Категории сбоев",
        "",
        f"Критерий сбоя: {criterion_txt}. В данных это `fall` и `stance_loss` из `fail_reason`.",
        "",
    ]
    reason_lines = baseline_fail_reason_lines(payload)
    if reason_lines:
        lines.append(
            "Базовый уровень OFAT (берётся один фактор на сценарий, "
            "эпизоды разных параметров не суммируются):"
        )
        lines.extend(reason_lines)
        lines.append("")
    lines.extend(stored_fail_reason_lines(episodes))
    lines.append("")
    if n_fail == 0:
        lines += format_sweep_sections(payload)
        lines += format_recommendation_sections(diag)
        return "\n".join(lines)

    lines += [
        "",
        "## Сбои при/после удара (шаг ≥ 450)",
        "",
        f"- эпизодов, не упавших до {impact['impulse_start']}-го шага: {impact['n_reached']}",
        f"- сбоев при/после удара (`during_impact` + `after_impact`): {impact['n_at_or_after']}",
        (
            f"- доля сбоев среди дошедших до окна удара: "
            f"{impact['share_among_reached'] * 100:.1f}%"
        ),
        (
            f"- сбоев до включения Fy (`before_impact`): {impact['n_before']} "
            "— их нельзя относить к последствию боковой силы"
        ),
        "",
        "## Разбивка сбоев по фазам удара",
        "",
    ]
    for phase, label, n, pct in phases:
        extra = ""
        if phase == PHASE_BEFORE and n:
            extra = " — падение ещё до включения силы Fy"
        lines.append(f"- **{label}** (`{phase}`): {n} шт. ({pct:.1f}% от сбоев){extra}")

    lines += ["", "## Показательные примеры", ""]

    if not examples:
        lines.append("Примеров нет.")
    else:
        for i, ep in enumerate(examples, start=1):
            reason = ep.get("fail_reason") or "поле fail_reason отсутствует"
            lines += [
                f"### Пример {i}. Эпизод {ep['episode']} — {reason}",
                "",
                f"- трение: {ep['friction']:.3f}",
                f"- боковая сила: {format_push(ep)}",
                f"- сбой: шаг {ep['fail_step']} ({ep['fail_vs_impulse']}), "
                f"t={ep['fail_time']:.3f} с, z={ep['final_z']:.3f}",
                f"- макс. |qvel[3]| (крен): {ep['max_abs_roll_rate']:.3f} рад/с",
                f"- макс. скорость корпуса: {ep['max_body_speed']:.3f} м/с",
                f"- высота в окне: {ep['z_first']:.3f} → {ep['z_last']:.3f}",
                "",
            ]

    lines += format_sweep_sections(payload)
    lines += format_recommendation_sections(diag)
    return "\n".join(lines)


def ask_gemini(report: str, api_key: str) -> str:
    from google import genai

    prompt = (
        "Ниже готовый отчёт по падениям квадропода Go2. "
        "Сделай углублённый разбор на русском: подтверди или поправь категории, "
        "укажи, чего не хватает простому PD, и дай ещё 3 практичных шага для контроллера. "
        "Пиши коротко.\n\n"
        + report
    )
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini вернул пустой ответ")
    return text.strip()


def main() -> None:
    if not FAILURES_PATH.exists():
        raise FileNotFoundError(
            f"нет файла {FAILURES_PATH}, сначала запусти step3_batch_stress.py"
        )

    payload = json.loads(FAILURES_PATH.read_text(encoding="utf-8"))
    episodes = classify_episodes(payload)
    diag = build_diagnosis_payload(payload)
    report = build_markdown(payload, episodes, diag)

    print(report.rstrip())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    recs_out = {key: value for key, value in diag.items() if key != "verifications_raw"}
    RECS_JSON_PATH.write_text(
        json.dumps(recs_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    api_key = get_api_key()
    if api_key:
        try:
            extra = ask_gemini(report, api_key)
            report = report + "\n## Углублённый разбор Gemini\n\n" + extra + "\n"
            print("\n## Углублённый разбор Gemini\n")
            print(extra)
        except Exception as exc:
            print(f"\nGemini недоступен ({type(exc).__name__}: {exc}). Сохранён локальный отчёт.")

    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"\nотчёт сохранён: {REPORT_PATH}")
    print(f"рекомендации JSON: {RECS_JSON_PATH}")


if __name__ == "__main__":
    main()
