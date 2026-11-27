"""Plain-language summary of a finished run.

Rates use wilson_ci. Pairwise comparison stays in paired_compare.
"""

from __future__ import annotations

import math

from step3_batch_stress import wilson_ci

REASON_PRIORITY = ("fall", "stance_loss", "ee_tracking_error", "torque_saturation")
REASON_TEXT = {
    "fall": "The robot fell.",
    "stance_loss": "The robot lost its stance.",
    "ee_tracking_error": "The end effector missed the target.",
    "torque_saturation": "The motors stayed saturated.",
}


def _missing(value) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() in ("", "None", "nan"):
        return True
    return False


def _text(value) -> str | None:
    if _missing(value):
        return None
    return str(value).strip()


def _number(value) -> float | None:
    if _missing(value) or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _failed(row: dict) -> bool:
    if _missing(row.get("failed")):
        return False
    return bool(row.get("failed"))


def _reason(row: dict) -> str | None:
    return _text(row.get("fail_reason"))


def _lead_reason(counts: dict[str, int]) -> str | None:
    if not counts:
        return None

    def rank(reason: str) -> tuple[int, int, str]:
        priority = REASON_PRIORITY.index(reason) if reason in REASON_PRIORITY else len(REASON_PRIORITY)
        return (-counts[reason], priority, reason)

    return min(counts, key=rank)


def _example(rows: list[dict], reason: str) -> dict | None:
    matches = [row for row in rows if _failed(row) and _reason(row) == reason]
    if not matches:
        return None

    def rank(row: dict) -> tuple[float, int]:
        force = _number(row.get("lateral_force"))
        episode = _number(row.get("episode"))
        return (-abs(force or 0.0), int(episode or 0))

    chosen = min(matches, key=rank)
    return {
        "seed": None if _number(chosen.get("seed")) is None else int(_number(chosen.get("seed"))),
        "episode": None if _number(chosen.get("episode")) is None else int(_number(chosen.get("episode"))),
        "fail_reason": reason,
        "friction": _number(chosen.get("friction")),
        "lateral_force": _number(chosen.get("lateral_force")),
        "scenario": _text(chosen.get("scenario")),
        "param_id": _text(chosen.get("param_id")),
        "level_label": _text(chosen.get("level_label")),
        "pd": _text(chosen.get("pd")),
        "controller": _text(chosen.get("controller")),
    }


def client_summary(rows: list[dict]) -> dict:
    """Failure rate, Wilson interval, and the leading reason with one example."""
    total = len(rows)
    failures = [row for row in rows if _failed(row)]
    counts: dict[str, int] = {}
    for row in failures:
        reason = _reason(row) or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    lead = _lead_reason(counts)
    lo, hi = wilson_ci(len(failures), total) if total else (float("nan"), float("nan"))
    rate = (len(failures) / total) if total else float("nan")
    return {
        "n": total,
        "n_failures": len(failures),
        "rate": rate,
        "ci95": [lo, hi],
        "reasons": counts,
        "lead_reason": lead,
        "example": None if lead is None else _example(rows, lead),
    }


def _disturbance(example: dict) -> str:
    parts: list[str] = []
    if example.get("scenario"):
        parts.append(f"scenario {example['scenario']}")
    if example.get("friction") is not None:
        parts.append(f"friction {example['friction']:.3f}")
    if example.get("lateral_force") is not None:
        parts.append(f"lateral force {example['lateral_force']:.1f} N")
    param = example.get("param_id")
    if param and param not in ("quick_run", "main_batch", "all"):
        parts.append(f"parameter {param}")
    level = example.get("level_label")
    if level and level not in ("baseline", "all"):
        parts.append(f"level {level}")
    seed = example.get("seed")
    where = ", ".join(parts) if parts else "this run"
    seed_text = f", seed {seed}" if seed is not None else ""
    return f"It showed up at {where}{seed_text}."


def client_text(summary: dict) -> str:
    """One paragraph a client can read without the raw table."""
    total = int(summary.get("n") or 0)
    if total <= 0:
        return "This run has no episodes yet."
    lo, hi = summary.get("ci95") or (float("nan"), float("nan"))
    interval = f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"
    failures = int(summary.get("n_failures") or 0)
    if failures <= 0:
        return (
            f"No failures in {total} episodes. "
            f"The Wilson 95% interval for the failure rate is {interval}."
        )
    rate = 100.0 * failures / total
    reason = summary.get("lead_reason") or "unknown"
    lead = REASON_TEXT.get(reason, f"The leading failure is {reason}.")
    example = summary.get("example") or {}
    disturbance = _disturbance(example) if example else ""
    return (
        f"{failures} of {total} episodes failed ({rate:.1f}%, Wilson 95% {interval}). "
        f"{lead} {disturbance}"
    ).strip()
