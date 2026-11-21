import json
import os
from collections import Counter, deque
from pathlib import Path

import numpy as np

FAILURES_PATH = Path("reports") / "batch_failures.json"
REPORT_PATH = Path("reports") / "final_diagnosis.md"
GEMINI_MODEL = "gemini-2.0-flash"

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


def fail_vs_impulse(fail_step: int, start: int, end: int) -> str:
    if fail_step < start:
        return f"до удара ({start - fail_step} шагов до начала {start}..{end})"
    if fail_step <= end:
        return f"во время удара (+{fail_step - start} шагов от начала {start}..{end})"
    return f"после удара (+{fail_step - end} шагов после конца {start}..{end})"


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
            "Преобладает низкое/среднее трение с просадкой корпуса "
            "(крен слабый, это ближе к разъезжанию, но μ >= 0.15, поэтому не отдельная категория)."
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
            "Опоры проскальзывают, ноги разъезжаются, высота корпуса падает без резкого опрокидывания."
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


def controller_recommendations(stats: list[tuple[str, int, float]]) -> list[str]:
    _ = stats
    return [
        "Добавить контур по ориентации корпуса (PD по крену/тангажу), а не только по углам суставов: текущий регулятор не видит ω корпуса.",
        "При большом |qvel[3]| расширять опору (hip abduction) и гасить крен, не увеличивая kp ног вслепую.",
        "На низком трении ограничивать горизонтальную силу стоп и снижать kp: жёсткая стойка на льду разгоняет проскальзывание.",
        "Оценивать скольжение по росту скорости корпуса без роста крена и переключать режим: мягче на скользком, жёстче при ударе; в коленях добавить компенсацию тяжести.",
    ]


def build_markdown(payload: dict, episodes: list[dict]) -> str:
    n_fail = len(episodes)
    n_all = payload.get("n_episodes")
    n_survived = payload.get("n_survived")
    rate = payload.get("failure_rate")
    stats = category_stats(episodes)
    phases = phase_stats(episodes)
    impact = impact_window_stats(payload, episodes)
    examples = pick_examples(episodes)
    recs = controller_recommendations(stats)

    lines = [
        "# Диагностика сбоев Go2",
        "",
        "Автоматический разбор `reports/batch_failures.json`: PD держит стартовую стойку, стресс — трение пола и боковая сила Fy на корпус в окне удара.",
        "",
        "## Общая статистика",
        "",
        f"- всего прогонов: {n_all}",
        f"- сбоев: {n_fail}" + (f" ({rate * 100:.1f}%)" if isinstance(rate, (int, float)) else ""),
        f"- устояли: {n_survived}",
        f"- PD: kp={payload.get('kp')}, kd={payload.get('kd')}",
        f"- окно удара: шаги {_as_int(payload.get('impulse_start'), DEFAULT_IMPULSE_START)}.."
        f"{_as_int(payload.get('impulse_end'), DEFAULT_IMPULSE_END)}",
        "",
        "## Категории сбоев",
        "",
        f"Правила: опрокидывание при макс. |qvel[3]| > {ROLL_RATE_THRESHOLD} рад/с; "
        f"разъезжание при трении < {SLIP_FRICTION_THRESHOLD} и |qvel[3]| ≤ {ROLL_RATE_THRESHOLD}; иначе смешанный случай.",
        "",
    ]
    if n_fail == 0:
        lines.append("Сбойных эпизодов нет.")
        return "\n".join(lines) + "\n"

    for name, n, n_before, pct in stats:
        before_note = (
            f", из них до удара (`before_impact`): {n_before}"
            " — самопроизвольное падение на старте/стойке, не следствие Fy"
            if n_before
            else ", из них до удара: 0"
        )
        lines.append(f"- **{name}**: {n} шт. ({pct:.1f}% от сбоев){before_note}")
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

    mixed = [ep for ep in episodes if ep["category"] == CAT_OTHER]
    if mixed:
        lines += [
            "",
            "## Пояснения к смешанным сбоям",
            "",
            "Правила классификации не менялись: это эпизоды вне порогов опрокидывания и разъезжания. "
            "Ниже — краткий комментарий, что в каждом случае выглядит преобладающим.",
            "",
        ]
        for ep in sorted(mixed, key=lambda item: _as_int(item.get("episode"))):
            lines.append(
                f"- эпизод {ep.get('episode')}: {format_push(ep)}, "
                f"μ={ep['friction']:.3f}, |qvel[3]|={ep['max_abs_roll_rate']:.2f} рад/с, "
                f"{ep.get('fail_vs_impulse', '')}. {mixed_comment(ep)}"
            )

    lines += ["", "## Показательные примеры", ""]

    if not examples:
        lines.append("Примеров нет.")
    else:
        for i, ep in enumerate(examples, start=1):
            lines += [
                f"### Пример {i}. Эпизод {ep['episode']} — {ep['category']}",
                "",
                f"- трение: {ep['friction']:.3f}",
                f"- боковая сила: {format_push(ep)}",
                f"- сбой: шаг {ep['fail_step']} ({ep['fail_vs_impulse']}), "
                f"t={ep['fail_time']:.3f} с, z={ep['final_z']:.3f}",
                f"- макс. |qvel[3]| (крен): {ep['max_abs_roll_rate']:.3f} рад/с",
                f"- макс. скорость корпуса: {ep['max_body_speed']:.3f} м/с",
                f"- высота в окне: {ep['z_first']:.3f} → {ep['z_last']:.3f}",
                "",
                explain_example(ep),
                "",
            ]

    lines += ["## Рекомендации для контроллера", ""]
    for rec in recs:
        lines.append(f"- {rec}")
    lines.append("")
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
    report = build_markdown(payload, episodes)

    print(report.rstrip())
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

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


if __name__ == "__main__":
    main()
