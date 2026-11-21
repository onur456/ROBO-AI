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

CAT_OVERTURN = "Опрокидывание от бокового импульса"
CAT_SLIP = "Разъезжание лап на скользкой поверхности"
CAT_OTHER = "Прочее / смешанный сбой"


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
        category = classify_failure(float(ep["friction"]), metrics["max_abs_roll_rate"])
        classified.append(
            {
                "episode": ep.get("episode"),
                "friction": float(ep["friction"]),
                "lateral_vel": float(ep["lateral_vel"]),
                "fail_step": int(ep["fail_step"]),
                "fail_time": float(ep["fail_time"]),
                "final_z": float(ep["final_z"]),
                "category": category,
                **metrics,
            }
        )
    return classified


def category_stats(episodes: list[dict]) -> list[tuple[str, int, float]]:
    counts = Counter(ep["category"] for ep in episodes)
    total = len(episodes)
    rows = []
    for name in (CAT_OVERTURN, CAT_SLIP, CAT_OTHER):
        n = counts.get(name, 0)
        pct = (100.0 * n / total) if total else 0.0
        rows.append((name, n, pct))
    return rows


def explain_example(ep: dict) -> str:
    if ep["category"] == CAT_OVERTURN:
        return (
            f"Сильный боковой толчок ({ep['lateral_vel']:+.2f} м/с) раскрутил корпус: "
            f"макс. |qvel[3]| = {ep['max_abs_roll_rate']:.2f} рад/с (> {ROLL_RATE_THRESHOLD}). "
            "Суставной PD держит стойку ног, но не гасит крен, поэтому робот опрокидывается."
        )
    if ep["category"] == CAT_SLIP:
        return (
            f"Трение пола низкое ({ep['friction']:.3f} < {SLIP_FRICTION_THRESHOLD}), "
            f"крен умеренный (|qvel[3]| = {ep['max_abs_roll_rate']:.2f} рад/с). "
            "Опоры проскальзывают, ноги разъезжаются, высота корпуса падает без резкого опрокидывания."
        )
    return (
        f"Смешанный случай: friction={ep['friction']:.3f}, "
        f"толчок {ep['lateral_vel']:+.2f} м/с, "
        f"|qvel[3]|={ep['max_abs_roll_rate']:.2f} рад/с, "
        f"скорость корпуса {ep['max_body_speed']:.2f} м/с. "
        "Одновременно есть и скольжение, и потеря равновесия, одного признака недостаточно."
    )


def _sort_category(name: str, items: list[dict]) -> list[dict]:
    if name == CAT_OVERTURN:
        return sorted(
            items,
            key=lambda ep: (ep["max_abs_roll_rate"], abs(ep["lateral_vel"])),
            reverse=True,
        )
    if name == CAT_SLIP:
        return sorted(items, key=lambda ep: (ep["friction"], -ep["max_body_speed"]))
    return sorted(
        items,
        key=lambda ep: (abs(ep["lateral_vel"]), ep["max_body_speed"]),
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
    examples = pick_examples(episodes)
    recs = controller_recommendations(stats)

    lines = [
        "# Диагностика сбоев Go2",
        "",
        "Автоматический разбор `reports/batch_failures.json`: PD держит стартовую стойку, стресс — трение пола и боковой толчок.",
        "",
        "## Общая статистика",
        "",
        f"- всего прогонов: {n_all}",
        f"- сбоев: {n_fail}" + (f" ({rate * 100:.1f}%)" if isinstance(rate, (int, float)) else ""),
        f"- устояли: {n_survived}",
        f"- PD: kp={payload.get('kp')}, kd={payload.get('kd')}",
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

    for name, n, pct in stats:
        lines.append(f"- **{name}**: {n} шт. ({pct:.1f}% от сбоев)")
    lines += ["", "## Показательные примеры", ""]

    if not examples:
        lines.append("Примеров нет.")
    else:
        for i, ep in enumerate(examples, start=1):
            lines += [
                f"### Пример {i}. Эпизод {ep['episode']} — {ep['category']}",
                "",
                f"- трение: {ep['friction']:.3f}",
                f"- боковой толчок: {ep['lateral_vel']:+.3f} м/с",
                f"- сбой: шаг {ep['fail_step']}, t={ep['fail_time']:.3f} с, z={ep['final_z']:.3f}",
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
