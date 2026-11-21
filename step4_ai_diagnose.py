import json
import os
from pathlib import Path

import numpy as np

FAILURES_PATH = Path("reports") / "batch_failures.json"
REPORT_PATH = Path("reports") / "diagnosis_summary.txt"
GEMINI_MODEL = "gemini-2.0-flash"


def _stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "max": None, "min": None, "count": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "count": int(arr.size),
    }


def _fmt_stats(stats: dict, ndigits: int = 4) -> str:
    if not stats["count"]:
        return "нет данных"
    return (
        f"среднее={stats['mean']:.{ndigits}f}, "
        f"макс={stats['max']:.{ndigits}f}, "
        f"мин={stats['min']:.{ndigits}f}"
    )


def _fmt_vec(values, ndigits: int = 3) -> str:
    arr = np.asarray(values, dtype=float)
    return "[" + ", ".join(f"{x:.{ndigits}f}" for x in arr) + "]"


def window_digest(window: list[dict]) -> dict:
    if not window:
        return {
            "n_samples": 0,
            "z_first": None,
            "z_last": None,
            "z_drop": None,
            "qvel_first": [],
            "qvel_last": [],
            "ctrl_first": [],
            "ctrl_last": [],
            "max_z_deviation": None,
            "max_qvel_norm": None,
            "max_abs_qvel": None,
            "max_abs_ctrl": None,
        }

    z = np.asarray([row["z"] for row in window], dtype=float)
    qvel = np.asarray([row["qvel_head"] for row in window], dtype=float)
    ctrl = np.asarray([row["ctrl"] for row in window], dtype=float)
    qvel_norm = np.linalg.norm(qvel, axis=1)

    return {
        "n_samples": len(window),
        "z_first": float(z[0]),
        "z_last": float(z[-1]),
        "z_drop": float(z[0] - z[-1]),
        "qvel_first": qvel[0].tolist(),
        "qvel_last": qvel[-1].tolist(),
        "ctrl_first": ctrl[0].tolist(),
        "ctrl_last": ctrl[-1].tolist(),
        "max_z_deviation": float(np.max(np.abs(z - z[0]))),
        "max_qvel_norm": float(np.max(qvel_norm)),
        "max_abs_qvel": float(np.max(np.abs(qvel))),
        "max_abs_ctrl": float(np.max(np.abs(ctrl))),
    }


def summarize_failures(payload: dict) -> dict:
    failures = payload.get("failures") or []
    friction = [float(ep["friction"]) for ep in failures]
    lateral = [float(ep["lateral_vel"]) for ep in failures]
    fail_step = [int(ep["fail_step"]) for ep in failures]
    fail_time = [float(ep["fail_time"]) for ep in failures]
    final_z = [float(ep["final_z"]) for ep in failures]
    episodes = []
    for ep in failures:
        digest = window_digest(ep.get("window") or [])
        episodes.append(
            {
                "episode": ep.get("episode"),
                "friction": float(ep["friction"]),
                "lateral_vel": float(ep["lateral_vel"]),
                "fail_step": int(ep["fail_step"]),
                "fail_time": float(ep["fail_time"]),
                "final_z": float(ep["final_z"]),
                "window": digest,
            }
        )
    return {
        "n_episodes": payload.get("n_episodes"),
        "n_failures": len(failures),
        "n_survived": payload.get("n_survived"),
        "failure_rate": payload.get("failure_rate"),
        "kp": payload.get("kp"),
        "kd": payload.get("kd"),
        "friction": _stats(friction),
        "lateral_vel": _stats(lateral),
        "fail_step": _stats(fail_step),
        "fail_time": _stats(fail_time),
        "final_z": _stats(final_z),
        "episodes": episodes,
    }


def build_report(summary: dict) -> str:
    lines = [
        "Диагностика сбоев Unitree Go2 (PD-стойка + стресс трения/бокового толчка).",
        "",
        "Общая сводка",
        f"- всего эпизодов в батче: {summary['n_episodes']}",
        f"- сбойных эпизодов: {summary['n_failures']}",
        f"- устояли: {summary['n_survived']}",
        f"- доля сбоев: {None if summary['failure_rate'] is None else round(summary['failure_rate'] * 100, 1)}%",
        f"- PD: kp={summary['kp']}, kd={summary['kd']}",
        f"- трение у сбоев: {_fmt_stats(summary['friction'])}",
        f"- боковая скорость у сбоев: {_fmt_stats(summary['lateral_vel'])}",
        f"- шаг сбоя: {_fmt_stats(summary['fail_step'], ndigits=1)}",
        f"- время сбоя, с: {_fmt_stats(summary['fail_time'])}",
        f"- финальный z: {_fmt_stats(summary['final_z'])}",
        "",
        "Выжимка телеметрии по каждому сбою (окно до падения)",
    ]

    if not summary["episodes"]:
        lines.append("- сбойных эпизодов нет.")
    else:
        for ep in summary["episodes"]:
            win = ep["window"]
            lines.extend(
                [
                    (
                        f"- эпизод {ep['episode']}: friction={ep['friction']:.4f}, "
                        f"lateral_vel={ep['lateral_vel']:.4f} м/с, "
                        f"сбой на шаге {ep['fail_step']} (t={ep['fail_time']:.4f} с), "
                        f"final_z={ep['final_z']:.4f}"
                    ),
                    (
                        f"  окно {win['n_samples']} отсчётов: "
                        f"z {win['z_first']:.4f} → {win['z_last']:.4f} "
                        f"(падение {win['z_drop']:.4f}, "
                        f"макс. отклонение z {win['max_z_deviation']:.4f})"
                    ),
                    (
                        f"  qvel: {_fmt_vec(win['qvel_first'])} → {_fmt_vec(win['qvel_last'])}; "
                        f"макс. |qvel|={win['max_abs_qvel']:.3f}, "
                        f"макс. ||qvel||={win['max_qvel_norm']:.3f}"
                    ),
                    (
                        f"  ctrl: {_fmt_vec(win['ctrl_first'])} → {_fmt_vec(win['ctrl_last'])}; "
                        f"макс. |ctrl|={win['max_abs_ctrl']:.3f}"
                    ),
                ]
            )

    lines.extend(
        [
            "",
            "Задача для анализа",
            "По этим метрикам кратко объясни вероятную причину падений Go2.",
            "Отдели случаи низкого трения от сильного бокового импульса, если паттерн виден.",
            "Дай 3–5 конкретных гипотез и что проверить дальше (контроллер, контакты, диапазон параметров).",
            "Пиши по-русски, коротко и по делу.",
        ]
    )
    return "\n".join(lines) + "\n"


def get_api_key() -> str | None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    return None


def ask_gemini(prompt: str, api_key: str) -> str:
    from google import genai

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = getattr(response, "text", None)
    if not text:
        raise RuntimeError("Gemini вернул пустой ответ")
    return text.strip()


def main() -> None:
    if not FAILURES_PATH.exists():
        raise FileNotFoundError(f"нет файла {FAILURES_PATH}, сначала запусти step3_batch_stress.py")

    payload = json.loads(FAILURES_PATH.read_text(encoding="utf-8"))
    summary = summarize_failures(payload)
    report = build_report(summary)

    print(report.rstrip())

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    api_key = get_api_key()
    if not api_key:
        REPORT_PATH.write_text(report, encoding="utf-8")
        print(f"\nAPI-ключ не задан, отчёт/промпт сохранён: {REPORT_PATH}")
        return

    try:
        diagnosis = ask_gemini(report, api_key)
    except Exception as exc:
        fallback = report + "\n\n# Gemini недоступен\n" + f"{type(exc).__name__}: {exc}\n"
        REPORT_PATH.write_text(fallback, encoding="utf-8")
        print(f"\nGemini недоступен ({exc}). Промпт сохранён: {REPORT_PATH}")
        return

    output = report + "\n\n# Ответ Gemini\n" + diagnosis + "\n"
    REPORT_PATH.write_text(output, encoding="utf-8")
    print("\n# Ответ Gemini")
    print(diagnosis)
    print(f"\nотчёт сохранён: {REPORT_PATH}")


if __name__ == "__main__":
    main()
