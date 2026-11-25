"""Таблица эпизодов, доля сбоев и доверительный интервал Уилсона."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paired_compare import compare_runs
from step3_batch_stress import wilson_ci

DEFAULT_PATH = "reports/batch_failures.json"

OUTCOME_COLUMNS = [
    "episode",
    "seed",
    "failed",
    "fail_fall",
    "fail_stance_loss",
    "fail_reason",
]


def resolve_path(filepath: str) -> Path:
    path = Path(filepath)
    if not path.is_absolute():
        path = ROOT / path
    return path


@st.cache_data(show_spinner="Читаю JSON…")
def load_data(filepath: str) -> pd.DataFrame:
    """Собрать outcomes из parameter_sweeps в одну таблицу.

    К каждой строке добавляются param_id (parameters[].id)
    и level_label (levels[].label).
    """
    path = resolve_path(filepath)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    sweeps = payload.get("parameter_sweeps") or {}
    scenario = sweeps.get("scenario")
    pd_name = payload.get("pd") or payload.get("mode")
    controller = payload.get("controller") or "pd"
    kp = payload.get("kp")
    kd = payload.get("kd")
    rows: list[dict] = []
    for param in sweeps.get("parameters") or []:
        param_id = param.get("id")
        for level in param.get("levels") or []:
            level_label = level.get("label")
            for outcome in level.get("outcomes") or []:
                row = dict(outcome)
                row["param_id"] = param_id
                row["level_label"] = level_label
                row["scenario"] = scenario
                row["pd"] = pd_name
                row["controller"] = controller
                row["kp"] = kp
                row["kd"] = kd
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["param_id", "level_label", *OUTCOME_COLUMNS])

    frame = pd.DataFrame(rows)
    front = ["param_id", "level_label"]
    rest = [column for column in frame.columns if column not in front]
    return frame[front + rest]


def apply_filters(
    frame: pd.DataFrame,
    param_ids: list[str],
    failed_mode: str,
) -> pd.DataFrame:
    filtered = frame[frame["param_id"].isin(param_ids)]
    if failed_mode == "Только упавшие":
        filtered = filtered[filtered["failed"] == True]  # noqa: E712
    elif failed_mode == "Только выжившие":
        filtered = filtered[filtered["failed"] == False]  # noqa: E712
    return filtered


def failure_rate_label(fails: int, total: int) -> tuple[str, str]:
    if total <= 0:
        return "—", "нет"
    lo, hi = wilson_ci(fails, total)
    rate = 100.0 * fails / total
    return f"{rate:.1f}%", f"[{lo * 100:.1f}%, {hi * 100:.1f}%]"


def failure_by_param(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["param_id", "failure_pct"])
    grouped = frame.groupby("param_id", as_index=False).agg(
        n=("failed", "size"),
        fails=("failed", "sum"),
    )
    grouped["failure_pct"] = 100.0 * grouped["fails"] / grouped["n"]
    return grouped.sort_values("failure_pct", ascending=True)


def render_metrics(frame: pd.DataFrame) -> None:
    total = int(len(frame))
    fails = int(frame["failed"].sum()) if total else 0
    rate, interval = failure_rate_label(fails, total)
    episodes, fallen, rate_col = st.columns(3)
    episodes.metric("Эпизоды", f"{total}")
    fallen.metric("Упавшие", f"{fails}")
    rate_col.metric("Failure Rate", rate, delta=interval, delta_color="off")


def render_param_chart(frame: pd.DataFrame) -> None:
    summary = failure_by_param(frame)
    if summary.empty:
        st.info("Нет строк для графика.")
        return
    figure = px.bar(
        summary,
        x="failure_pct",
        y="param_id",
        orientation="h",
        labels={"failure_pct": "Доля падений, %", "param_id": "Параметр"},
    )
    figure.update_layout(margin=dict(l=0, r=0, t=10, b=0), yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(figure, use_container_width=True)


def failure_heatmap(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rates = frame.pivot_table(
        index="param_id",
        columns="level_label",
        values="failed",
        aggfunc="mean",
    )
    return rates * 100.0


def render_heatmap(frame: pd.DataFrame) -> None:
    st.subheader("Тепловая карта падений по уровням параметров")
    table = failure_heatmap(frame)
    if table.empty:
        st.info("Нет строк для тепловой карты.")
        return
    figure = px.imshow(
        table,
        color_continuous_scale="RdYlGn_r",
        zmin=0,
        zmax=100,
        aspect="auto",
        labels={"color": "Доля падений, %", "x": "Уровень", "y": "Параметр"},
    )
    figure.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(figure, use_container_width=True)


def _json_value(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, str) and value == "":
        return value
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value


def episode_record(row: pd.Series) -> dict:
    return {str(key): _json_value(value) for key, value in row.to_dict().items()}


def episode_label(row: pd.Series) -> str:
    reason = "ok" if pd.isna(row.get("fail_reason")) else str(row["fail_reason"])
    state = "fail" if bool(row["failed"]) else "ok"
    return (
        f"{int(row['seed'])} · {row['param_id']} · {row['level_label']} · "
        f"ep {int(row['episode'])} · {state} · {reason}"
    )


def reproduce_command(row: pd.Series) -> str:
    """Команда из флагов, которые argparse в step3 реально принимает.

    Отдельного --seed нет: seed эпизода = 42 * 1000003 + номер эпизода,
    а номер попадает в прогон через --episodes.
    """
    scenario = row.get("scenario") or "nominal"
    pd_name = row.get("pd") or "baseline"
    controller = row.get("controller") or "pd"
    episode = int(row["episode"])
    seed = int(row["seed"])
    report = f"reports/repro_seed_{seed}_{row['param_id']}.json"
    return (
        "python step3_batch_stress.py "
        f"--mode ofat --scenario {scenario} --pd {pd_name} "
        f"--controller {controller} --episodes {episode} "
        f"--report {report}"
    )


def render_episode(frame: pd.DataFrame) -> None:
    st.subheader("Воспроизведение эпизода")
    if frame.empty:
        st.info("В текущем фильтре нет эпизодов.")
        return

    labels = [episode_label(row) for _, row in frame.iterrows()]
    choice = st.selectbox(
        "Эпизод, поиск по seed",
        options=labels,
        index=0,
        help="Список можно фильтровать вводом seed, параметра или уровня.",
    )
    row = frame.iloc[labels.index(choice)]
    record = episode_record(row)
    st.json(record)
    command = reproduce_command(row)
    st.code(command, language="bash")
    st.caption(
        "В step3 нет флага --seed. Этот seed получается из номера эпизода. "
        "--episodes задаёт прогон 1..N и включает выбранный номер. "
        "Режим ofat для nominal дополнительно считает moderate. "
        "--report пишет результат в отдельный файл и не трогает batch_failures.json."
    )
    file_name = f"episode_{int(row['seed'])}_{row['param_id']}.json"
    st.download_button(
        "Скачать JSON эпизода",
        data=json.dumps(record, ensure_ascii=False, indent=2),
        file_name=file_name,
        mime="application/json",
    )


def render_batch_tab() -> None:
    filepath = st.sidebar.text_input("Путь к JSON", value=DEFAULT_PATH)
    path = resolve_path(filepath)
    if not path.is_file():
        st.error(f"Файл не найден: {path}")
        return

    frame = load_data(filepath)
    param_options = sorted(frame["param_id"].dropna().astype(str).unique())
    selected_params = st.sidebar.multiselect(
        "Параметры (param_id)",
        options=param_options,
        default=param_options,
    )
    failed_mode = st.sidebar.selectbox(
        "Сбой (failed)",
        options=("Все", "Только упавшие", "Только выжившие"),
    )

    filtered = apply_filters(frame, selected_params, failed_mode)
    render_metrics(filtered)
    render_param_chart(filtered)
    render_heatmap(filtered)
    render_episode(filtered)
    st.caption(f"{len(filtered)} из {len(frame)} эпизодов")
    st.dataframe(filtered, use_container_width=True)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_table(result: dict) -> pd.DataFrame:
    rows = []
    for row in result.get("rows") or []:
        delta = row.get("delta_pp")
        delta_txt = None if delta is None or delta != delta else round(float(delta), 1)
        rows.append(
            {
                "сценарий": row.get("scenario"),
                "параметр": row.get("param"),
                "уровень": row.get("label") or row.get("level"),
                "общие seed": row.get("n_shared"),
                "0->1 хуже": row.get("n_0_to_1"),
                "1->0 лучше": row.get("n_1_to_0"),
                "Δ, п.п.": delta_txt,
                "p_mcnemar": row.get("p_mcnemar"),
                "p_holm": row.get("p_holm"),
                "вердикт": row.get("direction"),
            }
        )
    return pd.DataFrame(rows)


def render_compare_tab() -> None:
    st.subheader("Сравнение A/B")
    path_a = st.text_input("Файл А (До)", value="", key="compare_a")
    path_b = st.text_input("Файл Б (После)", value="", key="compare_b")
    if not path_a.strip() or not path_b.strip():
        st.info("Укажите пути к двум JSON: файл А (до) и файл Б (после).")
        return

    file_a = resolve_path(path_a.strip())
    file_b = resolve_path(path_b.strip())
    missing = [str(path) for path in (file_a, file_b) if not path.is_file()]
    if missing:
        st.error("Файл не найден: " + "; ".join(missing))
        return
    try:
        result = compare_runs(_load_json(file_a), _load_json(file_b))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        st.error(f"Не удалось сравнить файлы: {exc}")
        return

    shown: list[str] = []
    for note in result.get("warnings") or []:
        if note in shown:
            continue
        shown.append(note)
        st.warning(note)

    table = comparison_table(result)
    if table.empty:
        st.info("Сопоставимых уровней нет.")
        return
    st.dataframe(table, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Robo AI — Stress Test Dashboard", layout="wide")
    st.title("Robo AI — Stress Test Dashboard")
    tab_batch, tab_compare = st.tabs(["Анализ батча", "Сравнение A/B"])
    with tab_batch:
        render_batch_tab()
    with tab_compare:
        render_compare_tab()


if __name__ == "__main__":
    main()
