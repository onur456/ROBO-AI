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
    rows: list[dict] = []
    for param in sweeps.get("parameters") or []:
        param_id = param.get("id")
        for level in param.get("levels") or []:
            level_label = level.get("label")
            for outcome in level.get("outcomes") or []:
                row = dict(outcome)
                row["param_id"] = param_id
                row["level_label"] = level_label
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


def main() -> None:
    st.set_page_config(page_title="Robo AI — Stress Test Dashboard", layout="wide")
    st.title("Robo AI — Stress Test Dashboard")

    filepath = st.sidebar.text_input("Путь к JSON", value=DEFAULT_PATH)
    path = resolve_path(filepath)
    if not path.is_file():
        st.error(f"Файл не найден: {path}")
        st.stop()

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
    st.caption(f"{len(filtered)} из {len(frame)} эпизодов")
    st.dataframe(filtered, use_container_width=True)


if __name__ == "__main__":
    main()
