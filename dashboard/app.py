"""Episode table, failure rate, and Wilson confidence interval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paired_compare import BETTER, NO_PAIR, NOISE, WORSE, compare_runs
from step3_batch_stress import wilson_ci

DEFAULT_PATH = "reports/batch_failures.json"
FILTER_ALL = "All"
FILTER_FAILED = "Failed only"
FILTER_SURVIVED = "Survived only"
DIRECTION_EN = {
    WORSE: "Significant increase in failures",
    BETTER: "Significant decrease in failures",
    NOISE: "Within noise",
    NO_PAIR: "No shared seeds",
}
BATCH_COLUMNS_EN = {
    "param_id": "Parameter",
    "level_label": "Level",
    "episode": "Episode",
    "seed": "Seed",
    "failed": "Failed",
    "fail_fall": "Fall",
    "fail_stance_loss": "Stance loss",
    "fail_reason": "Failure reason",
    "scenario": "Scenario",
    "pd": "PD mode",
    "controller": "Controller",
    "kp": "kp",
    "kd": "kd",
}

OUTCOME_COLUMNS = [
    "episode",
    "seed",
    "failed",
    "fail_fall",
    "fail_stance_loss",
    "fail_reason",
]


def cli_report_path() -> str:
    """Path from `streamlit run dashboard/app.py -- --report ...`."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--report", default=DEFAULT_PATH)
    args, _unknown = parser.parse_known_args()
    return str(args.report)


def resolve_path(filepath: str) -> Path:
    path = Path(filepath)
    if not path.is_absolute():
        path = ROOT / path
    return path


@st.cache_data(show_spinner="Reading JSON…")
def load_data(filepath: str) -> pd.DataFrame:
    """Flatten outcomes from parameter_sweeps into one table.

    Each row gets param_id (parameters[].id)
    and level_label (levels[].label).
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
    if failed_mode == FILTER_FAILED:
        filtered = filtered[filtered["failed"] == True]  # noqa: E712
    elif failed_mode == FILTER_SURVIVED:
        filtered = filtered[filtered["failed"] == False]  # noqa: E712
    return filtered


def failure_rate_label(fails: int, total: int) -> tuple[str, str]:
    if total <= 0:
        return "—", "n/a"
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
    episodes.metric("Episodes", f"{total}")
    fallen.metric("Failed", f"{fails}")
    rate_col.metric("Failure Rate", rate, delta=interval, delta_color="off")


def render_param_chart(frame: pd.DataFrame) -> None:
    summary = failure_by_param(frame)
    if summary.empty:
        st.info("No rows to plot.")
        return
    figure = px.bar(
        summary,
        x="failure_pct",
        y="param_id",
        orientation="h",
        labels={"failure_pct": "Failure Rate, %", "param_id": "Parameter"},
    )
    figure.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Failure Rate, %",
        yaxis_title="Parameter",
        yaxis={"categoryorder": "total ascending"},
    )
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
    st.subheader("Failure heatmap by parameter level")
    table = failure_heatmap(frame)
    if table.empty:
        st.info("No rows for the heatmap.")
        return
    figure = px.imshow(
        table,
        color_continuous_scale="RdYlGn_r",
        zmin=0,
        zmax=100,
        aspect="auto",
        labels={"color": "Failure Rate, %", "x": "Level", "y": "Parameter"},
    )
    figure.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Level",
        yaxis_title="Parameter",
    )
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
    """Command built from flags that step3 argparse actually accepts.

    There is no --seed flag: the episode seed is 42 * 1000003 + episode number,
    and the episode number is selected with --episodes.
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
    st.subheader("Reproduce episode")
    if frame.empty:
        st.info("No episodes match the current filter.")
        return

    labels = [episode_label(row) for _, row in frame.iterrows()]
    choice = st.selectbox(
        "Episode, search by seed",
        options=labels,
        index=0,
        help="Type a seed, parameter, or level to filter the list.",
    )
    row = frame.iloc[labels.index(choice)]
    record = episode_record(row)
    st.json(record)
    command = reproduce_command(row)
    st.code(command, language="bash")
    st.caption(
        "step3 has no --seed flag. This seed comes from the episode number. "
        "--episodes runs episodes 1..N and includes the selected number. "
        "ofat on nominal also runs moderate. "
        "--report writes a separate file and does not touch batch_failures.json."
    )
    file_name = f"episode_{int(row['seed'])}_{row['param_id']}.json"
    st.download_button(
        "Download episode JSON",
        data=json.dumps(record, ensure_ascii=False, indent=2),
        file_name=file_name,
        mime="application/json",
    )


def render_batch_tab() -> None:
    filepath = st.sidebar.text_input("JSON path", value=cli_report_path())
    path = resolve_path(filepath)
    if not path.is_file():
        st.error(f"File not found: {path}")
        return

    frame = load_data(filepath)
    param_options = sorted(frame["param_id"].dropna().astype(str).unique())
    selected_params = st.sidebar.multiselect(
        "Parameters (param_id)",
        options=param_options,
        default=param_options,
    )
    failed_mode = st.sidebar.selectbox(
        "Outcome (failed)",
        options=(FILTER_ALL, FILTER_FAILED, FILTER_SURVIVED),
    )

    filtered = apply_filters(frame, selected_params, failed_mode)
    render_metrics(filtered)
    render_param_chart(filtered)
    render_heatmap(filtered)
    render_episode(filtered)
    st.caption(f"{len(filtered)} of {len(frame)} episodes")
    st.dataframe(filtered.rename(columns=BATCH_COLUMNS_EN), use_container_width=True)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_table(result: dict) -> pd.DataFrame:
    rows = []
    for row in result.get("rows") or []:
        delta = row.get("delta_pp")
        delta_txt = None if delta is None or delta != delta else round(float(delta), 1)
        rows.append(
            {
                "scenario": row.get("scenario"),
                "parameter": row.get("param"),
                "level": row.get("label") or row.get("level"),
                "shared_seeds": row.get("n_shared"),
                "worse_0_to_1": row.get("n_0_to_1"),
                "better_1_to_0": row.get("n_1_to_0"),
                "delta_pp": delta_txt,
                "p_mcnemar": row.get("p_mcnemar"),
                "p_holm": row.get("p_holm"),
                "verdict": DIRECTION_EN.get(row.get("direction"), row.get("direction")),
            }
        )
    return pd.DataFrame(rows).rename(
        columns={
            "scenario": "Scenario",
            "parameter": "Parameter",
            "level": "Level",
            "shared_seeds": "Shared seeds",
            "worse_0_to_1": "0->1 worse",
            "better_1_to_0": "1->0 better",
            "delta_pp": "Delta, pp",
            "p_mcnemar": "p_mcnemar",
            "p_holm": "p_holm",
            "verdict": "Verdict",
        }
    )


def display_warning(note: str) -> str:
    """Paired-compare warnings are already English."""
    return note


def render_compare_tab() -> None:
    st.subheader("A/B comparison")
    path_a = st.text_input("File A (before)", value="", key="compare_a")
    path_b = st.text_input("File B (after)", value="", key="compare_b")
    if not path_a.strip() or not path_b.strip():
        st.info("Enter both JSON paths: file A (before) and file B (after).")
        return

    file_a = resolve_path(path_a.strip())
    file_b = resolve_path(path_b.strip())
    missing = [str(path) for path in (file_a, file_b) if not path.is_file()]
    if missing:
        st.error("File not found: " + "; ".join(missing))
        return
    try:
        result = compare_runs(_load_json(file_a), _load_json(file_b))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        st.error(f"Could not compare the files: {exc}")
        return

    shown: list[str] = []
    for note in result.get("warnings") or []:
        if note in shown:
            continue
        shown.append(note)
        st.warning(display_warning(note))

    table = comparison_table(result)
    if table.empty:
        st.info("No comparable levels.")
        return
    st.dataframe(table, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="Robo AI — Stress Test Dashboard", layout="wide")
    st.title("Robo AI — Stress Test Dashboard")
    tab_batch, tab_compare = st.tabs(["Batch Analysis", "A/B Comparison"])
    with tab_batch:
        render_batch_tab()
    with tab_compare:
        render_compare_tab()


if __name__ == "__main__":
    main()
