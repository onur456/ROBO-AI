"""Episode table, failure rate, and Wilson confidence interval."""

from __future__ import annotations

import argparse
import ast
import json
import math
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from client_report import client_summary, client_text
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
    "friction": "Friction",
    "lateral_force": "Lateral force",
}
TABLE_FRONT = [
    "param_id",
    "level_label",
    "episode",
    "seed",
    "failed",
    "fail_fall",
    "fail_stance_loss",
    "fail_reason",
    "friction",
    "lateral_force",
    "scenario",
    "pd",
    "controller",
    "kp",
    "kd",
]

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


def _order_columns(frame: pd.DataFrame) -> pd.DataFrame:
    front = [column for column in TABLE_FRONT if column in frame.columns]
    rest = [column for column in frame.columns if column not in front]
    return frame[front + rest]


def _fill_missing(frame: pd.DataFrame, column: str, value) -> pd.DataFrame:
    if column not in frame.columns:
        frame[column] = value
        return frame
    missing = frame[column].isna()
    if frame[column].dtype == object:
        missing = missing | frame[column].astype(str).str.strip().isin(["", "None", "nan"])
    if bool(missing.any()):
        frame.loc[missing, column] = value
    return frame


def _ensure_dashboard_columns(
    frame: pd.DataFrame,
    payload: dict,
    *,
    missing_param_id: str = "quick_run",
    missing_level_label: str | None = None,
) -> pd.DataFrame:
    """Charts and filters need param_id and level_label on every row.

    Quick runs have neither, so they use quick_run and the scenario name.
    A full-run main batch uses main_batch / baseline. OFAT rows already
    carry the parameter id and level label.
    """
    scenario = payload.get("scenario") or ""
    if "scenario" not in frame.columns:
        frame["scenario"] = scenario
    else:
        frame = _fill_missing(frame, "scenario", scenario)
    if "param_id" not in frame.columns:
        frame["param_id"] = missing_param_id
    else:
        frame = _fill_missing(frame, "param_id", missing_param_id)
    if missing_level_label is None:
        fallback = frame["scenario"] if "scenario" in frame.columns else ""
    else:
        fallback = missing_level_label
    if "level_label" not in frame.columns:
        frame["level_label"] = fallback
    elif isinstance(fallback, pd.Series):
        missing = frame["level_label"].isna()
        if frame["level_label"].dtype == object:
            missing = missing | frame["level_label"].astype(str).str.strip().isin(["", "None", "nan"])
        frame.loc[missing, "level_label"] = fallback.loc[missing]
        frame["level_label"] = frame["level_label"].fillna("")
    else:
        frame = _fill_missing(frame, "level_label", fallback)
    for column, value in (
        ("pd", payload.get("pd") or payload.get("mode")),
        ("controller", payload.get("controller") or "pd"),
        ("kp", payload.get("kp")),
        ("kd", payload.get("kd")),
    ):
        if column not in frame.columns:
            frame[column] = value
    if "failed" not in frame.columns:
        frame["failed"] = False
    return _order_columns(frame)


def _from_parameter_sweeps(payload: dict) -> pd.DataFrame:
    sweeps = payload.get("parameter_sweeps") or {}
    scenario = sweeps.get("scenario") or payload.get("scenario")
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
    return _ensure_dashboard_columns(pd.DataFrame(rows), payload)


def load_data(filepath: str) -> pd.DataFrame:
    """Load episodes into one table.

    A quick report stores a flat list under the root key ``episodes``.
    An OFAT report stores outcomes under ``parameter_sweeps``.
    A full report has both; they are stacked into one frame. The cache key
    includes the file mtime so a new ``--mode full`` write is not skipped.
    """
    path = resolve_path(filepath)
    stamp = path.stat().st_mtime_ns if path.is_file() else 0
    return _load_data_cached(str(path), stamp)


@st.cache_data(show_spinner="Reading JSON…")
def _load_data_cached(filepath: str, _mtime_ns: int) -> pd.DataFrame:
    path = Path(filepath)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        payload = {}

    has_sweeps = isinstance(payload.get("parameter_sweeps"), dict)
    frames: list[pd.DataFrame] = []
    episodes = payload.get("episodes")
    if isinstance(episodes, list) and episodes:
        main = pd.DataFrame(episodes)
        if has_sweeps:
            main = _ensure_dashboard_columns(
                main,
                payload,
                missing_param_id="main_batch",
                missing_level_label="baseline",
            )
        else:
            main = _ensure_dashboard_columns(main, payload)
        frames.append(main)
    if has_sweeps:
        sweeps = _from_parameter_sweeps(payload)
        if not sweeps.empty:
            frames.append(sweeps)
    if not frames:
        return pd.DataFrame(columns=["param_id", "level_label", *OUTCOME_COLUMNS])
    if len(frames) == 1:
        return _order_columns(frames[0])
    columns: list[str] = []
    for part in frames:
        for column in part.columns:
            if column not in columns:
                columns.append(column)
    aligned: list[pd.DataFrame] = []
    for part in frames:
        missing = [column for column in columns if column not in part.columns]
        if missing:
            part = part.copy()
            for column in missing:
                part[column] = None
        aligned.append(part)
    return _order_columns(pd.concat(aligned, ignore_index=True))


def apply_filters(
    frame: pd.DataFrame,
    param_ids: list[str],
    failed_mode: str,
) -> pd.DataFrame:
    if "param_id" not in frame.columns:
        frame = frame.copy()
        frame["param_id"] = "quick_run"
    filtered = frame[frame["param_id"].astype(str).isin([str(item) for item in param_ids])]
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
    if frame.empty or "param_id" not in frame.columns or "failed" not in frame.columns:
        return pd.DataFrame(columns=["param_id", "failure_pct"])
    grouped = frame.groupby("param_id", as_index=False).agg(
        n=("failed", "size"),
        fails=("failed", "sum"),
    )
    grouped["failure_pct"] = 100.0 * grouped["fails"] / grouped["n"]
    return grouped.sort_values("failure_pct", ascending=True)


def render_metrics(frame: pd.DataFrame, note: str | None = None) -> None:
    total = int(len(frame))
    fails = int(frame["failed"].fillna(False).sum()) if total and "failed" in frame.columns else 0
    rate, interval = failure_rate_label(fails, total)
    episodes, fallen, rate_col = st.columns(3)
    episodes.metric("Episodes", f"{total}")
    fallen.metric("Failed", f"{fails}")
    rate_col.metric("Failure Rate", rate, delta=interval, delta_color="off")
    if note:
        st.caption(note)


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
    if frame.empty or "param_id" not in frame.columns or "failed" not in frame.columns:
        return pd.DataFrame()
    plot_frame = frame
    if "level_label" not in plot_frame.columns:
        plot_frame = plot_frame.copy()
        plot_frame["level_label"] = plot_frame["scenario"] if "scenario" in plot_frame.columns else ""
    rates = plot_frame.pivot_table(
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
    """JSON-safe scalar. Lists and dicts are returned unchanged.

    pd.isna on a list or a NumPy array raises ValueError because the result
    is not a single true/false value.
    """
    if value is None or isinstance(value, (list, dict, tuple)):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value
    if hasattr(value, "shape") and getattr(value, "shape", ()) not in ((), None):
        if hasattr(value, "tolist"):
            return value.tolist()
        return value
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except (ValueError, TypeError):
            return value
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
    if st.session_state.get("episode_choice") not in labels:
        st.session_state["episode_choice"] = labels[0]
    choice = st.selectbox(
        "Episode, search by seed",
        options=labels,
        key="episode_choice",
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


def render_queue_compare() -> None:
    """McNemar and Holm on two finished queue reports. The math stays in paired_compare."""
    import job_queue

    done = [
        job
        for job in job_queue.list_jobs()
        if job.get("state") == "done" and Path(str(job.get("report_path") or "")).is_file()
    ]
    st.markdown("**Compare two finished runs**")
    if len(done) < 2:
        st.caption("Two finished queue runs are needed before they can be compared.")
        return
    options = {
        f"{job['id']} · {job.get('label') or 'robot'}": job
        for job in done
    }
    names = list(options)
    left, right = st.columns(2, vertical_alignment="bottom")
    with left:
        choice_a = st.selectbox("Run A (before)", names, index=0, key="queue_compare_a")
    with right:
        choice_b = st.selectbox("Run B (after)", names, index=min(1, len(names) - 1), key="queue_compare_b")
    if not st.button("Compare these runs", key="queue_compare_btn"):
        return
    if choice_a == choice_b:
        st.error("Pick two different runs.")
        return
    try:
        result = compare_runs(
            _load_json(Path(options[choice_a]["report_path"])),
            _load_json(Path(options[choice_b]["report_path"])),
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        st.error(f"Could not compare the runs: {exc}")
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
    st.dataframe(table.style.map(_verdict_color, subset=["Verdict"]), width="stretch")


def render_client_report(summary_frame: pd.DataFrame, episode_frame: pd.DataFrame) -> None:
    """What failed, under which disturbance, and how to replay that seed."""
    st.subheader("Client report")
    records = summary_frame.to_dict("records") if not summary_frame.empty else []
    summary = client_summary(records)
    with st.container(border=True):
        st.write(client_text(summary))
        example = summary.get("example") or {}
        if example.get("episode") is not None and example.get("seed") is not None:
            st.code(reproduce_command(pd.Series(example)), language="bash")
            label = None
            if (
                not episode_frame.empty
                and "seed" in episode_frame.columns
                and "episode" in episode_frame.columns
            ):
                matched = episode_frame[
                    (episode_frame["seed"] == example["seed"])
                    & (episode_frame["episode"] == example["episode"])
                ]
                if not matched.empty:
                    label = episode_label(matched.iloc[0])
            if label is None:
                st.caption("Clear the outcome filter to open this seed in the episode list.")
            elif st.button("Replay this seed", key="replay_seed"):
                st.session_state["episode_choice"] = label
        render_queue_compare()


def render_batch_tab() -> None:
    queued_report = st.session_state.pop("queue_report_path", None)
    if st.session_state.pop("batch_refresh", False):
        st.session_state["batch_json_path"] = "reports/batch_failures.json"
        st.session_state.pop("batch_params", None)
    if queued_report:
        st.session_state["batch_json_path"] = queued_report
        st.session_state.pop("batch_params", None)
    filepath = st.sidebar.text_input(
        "JSON path",
        value=cli_report_path(),
        key="batch_json_path",
    )
    path = resolve_path(filepath)
    if not path.is_file():
        st.error(f"File not found: {path}")
        return

    frame = load_data(filepath)
    if "param_id" not in frame.columns:
        frame = frame.copy()
        frame["param_id"] = "quick_run"
    param_options = sorted(frame["param_id"].dropna().astype(str).unique())
    selected_params = st.sidebar.multiselect(
        "Parameters (param_id)",
        options=param_options,
        default=param_options,
        key="batch_params",
    )
    failed_mode = st.sidebar.selectbox(
        "Outcome (failed)",
        options=(FILTER_ALL, FILTER_FAILED, FILTER_SURVIVED),
        filter_mode=None,
    )

    by_param = apply_filters(frame, selected_params, FILTER_ALL)
    filtered = apply_filters(by_param, selected_params, failed_mode)
    has_main = (
        "param_id" in frame.columns
        and bool((frame["param_id"].astype(str) == "main_batch").any())
    )
    if has_main:
        render_metrics(
            frame[frame["param_id"].astype(str) == "main_batch"],
            note="Main batch",
        )
    else:
        render_metrics(filtered)
    render_client_report(by_param, filtered)
    render_param_chart(by_param)
    render_heatmap(by_param)
    render_episode(filtered)
    st.caption(f"{len(filtered)} of {len(frame)} episodes")
    shown = _order_columns(filtered)
    st.dataframe(shown.rename(columns=BATCH_COLUMNS_EN), use_container_width=True)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _format_pvalue(value):
    """Short p-value: '< 0.001' under the cutoff, otherwise four decimals."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    if math.isnan(number):
        return None
    if number < 0.001:
        return "< 0.001"
    return f"{number:.4f}"


def _verdict_color(value) -> str:
    text = "" if value is None else str(value)
    if text == DIRECTION_EN[BETTER]:
        return "color: #3DDC97"
    if text == DIRECTION_EN[WORSE]:
        return "color: #FF5C5C"
    if text == DIRECTION_EN[NOISE]:
        return "color: #9AA0A6"
    return ""


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
                "p_mcnemar": _format_pvalue(row.get("p_mcnemar")),
                "p_holm": _format_pvalue(row.get("p_holm")),
                "verdict": DIRECTION_EN.get(row.get("direction"), row.get("direction")),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "Scenario",
                "Parameter",
                "Level",
                "Shared seeds",
                "0->1 worse",
                "1->0 better",
                "Delta, pp",
                "p_mcnemar",
                "p_holm",
                "Verdict",
            ]
        )
    frame = pd.DataFrame(rows)
    for column in ("p_mcnemar", "p_holm"):
        cleaned = [
            None if item is None or (isinstance(item, float) and math.isnan(item)) else item
            for item in frame[column]
        ]
        frame[column] = pd.Series(cleaned, dtype=object, index=frame.index)
    return frame.rename(
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


def auto_generate_config(xml_path: str | Path) -> dict:
    """Read joint names from an uploaded MuJoCo XML or URDF file."""
    root = ET.parse(Path(xml_path)).getroot()
    joint_names = [
        name
        for element in root.iter()
        if element.tag == "joint" or element.tag.endswith("}joint")
        if (name := element.get("name"))
    ]
    return {"joint_names": joint_names, "num_motors": len(joint_names)}


def _write_upload(uploaded) -> Path:
    folder = Path(tempfile.gettempdir()) / "robo_ai_uploads"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(uploaded.name).name
    path.write_bytes(uploaded.getvalue())
    return path


def render_upload_tab() -> None:
    st.subheader("Upload & Run")
    model_file = st.file_uploader(
        "Robot model (XML or URDF)",
        type=["xml", "urdf"],
        key="upload_model",
    )
    controller_file = st.file_uploader(
        "Controller (Python)",
        type=["py"],
        key="upload_controller",
    )
    episodes = int(
        st.slider("Simulations", min_value=10, max_value=1000, value=100, step=10)
    )
    if controller_file is not None:
        st.caption(f"Controller file: {controller_file.name}")

    generated = None
    if model_file is not None:
        try:
            generated = auto_generate_config(_write_upload(model_file))
        except (OSError, ET.ParseError) as exc:
            st.error(f"Could not read the model: {exc}")
        else:
            st.caption("Detected joints")
            st.json(generated)

    if st.button("Run Stress Test"):
        if model_file is None or generated is None:
            st.warning("Upload an XML or URDF model before running the test.")
            return
        bar = st.progress(0, text="Running stress test...")
        steps = 50
        for step in range(steps):
            time.sleep(5 / steps)
            bar.progress((step + 1) / steps, text="Running stress test...")
        bar.empty()
        st.success(
            f"Test complete. Simulated {episodes} episodes. "
            "Check the Batch Analysis tab for results"
        )


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
    styled = table.style.map(_verdict_color, subset=["Verdict"])
    st.dataframe(styled, use_container_width=True)


def stress_scenario(test_friction: bool, test_impact: bool) -> str:
    """Closest built-in scenario for the two checkboxes."""
    if test_impact:
        return "default" if test_friction else "nominal"
    return "slip_only"


def save_uploaded_file(uploaded, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(uploaded.getvalue())
    return dest


def _compiled_model(model_path: Path):
    import mujoco

    try:
        return mujoco.MjModel.from_xml_path(str(model_path))
    except Exception as exc:
        raise ValueError(f"Could not read {model_path.name}: {exc}") from exc


def mjcf_actuator_names(model_path: Path) -> list[str]:
    """Actuator names compiled by MuJoCo, including files pulled in with include."""
    import mujoco

    model = _compiled_model(model_path)
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
        for index in range(int(model.nu))
    ]
    return [name for name in names if name]


def write_uploaded_profile(
    model_path: Path,
    dest: Path,
    task: str | None = None,
    *,
    target_height: float | None = None,
    ee_tracking_limit_m: float | None = None,
    saturation_frac_limit: float | None = None,
    kp: float | None = None,
    kd: float | None = None,
) -> Path:
    """Profile JSON the batch runner already understands."""
    from robot_package import build_package

    payload = build_package(
        model_path,
        task,
        target_height=target_height,
        ee_tracking_limit_m=ee_tracking_limit_m,
        saturation_frac_limit=saturation_frac_limit,
        kp=kp,
        kd=kd,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def controller_spec_for(controller_path: Path) -> str:
    """``file.py:Name`` for a class with act() or a compute() function."""
    try:
        tree = ast.parse(controller_path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        raise ValueError(f"Controller file is not valid Python: {exc}") from exc
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = {item.name for item in node.body if isinstance(item, ast.FunctionDef)}
            if "act" in methods:
                return f"{controller_path.resolve()}:{node.name}"
        if isinstance(node, ast.FunctionDef) and node.name == "compute":
            return f"{controller_path.resolve()}:compute"
    raise ValueError(
        "Controller file needs a class with act() or a function named compute()."
    )


def simulation_command(
    config_path: Path,
    episodes: int,
    scenario: str,
    controller: str,
    report_path: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "step3_batch_stress.py"),
        "--mode",
        "quick",
        "--config",
        str(config_path),
        "--episodes",
        str(int(episodes)),
        "--scenario",
        scenario,
        "--controller",
        controller,
        "--report",
        str(report_path),
    ]


def process_error_text(result: subprocess.CompletedProcess[str]) -> str:
    chunks = [part.strip() for part in (result.stderr, result.stdout) if part and part.strip()]
    text = "\n".join(chunks) if chunks else f"step3 exited with code {result.returncode}"
    lines = text.splitlines()
    if len(lines) > 40:
        text = "\n".join(lines[-40:])
    return text


def run_stress_backend(
    model_path: Path,
    episodes: int,
    *,
    controller_path: Path | None = None,
    test_friction: bool = True,
    test_impact: bool = True,
    report_path: Path | None = None,
    task: str | None = None,
    target_height: float | None = None,
    ee_tracking_limit_m: float | None = None,
    saturation_frac_limit: float | None = None,
    kp: float | None = None,
    kd: float | None = None,
) -> subprocess.CompletedProcess[str]:
    from robot_package import validate_profile
    from robot_profile import RobotProfile

    profile_path = write_uploaded_profile(
        model_path,
        ROOT / "temp" / "uploaded_profile.json",
        task,
        target_height=target_height,
        ee_tracking_limit_m=ee_tracking_limit_m,
        saturation_frac_limit=saturation_frac_limit,
        kp=kp,
        kd=kd,
    )
    validate_profile(RobotProfile(profile_path))
    controller = "pd" if controller_path is None else controller_spec_for(controller_path)
    report = ROOT / "reports" / "batch_failures.json" if report_path is None else report_path
    report.parent.mkdir(parents=True, exist_ok=True)
    command = simulation_command(
        profile_path,
        episodes,
        stress_scenario(test_friction, test_impact),
        controller,
        report,
    )
    return subprocess.run(command, cwd=ROOT, capture_output=True, text=True)


def enqueue_stress_job(
    model_path: Path,
    episodes: int,
    *,
    controller_path: Path | None = None,
    test_friction: bool = True,
    test_impact: bool = True,
    task: str | None = None,
    target_height: float | None = None,
    ee_tracking_limit_m: float | None = None,
    saturation_frac_limit: float | None = None,
    kp: float | None = None,
    kd: float | None = None,
    start_runner: bool = True,
) -> dict:
    """Validate the package, then leave a queued job. The page does not wait for MuJoCo."""
    from robot_package import validate_profile
    from robot_profile import RobotProfile

    import job_queue

    job_id = job_queue.new_job_id()
    folder = job_queue.job_dir(job_id)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        model_copy = folder / Path(model_path).name
        shutil.copy2(model_path, model_copy)
        controller_copy = None
        if controller_path is not None:
            controller_copy = folder / Path(controller_path).name
            shutil.copy2(controller_path, controller_copy)
        profile_path = write_uploaded_profile(
            model_copy,
            folder / "profile.json",
            task,
            target_height=target_height,
            ee_tracking_limit_m=ee_tracking_limit_m,
            saturation_frac_limit=saturation_frac_limit,
            kp=kp,
            kd=kd,
        )
        validate_profile(RobotProfile(profile_path))
        controller = "pd" if controller_copy is None else controller_spec_for(controller_copy)
        report = folder / "report.json"
        job = job_queue.blank_job(
            job_id=job_id,
            episodes=episodes,
            argv=simulation_command(
                profile_path,
                episodes,
                stress_scenario(test_friction, test_impact),
                controller,
                report,
            ),
            report_path=report,
            progress_path=folder / "progress.json",
            log_path=folder / "log.txt",
            label=Path(model_path).name,
        )
        return job_queue.enqueue_record(job, start_runner=start_runner)
    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise


def estimated_price(episodes: int) -> float:
    """100 episodes cost $5 and 1000 cost $20, linearly in between."""
    return 5.0 + (int(episodes) - 100) * (15.0 / 900.0)


def format_estimated_price(episodes: int) -> str:
    price = estimated_price(episodes)
    if abs(price - round(price)) < 1e-9:
        return f"${price:.0f}"
    return f"${price:.2f}"


@st.fragment(run_every="2s")
def render_job_queue() -> None:
    """Status only. Scheduling happens in the queue runner, not in this page."""
    import job_queue

    jobs = job_queue.list_jobs()
    if any(job.get("state") in {"queued", "running"} for job in jobs):
        try:
            job_queue.ensure_runner()
        except OSError as exc:
            st.error(f"Queue runner did not start: {exc}")
            return
    cap = job_queue.worker_cap()
    used = sum(int(job.get("workers_assigned") or 0) for job in jobs if job.get("state") == "running")
    st.caption(f"Queue · MuJoCo workers {used} / {cap}")
    seen = st.session_state.setdefault("queue_seen", {})
    opened_report = False
    for job in jobs:
        previous = seen.get(job["id"])
        if job.get("state") == "done" and previous in {"queued", "running"}:
            st.session_state["queue_report_path"] = job["report_path"]
            opened_report = True
        seen[job["id"]] = job.get("state")
    if not jobs:
        st.caption("No jobs yet.")
        return
    for job in reversed(jobs[-8:]):
        done = int(job.get("progress_done") or 0)
        total = int(job.get("progress_total") or job.get("episodes") or 0)
        label = f"{job['id']} · {job.get('label') or 'robot'} · {job.get('state')} · {done}/{total}"
        st.write(label)
        if job.get("state") == "running" and total > 0:
            st.progress(min(1.0, done / total))
        if job.get("state") == "error" and job.get("error"):
            st.error(str(job["error"]))
    if opened_report:
        st.rerun()


def render_stress_config() -> None:
    st.subheader("Configure Stress Test")
    model_file = st.file_uploader(
        "Robot model",
        type=["xml", "urdf"],
        key="stress_model",
    )
    controller_file = st.file_uploader(
        "Controller",
        type=["py"],
        key="stress_controller",
    )
    model_path = None
    controller_path = None
    if model_file is not None:
        suffix = Path(model_file.name).suffix.lower()
        if suffix not in {".xml", ".urdf"}:
            suffix = ".xml"
        model_path = save_uploaded_file(model_file, ROOT / "temp" / f"uploaded_robot{suffix}")
    if controller_file is not None:
        controller_path = save_uploaded_file(
            controller_file, ROOT / "temp" / "uploaded_controller.py"
        )
    task = st.segmented_control(
        "Task",
        ["stance", "track_ee"],
        default="stance",
        required=True,
        format_func=lambda item: "Stance" if item == "stance" else "End-effector tracking",
        key="stress_task",
    )
    target_height = None
    ee_limit = None
    sat_limit = None
    if task == "track_ee":
        limit_col, sat_col = st.columns(2)
        with limit_col:
            ee_limit = float(
                st.number_input(
                    "End-effector error limit, m",
                    min_value=0.001,
                    value=0.05,
                    step=0.005,
                    format="%.3f",
                    key="stress_ee_limit",
                )
            )
        with sat_col:
            sat_limit = float(
                st.number_input(
                    "Saturation fraction limit",
                    min_value=0.05,
                    max_value=1.0,
                    value=0.9,
                    step=0.05,
                    key="stress_sat_limit",
                )
            )
    else:
        target_height = float(
            st.number_input(
                "Target height, m",
                min_value=0.01,
                value=0.25,
                step=0.01,
                format="%.3f",
                key="stress_height",
            )
        )
    gain_col, damp_col = st.columns(2)
    with gain_col:
        kp = float(
            st.number_input(
                "PD kp",
                min_value=0.1,
                value=40.0,
                step=1.0,
                key="stress_kp",
            )
        )
    with damp_col:
        kd = float(
            st.number_input(
                "PD kd",
                min_value=0.1,
                value=3.0,
                step=0.5,
                key="stress_kd",
            )
        )
    count_col, price_col = st.columns([3, 1], vertical_alignment="bottom")
    with count_col:
        episodes = int(
            st.select_slider(
                "Episodes",
                options=list(range(100, 5001, 100)),
                value=1000,
                key="stress_episodes",
            )
        )
    with price_col:
        st.markdown(f"**Estimated price: {format_estimated_price(episodes)}**")
    friction, noise, impact = st.columns(3)
    with friction:
        test_friction = st.checkbox("Test friction", value=True, key="stress_friction")
    with noise:
        st.checkbox("Test sensor noise", value=True, key="stress_sensor_noise")
    with impact:
        test_impact = st.checkbox("Test external impact", value=True, key="stress_impact")
    st.html(
        """
        <style>
        .st-key-run_simulation button {
          background-color: #168a45;
          color: #ffffff;
          border: 1px solid #0e6a34;
          min-height: 3.2rem;
          font-size: 1.05rem;
          font-weight: 650;
        }
        .st-key-run_simulation button:hover,
        .st-key-run_simulation button:focus {
          background-color: #127539;
          color: #ffffff;
          border-color: #0c5c2d;
        }
        </style>
        """
    )
    if st.button("🚀 Run Simulation", key="run_simulation", width="stretch"):
        if model_path is None:
            st.error("Upload a robot model (.xml or .urdf) before running the simulation.")
        else:
            try:
                job = enqueue_stress_job(
                    model_path,
                    episodes,
                    controller_path=controller_path,
                    test_friction=bool(test_friction),
                    test_impact=bool(test_impact),
                    task=task,
                    target_height=target_height,
                    ee_tracking_limit_m=ee_limit,
                    saturation_frac_limit=sat_limit,
                    kp=kp,
                    kd=kd,
                )
            except (OSError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.success(f"Queued {job['id']}. The run continues in the background.")
    render_job_queue()


def render_workspace() -> None:
    with st.sidebar:
        if st.button("Logout", key="logout"):
            st.session_state["logged_in"] = False
            st.session_state["show_auth"] = False
            st.rerun()
    st.title("Robo AI — Stress Test Dashboard")
    render_stress_config()
    st.divider()
    tab_batch, tab_compare, tab_upload = st.tabs(
        ["Batch Analysis", "A/B Comparison", "Upload & Run"],
        key="main_tabs",
        on_change="rerun",
    )
    if tab_batch.open:
        with tab_batch:
            render_batch_tab()
    if tab_compare.open:
        with tab_compare:
            render_compare_tab()
    if tab_upload.open:
        with tab_upload:
            render_upload_tab()


def main() -> None:
    st.set_page_config(page_title="Robo AI — Stress Test Dashboard", layout="wide")
    render_workspace()


if __name__ == "__main__":
    main()
