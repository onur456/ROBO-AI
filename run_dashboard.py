"""Launch the dashboard: python run_dashboard.py [--report PATH]."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "dashboard" / "app.py"
MISSING_STREAMLIT = (
    "Error: Streamlit is not installed. "
    "Please run: pip install streamlit pandas plotly"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the Robo AI Stress Test Dashboard")
    parser.add_argument(
        "--report",
        default="reports/batch_failures.json",
        help="Batch JSON. Prefills the dashboard sidebar",
    )
    return parser.parse_args()


def resolve_report(given: str) -> Path:
    path = Path(given)
    if not path.is_absolute():
        path = ROOT / path
    return path


def streamlit_command(report: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--",
        "--report",
        report,
    ]


def main() -> None:
    args = parse_args()
    report_path = resolve_report(args.report)
    if not report_path.is_file():
        print(
            f"Warning: file not found: {report_path}. "
            "The dashboard still starts; the path can be entered in the sidebar.",
            flush=True,
        )
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print(MISSING_STREAMLIT, flush=True)
        sys.exit(1)
    try:
        completed = subprocess.run(streamlit_command(args.report), cwd=ROOT, check=False)
    except FileNotFoundError:
        print(MISSING_STREAMLIT, flush=True)
        sys.exit(1)
    sys.exit(completed.returncode)


if __name__ == "__main__":
    main()
