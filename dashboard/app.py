"""Robo AI entry point: landing page, then the workspace."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.workspace import (
    _verdict_color,
    apply_filters,
    auto_generate_config,
    comparison_table,
    failure_heatmap,
    load_data,
    render_workspace,
)

__all__ = [
    "_verdict_color",
    "apply_filters",
    "auto_generate_config",
    "comparison_table",
    "failure_heatmap",
    "load_data",
    "render_workspace",
]

_LANDING = """
<style>
.robo-landing {
  background: #12160f;
  color: #f4f1e8;
  border: 1px solid #343b2c;
  border-radius: 20px;
  padding: 52px 56px 40px;
}
.robo-kicker {
  margin: 0 0 18px;
  color: #dff25a;
  font-size: 12px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
}
.robo-landing h1 {
  margin: 0 0 18px;
  max-width: 16ch;
  color: #f7f4ec;
  font-size: 52px;
  line-height: 0.98;
  font-weight: 560;
  letter-spacing: -0.035em;
}
.robo-lead {
  margin: 0 0 28px;
  max-width: 62ch;
  color: #d5d0c4;
  font-size: 18px;
  line-height: 1.55;
}
.robo-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}
.robo-chip {
  border: 1px solid #4a533c;
  border-radius: 999px;
  padding: 8px 14px;
  color: #f4f1e8;
  font-size: 13px;
}
</style>
<section class="robo-landing">
  <p class="robo-kicker">Reliability lab for robot controllers</p>
  <h1>Robo AI — Automated Stress Testing for Robotics</h1>
  <p class="robo-lead">
    Bring a URDF or MJCF model. We run the stress batch, score failures with
    Wilson confidence intervals, and show the bug in about 15 seconds.
  </p>
  <div class="robo-row">
    <span class="robo-chip">URDF and MJCF</span>
    <span class="robo-chip">Wilson intervals</span>
    <span class="robo-chip">Bugs in 15 seconds</span>
  </div>
</section>
"""


def render_landing() -> None:
    st.html(_LANDING)
    st.space("small")
    if st.button("Login / Sign Up", type="primary", key="open_auth"):
        st.session_state["show_auth"] = True
    if st.session_state.get("show_auth"):
        with st.container(border=True):
            st.subheader("Sign in")
            st.text_input("Email", key="auth_email")
            st.text_input("Password", type="password", key="auth_password")
            st.caption("No account database. Any email opens the workspace.")
            if st.button("Enter Workspace", type="primary", key="enter_workspace"):
                st.session_state["logged_in"] = True
                st.rerun()


def main() -> None:
    st.set_page_config(page_title="Robo AI", layout="wide")
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if not st.session_state.get("logged_in"):
        render_landing()
        return
    render_workspace()


if __name__ == "__main__":
    main()
