import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import mujoco

import controllers
from controllers import JointPDController, make_controller, prepare_controller_spec

MODEL_PATH = r"C:\Users\HP\Desktop\mujoco_menagerie-main\mujoco_menagerie-main\unitree_go2\scene.xml"
N_ENVS = 80
N_EPISODES = N_ENVS
N_SWEEP_EPISODES = 50
CHUNK_SIZE = 10
MAX_STEPS = 1500
WINDOW_SIZE = 200
SAG_Z = 0.20
CONTACT_Z = 0.12
TILT_FAIL_RAD = math.radians(45.0)
STANCE_LOSS_FRAC = 0.85
DEGRADATION_FRAC = 0.03
POST_IMPACT_MEASURE_S = 0.40
N_THRESHOLD_EPISODES = 200
FY_CRIT_N_EPISODES = 30
FY_CRIT_ITERS = 8
FY_CRIT_LO = 0.0
FY_CRIT_HI = 400.0
FY_CRIT_BOOTSTRAP = 500
IMPULSE_HEIGHT_JITTER_M = 0.03
IMPULSE_DURATION_JITTER = 0.20
UPRIGHT_HOLD_S = 0.20
SETTLE_TILT_RAD = math.radians(15.0)
SETTLE_SPEED = 0.15
SETTLE_Z_TOL = 0.02
SETTLE_HOLD_S = 0.10
JOINT_JITTER_RAD = 0.02
IMPULSE_JITTER_STEPS = 10
MASS_JITTER = 0.03
FALL_HEIGHT = SAG_Z
FRICTION_RANGE = (0.05, 1.0)
LATERAL_FORCE_RANGE = (-350.0, 350.0)
HEAVY_FORCE_RANGE = (-500.0, 500.0)
IMPULSE_START = 450
IMPULSE_END = 500
PROBE_FRICTIONS = (0.05, 1.0)
PROBE_LATERAL_FORCE = 180.0
NOMINAL_FRICTION_RANGE = (0.6, 1.0)
NOMINAL_FORCE_RANGE = (-100.0, 100.0)
MODERATE_FRICTION_RANGE = (0.4, 1.0)
MODERATE_FORCE_RANGE = (-250.0, 250.0)
GRID_MU = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
GRID_FY = (0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 350.0)
CLIFF_MU = (0.2, 0.8)
CLIFF_FY = tuple(float(v) for v in range(140, 221, 10))
N_GRID_EPISODES = 100
N_STANCE_EPISODES = 100
N_SAG_EPISODES = 100
N_TRACE_EPISODES = 3
GRID_PATH = Path("reports") / "baseline_grid.csv"
GRID_JSON_PATH = Path("reports") / "baseline_grid.json"
CLIFF_PATH = Path("reports") / "cliff_grid.csv"
CLIFF_SVG_PATH = Path("reports") / "cliff_failure_vs_fy.svg"
MU02_TRACES_PATH = Path("reports") / "mu02_traces.json"
MU02_Z_SVG_PATH = Path("reports") / "mu02_z.svg"
MU02_ROLL_SVG_PATH = Path("reports") / "mu02_roll.svg"
SAG_CSV_PATH = Path("reports") / "sag_compare.csv"
SAG_JSON_PATH = Path("reports") / "sag_compare.json"
THRESHOLDS_PATH = Path("reports") / "outcome_thresholds.json"
CRITERIA_SVG_PATH = Path("reports") / "criteria_sensitivity.svg"
ICE_TRACE_DIR = Path("reports")
FYCRIT_PATH = Path("reports") / "fy_crit_ofat.json"
TORQUE_TEST_PATH = Path("reports") / "torque_limit_test.json"
N_TORQUE_EPISODES = 100
TORQUE_TEST_SCALES = (1.0, 0.5, 0.25, 0.05)
LIMIT_INACTIVE_NOTE = "limit not reached / no effect at this level"
LIMIT_ACTIVE_NOTE = "saturation active"
SCENARIOS = {
    "default": {
        "id": "default",
        "label": "baseline: friction and lateral impact (Fy up to +/-350 N)",
        "friction_range": FRICTION_RANGE,
        "lateral_force_range": LATERAL_FORCE_RANGE,
        "probe_force": PROBE_LATERAL_FORCE,
    },
    "nominal": {
        "id": "nominal",
        "label": "nominal: mu 0.6-1.0, Fy up to +/-100 N",
        "friction_range": NOMINAL_FRICTION_RANGE,
        "lateral_force_range": NOMINAL_FORCE_RANGE,
        "probe_force": 80.0,
    },
    "moderate": {
        "id": "moderate",
        "label": "moderate: mu 0.4-1.0, Fy up to +/-250 N",
        "friction_range": MODERATE_FRICTION_RANGE,
        "lateral_force_range": MODERATE_FORCE_RANGE,
        "probe_force": 180.0,
    },
    "extreme": {
        "id": "extreme",
        "label": "extreme: mu 0.05-1.0, Fy up to +/-350 N (former default)",
        "friction_range": FRICTION_RANGE,
        "lateral_force_range": LATERAL_FORCE_RANGE,
        "probe_force": PROBE_LATERAL_FORCE,
    },
    "slip_only": {
        "id": "slip_only",
        "label": "slip and stance without a lateral impact (Fy=0)",
        "friction_range": FRICTION_RANGE,
        "lateral_force_range": (0.0, 0.0),
        "probe_force": 0.0,
    },
    "heavy_impact": {
        "id": "heavy_impact",
        "label": "stronger lateral impact (Fy up to +/-500 N)",
        "friction_range": FRICTION_RANGE,
        "lateral_force_range": HEAVY_FORCE_RANGE,
        "probe_force": 250.0,
    },
}
STANCE_CHECK_FRICTION = 0.8
STANCE_CHECK_FORCE = 0.0
SEED = 42
COM_PROBE_FRICTION = 0.8
COM_PROBE_FORCE = 180.0
COM_PROBE_OFFSETS = (
    ("zero", (0.0, 0.0, 0.0)),
    ("only_dY", (0.0, 0.04, 0.0)),
    ("only_dZ", (0.0, 0.0, 0.03)),
    ("dY+dZ", (0.0, 0.04, 0.03)),
)
TORQUE_PROBE_SCALES = (1.0, 0.5, 0.25, 0.05)
DELAY_PROBE_STEPS = (0, 5, 15)
DAMPING_PROBE_SCALES = (1.0, 0.0, 10.0)
INERTIA_PROBE_SCALES = (1.0, 0.5, 2.0)
CONTACT_PROBE_VARIANTS = (
    ("baseline", None, None),
    ("stiff", (0.002, 1.0), (0.9, 0.99, 0.0001, 0.5, 2.0)),
    ("soft", (0.08, 0.5), (0.0, 0.9, 0.05, 0.5, 2.0)),
)
NOISE_PROBE_VARIANTS = (
    ("none", 0.0, 0.0),
    ("medium", 0.03, 0.3),
    ("strong", 0.12, 1.2),
)
BIAS_PROBE_VARIANTS = (
    ("none", 0.0, 0.0),
    ("weak_neg", -0.015, 0.0),
    ("weak_pos", 0.015, 0.0),
    ("offset", 0.05, 0.0),
    ("drift", 0.0, 0.0002),
)
PARAMETER_SWEEPS = (
    {
        "id": "com_x",
        "name": "COM offset along X",
        "baseline": "zero",
        "levels": (
            {
                "id": "zero",
                "label": "baseline 0 m",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.0, 0.0]},
            },
            {
                "id": "neg",
                "label": "backward -0.03 m",
                "factor": {"kind": "com_offset", "offset": [-0.03, 0.0, 0.0]},
            },
            {
                "id": "pos",
                "label": "forward +0.03 m",
                "factor": {"kind": "com_offset", "offset": [0.03, 0.0, 0.0]},
            },
        ),
    },
    {
        "id": "com_y",
        "name": "COM offset along Y",
        "baseline": "zero",
        "levels": (
            {
                "id": "zero",
                "label": "baseline 0 m",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.0, 0.0]},
            },
            {
                "id": "weak_neg",
                "label": "mild -0.02 m",
                "factor": {"kind": "com_offset", "offset": [0.0, -0.02, 0.0]},
            },
            {
                "id": "weak_pos",
                "label": "mild +0.02 m",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.02, 0.0]},
            },
            {
                "id": "harsh_neg",
                "label": "harsh -0.04 m",
                "factor": {"kind": "com_offset", "offset": [0.0, -0.04, 0.0]},
            },
            {
                "id": "harsh_pos",
                "label": "harsh +0.04 m",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.04, 0.0]},
            },
        ),
    },
    {
        "id": "com_z",
        "name": "COM offset along Z",
        "baseline": "zero",
        "levels": (
            {
                "id": "zero",
                "label": "baseline 0 m",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.0, 0.0]},
            },
            {
                "id": "weak",
                "label": "mild +0.015 m",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.0, 0.015]},
            },
            {
                "id": "harsh",
                "label": "harsh +0.03 m",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.0, 0.03]},
            },
        ),
    },
    {
        "id": "torque_limit",
        "name": "motor torque limit",
        "baseline": "full",
        "levels": (
            {
                "id": "full",
                "label": "baseline 100% ctrlrange",
                "factor": {"kind": "torque_limit", "scale": 1.0},
            },
            {
                "id": "half",
                "label": "50% ctrlrange",
                "factor": {"kind": "torque_limit", "scale": 0.5},
            },
            {
                "id": "harsh",
                "label": "25% ctrlrange",
                "factor": {"kind": "torque_limit", "scale": 0.25},
            },
        ),
    },
    {
        "id": "action_delay",
        "name": "command delay",
        "baseline": "zero",
        "levels": (
            {
                "id": "zero",
                "label": "baseline 0 steps",
                "factor": {"kind": "delay", "steps": 0},
            },
            {
                "id": "weak",
                "label": "mild 5 steps",
                "factor": {"kind": "delay", "steps": 5},
            },
            {
                "id": "harsh",
                "label": "harsh 15 steps",
                "factor": {"kind": "delay", "steps": 15},
            },
        ),
    },
    {
        "id": "joint_damping",
        "name": "joint damping",
        "baseline": "base",
        "levels": (
            {
                "id": "base",
                "label": "baseline 1x",
                "factor": {"kind": "damping", "scale": 1.0},
            },
            {
                "id": "weak",
                "label": "mild 0x",
                "factor": {"kind": "damping", "scale": 0.0},
            },
            {
                "id": "harsh",
                "label": "harsh 10x",
                "factor": {"kind": "damping", "scale": 10.0},
            },
        ),
    },
    {
        "id": "floor_contact",
        "name": "floor contact softness",
        "baseline": "baseline",
        "levels": (
            {
                "id": "baseline",
                "label": "baseline solref",
                "factor": {"kind": "contact", "preset": "baseline"},
            },
            {
                "id": "stiff",
                "label": "stiff floor",
                "factor": {
                    "kind": "contact",
                    "preset": "stiff",
                    "solref": [0.002, 1.0],
                    "solimp": [0.9, 0.99, 0.0001, 0.5, 2.0],
                },
            },
            {
                "id": "soft",
                "label": "soft floor",
                "factor": {
                    "kind": "contact",
                    "preset": "soft",
                    "solref": [0.08, 0.5],
                    "solimp": [0.0, 0.9, 0.05, 0.5, 2.0],
                },
            },
        ),
    },
    {
        "id": "sensor_noise",
        "name": "sensor noise",
        "baseline": "none",
        "levels": (
            {
                "id": "none",
                "label": "no noise",
                "factor": {"kind": "noise", "q": 0.0, "v": 0.0},
            },
            {
                "id": "medium",
                "label": "medium noise",
                "factor": {"kind": "noise", "q": 0.03, "v": 0.3},
            },
            {
                "id": "strong",
                "label": "high noise",
                "factor": {"kind": "noise", "q": 0.12, "v": 1.2},
            },
        ),
    },
    {
        "id": "control_hold",
        "name": "command update rate",
        "baseline": "every",
        "levels": (
            {
                "id": "every",
                "label": "every step (500 Hz)",
                "factor": {"kind": "ctrl_hold", "steps": 1},
            },
            {
                "id": "weak",
                "label": "every 5 steps (100 Hz)",
                "factor": {"kind": "ctrl_hold", "steps": 5},
            },
            {
                "id": "harsh",
                "label": "every 20 steps (25 Hz)",
                "factor": {"kind": "ctrl_hold", "steps": 20},
            },
        ),
    },
    {
        "id": "sensor_bias",
        "name": "sensor bias",
        "baseline": "none",
        "levels": (
            {
                "id": "none",
                "label": "no bias",
                "factor": {"kind": "bias", "q": 0.0, "drift": 0.0},
            },
            {
                "id": "weak_neg",
                "label": "mild -0.015 rad",
                "factor": {"kind": "bias", "q": -0.015, "drift": 0.0},
            },
            {
                "id": "weak_pos",
                "label": "mild +0.015 rad",
                "factor": {"kind": "bias", "q": 0.015, "drift": 0.0},
            },
            {
                "id": "offset",
                "label": "bias +0.05 rad",
                "factor": {"kind": "bias", "q": 0.05, "drift": 0.0},
            },
            {
                "id": "drift",
                "label": "drift +0.0002 rad/step",
                "factor": {"kind": "bias", "q": 0.0, "drift": 0.0002},
            },
        ),
    },
)
PD_MODES = {
    "baseline": {"kp": 40.0, "kd": 3.0},
    "stiff": {"kp": 40.0, "kd": 3.0},
    "weak_pd": {"kp": 25.0, "kd": 1.5},
}
MAIN_MODE = "baseline"
ACTIVE_CONTROLLER = "pd"
STANCE_REF = {"z": None, "frac": STANCE_LOSS_FRAC, "kp": None, "kd": None}
STANCE_THRESH = {
    "z_threshold": None,
    "slip_threshold": None,
    "z_ref": None,
    "degradation_frac": DEGRADATION_FRAC,
    "n": 0,
    "derivation": None,
}
IMPULSE_CFG = {
    "realistic": False,
    "height_jitter_m": IMPULSE_HEIGHT_JITTER_M,
    "duration_jitter": IMPULSE_DURATION_JITTER,
}
KP = PD_MODES[MAIN_MODE]["kp"]
KD = PD_MODES[MAIN_MODE]["kd"]
REPORT_PATH = Path("reports") / "batch_failures.json"

GEOM_TYPE_NAMES = {
    int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    int(mujoco.mjtGeom.mjGEOM_HFIELD): "hfield",
    int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
    int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
    int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
    int(mujoco.mjtGeom.mjGEOM_BOX): "box",
    int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch stress for Go2 PD stance")
    parser.add_argument(
        "--mode",
        choices=("full", "quick", "ofat", "grid", "sag", "ff", "criteria", "fycrit", "pd_ff", "torque"),
        default="full",
        help=(
            "full: single probes, PD comparison, OFAT, and the main batch; "
            "quick: main batch only, written to reports/batch_failures.json; "
            "ofat: OFAT on nominal (control) and moderate (sensitivity); "
            "grid: stance, Fy/mu grid, cliff, and mu=0.2 traces (does not touch batch_failures.json); "
            "sag: stance-variant comparison; "
            "ff: gravity feed-forward diagnosis; "
            "criteria: 3-sigma thresholds, sensitivity, ice; "
            "fycrit: OFAT in Fy_crit units; "
            "pd_ff: PD plus contact FF, baseline unchanged; "
            "torque: torque_limit check at 100/50/25/5%%"
        ),
    )
    parser.add_argument(
        "--pd",
        choices=tuple(PD_MODES),
        default=MAIN_MODE,
        help="PD mode: baseline=kp40/kd3, weak_pd=kp25/kd1.5 (legacy), stiff=alias of baseline",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default="default",
        help=(
            "default/extreme: mu 0.05-1, Fy up to +/-350 N; "
            "nominal: mu 0.6-1, Fy up to +/-100 N; "
            "moderate: mu 0.4-1, Fy up to +/-250 N; "
            "slip_only: slip/stance at Fy=0; "
            "heavy_impact: stronger impact (Fy up to +/-500 N)"
        ),
    )
    parser.add_argument(
        "--stance-loss-frac",
        type=float,
        default=STANCE_LOSS_FRAC,
        help="stance_loss: settled z < frac * reference stance height (default 0.85)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=300,
        help="episodes per OFAT level and in the main batch (default 300)",
    )
    parser.add_argument(
        "--controller",
        default=ACTIVE_CONTROLLER,
        help=(
            "swappable controller: pd (kp/kd from --pd: baseline, stiff, weak_pd); "
            "stance_shift (same PD, target offset 0.2 rad); "
            "external path.py:ClassOrFunction or module:Attr. "
            "A function is called as compute(obs, step, dt, target_qpos). "
            "A class is constructed with kp and kd when it accepts them"
        ),
    )
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help=(
            "paired comparison of two JSON files on shared seeds: A before, B after. "
            "Computes Wilson, McNemar, and Holm, then exits without a simulation"
        ),
    )
    parser.add_argument(
        "--compare-out",
        default="reports/paired_compare.json",
        help="paired-comparison JSON. batch_failures.json and final_diagnosis.md are not used",
    )
    parser.add_argument(
        "--report",
        default=None,
        help=(
            "where to write the batch JSON. Without the flag, reports/batch_failures.json. "
            "A new path is not merged into the main report"
        ),
    )
    return parser.parse_args()


def geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""


def geom_type_name(model: mujoco.MjModel, geom_id: int) -> str:
    type_id = int(model.geom_type[geom_id])
    return GEOM_TYPE_NAMES.get(type_id, str(type_id))


def find_floor_geom_id(model: mujoco.MjModel) -> int:
    for geom_id in range(model.ngeom):
        if geom_name(model, geom_id).lower() in {"floor", "ground", "groundplane"}:
            return geom_id
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            return geom_id
    return 0


def find_torso_body_id(model: mujoco.MjModel) -> int:
    for name in ("torso", "base", "trunk"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            return int(body_id)
    return 1


def apply_torso_impulse(
    data: mujoco.MjData,
    torso_body_id: int,
    step: int,
    force: float,
    impulse_start: int = IMPULSE_START,
    impulse_end: int = IMPULSE_END,
    force_x: float | None = None,
    height_offset: float = 0.0,
) -> None:
    data.xfrc_applied[torso_body_id] = 0.0
    if not (impulse_start <= step <= impulse_end):
        return
    fy = float(force)
    fx = 0.0 if force_x is None else float(force_x)
    data.xfrc_applied[torso_body_id, 0] = fx
    data.xfrc_applied[torso_body_id, 1] = fy
    data.xfrc_applied[torso_body_id, 3] = float(height_offset) * fy
    data.xfrc_applied[torso_body_id, 4] = -float(height_offset) * fx


def impact_phase(
    fail_step: int,
    impulse_start: int = IMPULSE_START,
    impulse_end: int = IMPULSE_END,
) -> str:
    if fail_step < impulse_start:
        return "before_impact"
    if fail_step <= impulse_end:
        return "during_impact"
    return "after_impact"


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    half = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / den
    return (max(0.0, center - half), min(1.0, center + half))


def outcome_record(ep: dict) -> dict:
    return {
        "episode": int(ep["episode"]),
        "seed": int(ep.get("seed") if ep.get("seed") is not None else episode_seed(int(ep["episode"]))),
        "failed": bool(ep.get("failed")),
        "fail_fall": bool(ep.get("fail_fall") or ep.get("fail_tilt")),
        "fail_stance_loss": bool(ep.get("fail_stance_loss")),
        "fail_reason": ep.get("fail_reason"),
    }


def mcnemar_exact_paired(base_outcomes: list[dict], treat_outcomes: list[dict]) -> dict:
    """Exact McNemar on paired binary outcomes (shared episode/seed)."""
    from statsmodels.stats.contingency_tables import mcnemar

    base_by = {int(o["episode"]): bool(o["failed"]) for o in base_outcomes}
    treat_by = {int(o["episode"]): bool(o["failed"]) for o in treat_outcomes}
    keys = sorted(set(base_by) & set(treat_by))
    n00 = n01 = n10 = n11 = 0
    for key in keys:
        b = base_by[key]
        t = treat_by[key]
        if b and t:
            n11 += 1
        elif b and not t:
            n01 += 1
        elif (not b) and t:
            n10 += 1
        else:
            n00 += 1
    table = np.array([[n00, n01], [n10, n11]], dtype=int)
    n_disc = n01 + n10
    if n_disc == 0 or len(keys) == 0:
        pvalue = 1.0
    else:
        pvalue = float(mcnemar(table, exact=True).pvalue)
    return {
        "test": "mcnemar_exact",
        "n_paired": len(keys),
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
        "n_discordant": n_disc,
        "pvalue": pvalue,
    }


def holm_adjust(pvalues: list[float], alpha: float = 0.05) -> tuple[list[float], list[bool]]:
    from statsmodels.stats.multitest import multipletests

    if not pvalues:
        return [], []
    reject, p_adj, _, _ = multipletests(pvalues, alpha=alpha, method="holm")
    return [float(p) for p in p_adj], [bool(r) for r in reject]


HOLM_FAMILY_NOTE = (
    "Holm adjustment over every exact-McNemar p-value in this scenario: "
    "each non-baseline level of each factor against that factor's baseline "
    "(paired episodes that share a seed/index)"
)


def episode_seed(episode: int) -> int:
    return int(SEED) * 1_000_003 + int(episode)


def sample_episode_perturbation(episode: int, n_joints: int) -> dict:
    rng = np.random.default_rng(episode_seed(episode))
    shift = int(rng.integers(-IMPULSE_JITTER_STEPS, IMPULSE_JITTER_STEPS + 1))
    base_dur = IMPULSE_END - IMPULSE_START
    realistic = bool(IMPULSE_CFG.get("realistic"))
    if realistic:
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        dir_x = math.cos(ang)
        dir_y = math.sin(ang)
        height = float(
            rng.uniform(
                -float(IMPULSE_CFG["height_jitter_m"]),
                float(IMPULSE_CFG["height_jitter_m"]),
            )
        )
        dur_scale = 1.0 + float(
            rng.uniform(
                -float(IMPULSE_CFG["duration_jitter"]),
                float(IMPULSE_CFG["duration_jitter"]),
            )
        )
        duration = max(1, int(round(base_dur * dur_scale)))
        start = IMPULSE_START + shift
        end = start + duration
    else:
        dir_x = 0.0
        dir_y = 1.0
        height = 0.0
        start = IMPULSE_START + shift
        end = IMPULSE_END + shift
    return {
        "seed": episode_seed(episode),
        "joint_jitter": rng.uniform(-JOINT_JITTER_RAD, JOINT_JITTER_RAD, size=n_joints),
        "impulse_start": int(start),
        "impulse_end": int(end),
        "mass_scale": 1.0 + float(rng.uniform(-MASS_JITTER, MASS_JITTER)),
        "impulse_dir_x": dir_x,
        "impulse_dir_y": dir_y,
        "impulse_height": height,
        "impulse_realistic": realistic,
    }


def freejoint_roll_pitch(qpos: np.ndarray) -> tuple[float, float]:
    w, x, y, z = (float(qpos[3]), float(qpos[4]), float(qpos[5]), float(qpos[6]))
    sinp = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sinp)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    return roll, pitch


def tilt_from_vertical(qpos: np.ndarray) -> float:
    x = float(qpos[4])
    y = float(qpos[5])
    rzz = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    return math.acos(rzz)


def clip_hinge_qpos(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == free:
            continue
        if not bool(model.jnt_limited[joint_id]):
            continue
        adr = int(model.jnt_qposadr[joint_id])
        lo, hi = model.jnt_range[joint_id]
        data.qpos[adr] = min(max(float(data.qpos[adr]), float(lo)), float(hi))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"results saved: {path}")


def svg_escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def write_line_svg(
    path: Path,
    series: list[dict],
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    y_min: float | None = None,
    y_max: float | None = None,
    width: int = 900,
    height: int = 520,
) -> None:
    pad_l, pad_r, pad_t, pad_b = 70, 24, 40, 56
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b
    xs_all = [x for item in series for x in item["x"]]
    ys_all = [y for item in series for y in item["y"]]
    for item in series:
        ys_all.extend(item.get("y_lo") or [])
        ys_all.extend(item.get("y_hi") or [])
    x_min, x_max = min(xs_all), max(xs_all)
    if x_max <= x_min:
        x_max = x_min + 1.0
    auto_lo = min(ys_all) if ys_all else 0.0
    auto_hi = max(ys_all) if ys_all else 1.0
    lo = auto_lo if y_min is None else y_min
    hi = auto_hi if y_max is None else y_max
    if hi <= lo:
        hi = lo + 1.0
    span = hi - lo
    lo -= 0.05 * span
    hi += 0.05 * span

    def sx(x: float) -> float:
        return pad_l + (float(x) - x_min) / (x_max - x_min) * plot_w

    def sy(y: float) -> float:
        return pad_t + (hi - float(y)) / (hi - lo) * plot_h

    colors = ("#1f4e79", "#c0392b", "#1e8449", "#8e44ad", "#d35400", "#2c3e50")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="24" text-anchor="middle" font-size="16" '
        f'font-family="Segoe UI, Arial">{svg_escape(title)}</text>',
        f'<text x="{width / 2:.1f}" y="{height - 12}" text-anchor="middle" font-size="12" '
        f'font-family="Segoe UI, Arial">{svg_escape(xlabel)}</text>',
        f'<text x="16" y="{pad_t + plot_h / 2:.1f}" text-anchor="middle" font-size="12" '
        f'font-family="Segoe UI, Arial" transform="rotate(-90 16 {pad_t + plot_h / 2:.1f})">'
        f"{svg_escape(ylabel)}</text>",
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        'fill="#fafafa" stroke="#cccccc"/>',
    ]
    for tick in range(6):
        yv = lo + (hi - lo) * tick / 5
        y = sy(yv)
        parts.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            'stroke="#eeeeee"/>'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-size="11" '
            f'font-family="Segoe UI, Arial">{yv:.2f}</text>'
        )
    for i, item in enumerate(series):
        color = item.get("color") or colors[i % len(colors)]
        xs = item["x"]
        ys = item["y"]
        y_lo = item.get("y_lo")
        y_hi = item.get("y_hi")
        if y_lo is not None and y_hi is not None:
            for x, lo_v, hi_v in zip(xs, y_lo, y_hi):
                parts.append(
                    f'<line x1="{sx(x):.1f}" y1="{sy(lo_v):.1f}" x2="{sx(x):.1f}" '
                    f'y2="{sy(hi_v):.1f}" stroke="{color}" stroke-width="1.2"/>'
                )
        d = " ".join(
            f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}"
            for j, (x, y) in enumerate(zip(xs, ys))
        )
        parts.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="1.8"/>'
        )
        for x, y in zip(xs, ys):
            parts.append(
                f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.4" fill="{color}"/>'
            )
    legend_y = pad_t + 14
    for i, item in enumerate(series):
        color = item.get("color") or colors[i % len(colors)]
        lx = pad_l + 10
        ly = legend_y + i * 16
        parts.append(
            f'<rect x="{lx}" y="{ly - 8}" width="12" height="3" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{lx + 18}" y="{ly - 4}" font-size="11" '
            f'font-family="Segoe UI, Arial">{svg_escape(item.get("label", ""))}</text>'
        )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def find_foot_geom_ids(model: mujoco.MjModel) -> list[int]:
    ids: list[int] = []
    for name in ("FL", "FR", "RL", "RR"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            ids.append(int(geom_id))
    if ids:
        return ids
    return [
        geom_id
        for geom_id in range(model.ngeom)
        if int(model.geom_priority[geom_id]) > 0
    ]


def set_sliding_friction(model: mujoco.MjModel, geom_ids: list[int], friction: float) -> None:
    for geom_id in geom_ids:
        model.geom_friction[geom_id, 0] = friction
        actual = float(model.geom_friction[geom_id, 0])
        if abs(actual - friction) > 1e-12:
            raise RuntimeError(
                f"failed to write geom_friction[{geom_id}, 0]={friction}, current value {actual}"
            )


def print_floor_selection(
    model: mujoco.MjModel, floor_id: int, foot_ids: list[int]
) -> None:
    print("Floor geometry:")
    print(f"  index={floor_id}")
    print(f"  name={geom_name(model, floor_id)!r}")
    print(f"  type={geom_type_name(model, floor_id)}")
    print(f"  priority={int(model.geom_priority[floor_id])}")
    print(f"  condim={int(model.geom_condim[floor_id])}")
    print(f"  geom_friction.shape={tuple(model.geom_friction.shape)}")
    print(f"  geom_friction[{floor_id}]={model.geom_friction[floor_id].tolist()}")
    print("Foot contact geoms (higher priority; MuJoCo takes contact friction from them):")
    for geom_id in foot_ids:
        print(
            f"  id={geom_id} name={geom_name(model, geom_id)!r} "
            f"type={geom_type_name(model, geom_id)} "
            f"priority={int(model.geom_priority[geom_id])} "
            f"friction={model.geom_friction[geom_id].tolist()}"
        )


def reset_to_stance(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)


def read_body_ipos(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    return np.array(model.body_ipos[body_id], copy=True)


def write_body_ipos(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int, ipos: np.ndarray
) -> np.ndarray:
    model.body_ipos[body_id] = ipos
    written = read_body_ipos(model, body_id)
    if float(np.max(np.abs(written - np.asarray(ipos, dtype=float)))) > 1e-12:
        raise RuntimeError(
            f"failed to write body_ipos[{body_id}]={np.asarray(ipos).tolist()}, "
            f"current value {written.tolist()}"
        )
    mujoco.mj_setConst(model, data)
    return written


def read_ctrlrange(model: mujoco.MjModel) -> np.ndarray:
    return np.array(model.actuator_ctrlrange, copy=True)


def write_ctrlrange(model: mujoco.MjModel, ctrlrange: np.ndarray) -> np.ndarray:
    model.actuator_ctrlrange[:] = ctrlrange
    written = read_ctrlrange(model)
    if float(np.max(np.abs(written - np.asarray(ctrlrange, dtype=float)))) > 1e-12:
        raise RuntimeError("failed to write actuator_ctrlrange")
    return written


STOCK_CTRLRANGE: dict[str, np.ndarray | None] = {"value": None}


def stock_ctrlrange(model: mujoco.MjModel) -> np.ndarray:
    if STOCK_CTRLRANGE["value"] is None:
        STOCK_CTRLRANGE["value"] = read_ctrlrange(model)
    return np.array(STOCK_CTRLRANGE["value"], copy=True)


def apply_torque_limit(model: mujoco.MjModel, scale: float) -> np.ndarray:
    written = write_ctrlrange(model, stock_ctrlrange(model) * float(scale))
    model.actuator_ctrllimited[:] = True
    return written


def actuator_dof_gears(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    dofs = np.empty(model.nu, dtype=int)
    gears = np.empty(model.nu, dtype=float)
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        dofs[i] = int(model.jnt_dofadr[jid])
        gears[i] = float(model.actuator_gear[i, 0])
    return dofs, gears


def actuator_joint_rows(model: mujoco.MjModel) -> list[dict]:
    rows: list[dict] = []
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        dof = int(model.jnt_dofadr[jid])
        gear = float(model.actuator_gear[i, 0])
        rows.append(
            {
                "actuator": i,
                "actuator_name": mujoco.mj_id2name(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, i
                )
                or "",
                "joint": jid,
                "joint_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
                or "",
                "dof": dof,
                "gear": gear,
            }
        )
    return rows


def add_mapped_gravity_ff(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        dof = int(model.jnt_dofadr[jid])
        gear = float(model.actuator_gear[i, 0])
        if abs(gear) < 1e-12:
            continue
        data.ctrl[i] += float(data.qfrc_bias[dof]) / gear


def add_legacy_gravity_ff(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.ctrl[:] += np.asarray(data.qfrc_bias[6 : 6 + model.nu], dtype=float)


def joint_observation(
    data: mujoco.MjData,
    rng: np.random.Generator | None,
    noise_q: float,
    noise_v: float,
    bias_q: float,
) -> dict:
    """Joint observation. Noise and bias are the same for every controller."""
    q_obs = np.asarray(data.qpos[7:], dtype=float)
    v_obs = np.asarray(data.qvel[6:], dtype=float)
    if rng is not None and (noise_q > 0.0 or noise_v > 0.0):
        if noise_q > 0.0:
            q_obs = q_obs + rng.normal(0.0, noise_q, size=q_obs.shape)
        if noise_v > 0.0:
            v_obs = v_obs + rng.normal(0.0, noise_v, size=v_obs.shape)
    if abs(bias_q) > 0.0:
        q_obs = q_obs + bias_q
    return {"q": q_obs, "v": v_obs, "time": float(data.time)}


def command_from_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller,
    target_qpos: np.ndarray,
    step: int,
    dt: float,
    rng: np.random.Generator | None,
    noise_q: float,
    noise_v: float,
    bias_q: float,
    gravity_ff: bool = False,
    ff_ctrl: np.ndarray | None = None,
) -> np.ndarray:
    obs = joint_observation(data, rng, noise_q, noise_v, bias_q)
    label = getattr(controller, "name", None) or type(controller).__name__
    raw_cmd = controller.act(obs, step, dt, target_qpos)
    try:
        cmd = np.asarray(raw_cmd, dtype=float).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"controller {label!r} did not return a numeric command vector: {exc}"
        ) from exc
    n_act = int(model.nu)
    if cmd.size != n_act or not np.all(np.isfinite(cmd)):
        raise ValueError(
            f"controller {label!r} returned a command of length {int(cmd.size)} "
            f"(expected {n_act} finite numbers)"
        )
    data.ctrl[:] = cmd
    if gravity_ff:
        add_mapped_gravity_ff(model, data)
    if ff_ctrl is not None:
        data.ctrl[:] = np.asarray(data.ctrl, dtype=float) + np.asarray(ff_ctrl, dtype=float)
    return np.array(data.ctrl, copy=True)


def apply_pd(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    kp: float = KP,
    kd: float = KD,
    clip_ctrl: bool = False,
    rng: np.random.Generator | None = None,
    obs_noise_q: float = 0.0,
    obs_noise_v: float = 0.0,
    obs_bias_q: float = 0.0,
    gravity_ff: bool = False,
    ff_ctrl: np.ndarray | None = None,
) -> None:
    if model.nu != len(target_qpos):
        return
    command_from_controller(
        model,
        data,
        JointPDController(kp, kd, name="pd"),
        target_qpos,
        0,
        float(model.opt.timestep),
        rng,
        obs_noise_q,
        obs_noise_v,
        obs_bias_q,
        gravity_ff=gravity_ff,
        ff_ctrl=ff_ctrl,
    )
    if clip_ctrl:
        np.clip(
            data.ctrl,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
            out=data.ctrl,
        )


def episode_status(result: dict) -> str:
    if not result.get("failed") and result.get("fail_step") is None:
        return "survived"
    step = result.get("fail_step")
    kind = result.get("fall_kind")
    z_end = result.get("z_end")
    body_up = z_end is not None and float(z_end) > CONTACT_Z + 0.02
    if result.get("fail_fall") or kind in ("tilt", "contact_z", "upright_hold"):
        if kind == "contact_z":
            return f"fall (torso contact) at step {step}"
        if kind in ("tilt", "upright_hold") or body_up:
            return f"tip-over (tilt) at step {step}"
        return f"fall at step {step}"
    if result.get("fail_stance_loss"):
        return f"stance_loss at step {step}"
    if step is not None:
        return f"failure at step {step}"
    return "survived"


def saturation_note(sat: float | None) -> str:
    if sat is None or float(sat) <= 1e-12:
        return LIMIT_INACTIVE_NOTE
    return LIMIT_ACTIVE_NOTE


def format_sat_pct(sat: float | None) -> str:
    if sat is None or float(sat) <= 1e-12:
        return "0.0%"
    pct = float(sat) * 100.0
    if pct < 0.05:
        return f"{pct:.3f}%"
    return f"{pct:.1f}%"


def z_max_abs_delta(ref: dict, result: dict) -> float:
    z0 = np.asarray(ref["z"])
    z1 = np.asarray(result["z"])
    n = min(len(z0), len(z1))
    if n == 0:
        return 0.0
    return float(np.max(np.abs(z0[:n] - z1[:n])))


def print_probe_line(result: dict, extra: str = "", ref: dict | None = None) -> None:
    dz = ""
    if ref is not None and result is not ref:
        delta = z_max_abs_delta(ref, result)
        result["dz_vs_ref"] = delta
        dz = f"  max|Δz| vs {ref['label']}={delta:.6f}"
    prefix = f"{extra} " if extra else ""
    print(
        f"  {result['label']}: {prefix}{episode_status(result)}, "
        f"z_end={result['z_end']:.4f}, z_min={result['min_z']:.4f}{dz}"
    )


def print_z_deltas(results: list[dict], label_key: str, ref_idx: int = 0) -> bool:
    if len(results) < 2:
        return False
    z0 = np.asarray(results[ref_idx]["z"])
    any_diff = False
    ref_label = results[ref_idx][label_key]
    for result in results:
        if result is results[ref_idx]:
            continue
        z1 = np.asarray(result["z"])
        n = min(len(z0), len(z1))
        z_diff = float(np.max(np.abs(z0[:n] - z1[:n])))
        any_diff = any_diff or z_diff >= 1e-9
        print(
            f"  max |Δz| {ref_label} vs {result[label_key]}={z_diff:.6f} "
            f"(over the first {n} steps)"
        )
    return any_diff


def leg_dof_ids(model: mujoco.MjModel) -> np.ndarray:
    ids: list[int] = []
    free = int(mujoco.mjtJoint.mjJNT_FREE)
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) == free:
            continue
        ids.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(ids, dtype=int)


def read_dof_damping(model: mujoco.MjModel) -> np.ndarray:
    return np.array(model.dof_damping, copy=True)


def write_dof_damping(model: mujoco.MjModel, damping: np.ndarray) -> np.ndarray:
    model.dof_damping[:] = damping
    written = read_dof_damping(model)
    if float(np.max(np.abs(written - np.asarray(damping, dtype=float)))) > 1e-12:
        raise RuntimeError("failed to write dof_damping")
    return written


def read_body_inertia(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    return np.array(model.body_inertia[body_id], copy=True)


def write_body_inertia(
    model: mujoco.MjModel, data: mujoco.MjData, body_id: int, inertia: np.ndarray
) -> np.ndarray:
    model.body_inertia[body_id] = inertia
    written = read_body_inertia(model, body_id)
    if float(np.max(np.abs(written - np.asarray(inertia, dtype=float)))) > 1e-12:
        raise RuntimeError(
            f"failed to write body_inertia[{body_id}]={np.asarray(inertia).tolist()}, "
            f"current value {written.tolist()}"
        )
    mujoco.mj_setConst(model, data)
    return written


def read_geom_contact(
    model: mujoco.MjModel, geom_ids: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    solref = np.array([model.geom_solref[gid] for gid in geom_ids], copy=True)
    solimp = np.array([model.geom_solimp[gid] for gid in geom_ids], copy=True)
    return solref, solimp


def write_geom_contact(
    model: mujoco.MjModel,
    geom_ids: list[int],
    solref: np.ndarray,
    solimp: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    for i, geom_id in enumerate(geom_ids):
        model.geom_solref[geom_id] = solref[i]
        model.geom_solimp[geom_id] = solimp[i]
    written_ref, written_imp = read_geom_contact(model, geom_ids)
    if float(np.max(np.abs(written_ref - solref))) > 1e-12:
        raise RuntimeError("failed to write geom_solref")
    if float(np.max(np.abs(written_imp - solimp))) > 1e-12:
        raise RuntimeError("failed to write geom_solimp")
    return written_ref, written_imp


def apply_model_factor(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    friction_geoms: list[int],
    torso_body_id: int,
    factor: dict | None,
) -> None:
    if not factor:
        return
    kind = factor.get("kind")
    if kind == "com_offset":
        origin = read_body_ipos(model, torso_body_id)
        offset = np.asarray(factor.get("offset", (0.0, 0.0, 0.0)), dtype=float)
        write_body_ipos(model, data, torso_body_id, origin + offset)
        return
    if kind == "damping":
        origin = read_dof_damping(model)
        legs = leg_dof_ids(model)
        scaled = origin.copy()
        scaled[legs] = origin[legs] * float(factor.get("scale", 1.0))
        write_dof_damping(model, scaled)
        return
    if kind == "contact":
        preset = factor.get("preset")
        if preset in (None, "baseline"):
            return
        n_geoms = len(friction_geoms)
        solref = np.tile(np.asarray(factor["solref"], dtype=float), (n_geoms, 1))
        solimp = np.tile(np.asarray(factor["solimp"], dtype=float), (n_geoms, 1))
        write_geom_contact(model, friction_geoms, solref, solimp)
        return
    if kind == "torque_limit":
        apply_torque_limit(model, float(factor.get("scale", 1.0)))


def control_from_factor(factor: dict | None) -> dict:
    cfg = {
        "delay": 0,
        "noise_q": 0.0,
        "noise_v": 0.0,
        "clip_ctrl": False,
        "hold": 1,
        "bias_q": 0.0,
        "drift_q": 0.0,
    }
    if not factor:
        return cfg
    kind = factor.get("kind")
    if kind == "delay":
        cfg["delay"] = int(factor.get("steps", 0))
    elif kind == "noise":
        cfg["noise_q"] = float(factor.get("q", 0.0))
        cfg["noise_v"] = float(factor.get("v", 0.0))
    elif kind == "torque_limit":
        cfg["clip_ctrl"] = True
    elif kind == "ctrl_hold":
        cfg["hold"] = max(1, int(factor.get("steps", 1)))
    elif kind == "bias":
        cfg["bias_q"] = float(factor.get("q", 0.0))
        cfg["drift_q"] = float(factor.get("drift", 0.0))
    return cfg


def snapshot(
    step: int,
    data: mujoco.MjData,
    z: float,
    roll: float | None = None,
    pitch: float | None = None,
    tilt: float | None = None,
    speed: float | None = None,
) -> dict:
    row = {
        "step": step,
        "time": float(data.time),
        "z": z,
        "qpos_head": data.qpos[:7].tolist(),
        "qvel_head": data.qvel[:6].tolist(),
        "ctrl": data.ctrl.tolist(),
    }
    if roll is not None:
        row["roll"] = float(roll)
    if pitch is not None:
        row["pitch"] = float(pitch)
    if tilt is not None:
        row["tilt"] = float(tilt)
    if speed is not None:
        row["body_speed"] = float(speed)
    return row


def value_stats(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "min": None, "max": None, "count": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "count": int(arr.size),
    }


def format_stats(stats: dict) -> str:
    if stats["count"] == 0:
        return "no data"
    return (
        f"mean={stats['mean']:.4f}, "
        f"range=[{stats['min']:.4f}, {stats['max']:.4f}]"
    )


def simulate_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    friction: float,
    lateral_force: float,
    kp: float,
    kd: float,
    *,
    episode: int = 1,
    jitter: bool = True,
    keep_window: bool = False,
    record_trace: bool = False,
    factor: dict | None = None,
    gravity_ff: bool = False,
    ff_ctrl: np.ndarray | None = None,
    keyframe_ff: bool = False,
    target_override: np.ndarray | None = None,
    stance_ref_z: float | None = None,
    stance_loss_frac: float | None = None,
    foot_ids: list[int] | None = None,
    controller=None,
    controller_name: str | None = None,
) -> dict:
    cfg = control_from_factor(factor)
    delay = max(0, int(cfg["delay"]))
    noise_q = float(cfg["noise_q"])
    noise_v = float(cfg["noise_v"])
    clip_ctrl = bool(cfg["clip_ctrl"])
    hold = max(1, int(cfg["hold"]))
    bias_q = float(cfg["bias_q"])
    drift_q = float(cfg["drift_q"])
    n_joints = int(len(target_qpos))
    pert = (
        sample_episode_perturbation(episode, n_joints)
        if jitter
        else {
            "seed": episode_seed(episode),
            "joint_jitter": np.zeros(n_joints),
            "impulse_start": IMPULSE_START,
            "impulse_end": IMPULSE_END,
            "mass_scale": 1.0,
            "impulse_dir_x": 0.0,
            "impulse_dir_y": 1.0,
            "impulse_height": 0.0,
            "impulse_realistic": False,
        }
    )
    impulse_start = int(pert["impulse_start"])
    impulse_end = int(pert["impulse_end"])
    orig_mass = float(model.body_mass[torso_body_id])
    dt = float(model.opt.timestep)
    upright_needed = max(1, int(math.ceil(UPRIGHT_HOLD_S / max(dt, 1e-9))))
    settle_needed = max(1, int(math.ceil(SETTLE_HOLD_S / max(dt, 1e-9))))
    q_target = np.asarray(
        target_override if target_override is not None else target_qpos, dtype=float
    )

    try:
        model.body_mass[torso_body_id] = orig_mass * float(pert["mass_scale"])
        mujoco.mj_setConst(model, data)
        reset_to_stance(model, data)
        if jitter:
            data.qpos[7 : 7 + n_joints] = (
                np.asarray(data.qpos[7 : 7 + n_joints], dtype=float)
                + np.asarray(pert["joint_jitter"], dtype=float)
            )
            clip_hinge_qpos(model, data)
        set_sliding_friction(model, friction_geoms, friction)
        mujoco.mj_forward(model, data)
        z_start = float(data.qpos[2])
        key_ctrl = np.array(data.ctrl, copy=True)
        feet = list(foot_ids) if foot_ids else find_foot_geom_ids(model)
        body_xy0 = np.array(data.qpos[:2], dtype=float)
        foot_rel0 = {
            gid: np.array(data.geom_xpos[gid][:2], dtype=float) - body_xy0
            for gid in feet
        }
        contact_friction = None
        if data.ncon > 0:
            contact_friction = np.asarray(data.contact[0].friction, dtype=float).tolist()

        window: deque[dict] | None = deque(maxlen=WINDOW_SIZE) if keep_window else None
        ctrl_buf: deque[np.ndarray] = deque(maxlen=delay + 1)
        last_desired: np.ndarray | None = None
        limits = np.array(model.actuator_ctrlrange, copy=True)
        limit_abs = np.maximum(np.abs(limits[:, 0]), np.abs(limits[:, 1]))
        act_dofs, act_gears = actuator_dof_gears(model)
        obs_rng = (
            np.random.default_rng(int(pert["seed"]))
            if noise_q > 0.0 or noise_v > 0.0
            else None
        )
        zs: list[float] = []
        ys: list[float] = []
        rolls: list[float] = []
        pitches: list[float] = []
        fail_step = None
        fail_reason = None
        sag_step = None
        sag_below = False
        max_abs_ctrl = 0.0
        max_abs_ctrl_raw = 0.0
        max_util_raw = 0.0
        max_tau_over_limit = 0.0
        sat_steps = 0
        hit_limit = False
        min_z = z_start
        max_abs_roll = 0.0
        max_abs_pitch = 0.0
        max_tilt = 0.0
        max_body_speed = 0.0
        upright_count = 0
        settle_count = 0
        settle_time_s = None
        pre_impact_z = z_start
        applied_ff = np.array(key_ctrl, copy=True) if keyframe_ff else ff_ctrl
        foot_slip_max = 0.0
        foot_slip_world_max = 0.0
        z_ref = STANCE_THRESH.get("z_ref") or STANCE_REF["z"] if stance_ref_z is None else stance_ref_z
        if stance_ref_z is not None:
            z_ref = stance_ref_z
        loss_frac = STANCE_LOSS_FRAC if stance_loss_frac is None else float(stance_loss_frac)
        settle_window = max(1, int(math.ceil(UPRIGHT_HOLD_S / max(dt, 1e-9))))
        measure_step = impulse_end + max(1, int(round(POST_IMPACT_MEASURE_S / max(dt, 1e-9))))
        z_settle = None
        foot_slip_at_measure = None
        body_speed_at_measure = None
        roll_at_measure = None
        foot_rel_trace: list[list[float]] = []
        speed_trace: list[float] = []
        mag = abs(float(lateral_force))
        if pert.get("impulse_realistic"):
            force_x = mag * float(pert.get("impulse_dir_x") or 0.0)
            force_y = mag * float(pert.get("impulse_dir_y") or 1.0)
        else:
            force_x = 0.0
            force_y = float(lateral_force)
        height_offset = float(pert.get("impulse_height") or 0.0)
        policy = (
            controller
            if controller is not None
            else make_controller(controller_name or ACTIVE_CONTROLLER, kp, kd)
        )
        if hasattr(policy, "reset"):
            policy.reset()

        for step in range(1, MAX_STEPS + 1):
            if last_desired is None or (step - 1) % hold == 0:
                last_desired = command_from_controller(
                    model,
                    data,
                    policy,
                    q_target,
                    step,
                    dt,
                    obs_rng,
                    noise_q,
                    noise_v,
                    bias_q + drift_q * (step - 1),
                    gravity_ff=gravity_ff,
                    ff_ctrl=applied_ff,
                )
            ctrl_buf.append(np.array(last_desired, copy=True))
            raw = np.array(ctrl_buf[0], copy=True)
            data.ctrl[:] = raw
            max_abs_ctrl_raw = max(max_abs_ctrl_raw, float(np.max(np.abs(raw))))
            util = float(np.max(np.abs(raw) / np.maximum(limit_abs, 1e-9)))
            max_util_raw = max(max_util_raw, util)
            exceeded = np.any(raw < limits[:, 0] - 1e-9) | np.any(
                raw > limits[:, 1] + 1e-9
            )
            np.clip(data.ctrl, limits[:, 0], limits[:, 1], out=data.ctrl)
            applied_util = np.abs(data.ctrl) / np.maximum(limit_abs, 1e-9)
            if bool(exceeded) or float(np.max(applied_util)) >= 1.0 - 1e-6:
                sat_steps += 1
                hit_limit = True
            max_abs_ctrl = max(max_abs_ctrl, float(np.max(np.abs(data.ctrl))))
            apply_torso_impulse(
                data,
                torso_body_id,
                step,
                force_y,
                impulse_start=impulse_start,
                impulse_end=impulse_end,
                force_x=force_x,
                height_offset=height_offset,
            )
            mujoco.mj_step(model, data)
            tau = np.abs(np.asarray(data.qfrc_actuator[act_dofs], dtype=float))
            tau_lim = np.maximum(np.abs(act_gears) * limit_abs, 1e-9)
            max_tau_over_limit = max(max_tau_over_limit, float(np.max(tau / tau_lim)))
            z = float(data.qpos[2])
            roll, pitch = freejoint_roll_pitch(data.qpos)
            tilt = tilt_from_vertical(data.qpos)
            speed = float(np.linalg.norm(data.qvel[:3]))
            min_z = min(min_z, z)
            max_abs_roll = max(max_abs_roll, abs(roll))
            max_abs_pitch = max(max_abs_pitch, abs(pitch))
            max_tilt = max(max_tilt, tilt)
            max_body_speed = max(max_body_speed, speed)
            body_xy = np.array(data.qpos[:2], dtype=float)
            slip_now = 0.0
            rels = []
            for gid in feet:
                xy = np.array(data.geom_xpos[gid][:2], dtype=float)
                rel = xy - body_xy
                slip_now = max(slip_now, float(np.linalg.norm(rel - foot_rel0[gid])))
                foot_slip_world_max = max(
                    foot_slip_world_max, float(np.linalg.norm(xy - (foot_rel0[gid] + body_xy0)))
                )
                rels.append(rel.tolist())
            foot_slip_max = max(foot_slip_max, slip_now)
            zs.append(z)
            ys.append(float(data.qpos[1]))
            if record_trace:
                rolls.append(roll)
                pitches.append(pitch)
                speed_trace.append(speed)
                foot_rel_trace.append([c for rel in rels for c in rel])
            if step == measure_step:
                z_settle = z
                foot_slip_at_measure = slip_now
                body_speed_at_measure = speed
                roll_at_measure = abs(roll)
            if window is not None:
                window.append(snapshot(step, data, z, roll, pitch, tilt, speed))
            if step == impulse_start - 1 or (impulse_start <= 1 and step == 1):
                pre_impact_z = z
            if not sag_below and z < SAG_Z:
                sag_below = True
                sag_step = step
            if tilt > TILT_FAIL_RAD:
                upright_count += 1
            else:
                upright_count = 0
            settled = (
                step > impulse_end
                and abs(z - pre_impact_z) <= SETTLE_Z_TOL
                and abs(roll) <= SETTLE_TILT_RAD
                and abs(pitch) <= SETTLE_TILT_RAD
                and speed <= SETTLE_SPEED
            )
            if settled:
                settle_count += 1
                if settle_time_s is None and settle_count >= settle_needed:
                    settle_time_s = float(data.time) - impulse_end * dt
            else:
                settle_count = 0
            reason = None
            if z < CONTACT_Z:
                reason = "contact_z"
            elif abs(roll) > TILT_FAIL_RAD or abs(pitch) > TILT_FAIL_RAD:
                reason = "tilt"
            elif upright_count >= upright_needed:
                reason = "upright_hold"
            if reason is not None:
                fail_step = step
                fail_reason = reason
                break
    finally:
        model.body_mass[torso_body_id] = orig_mass
        mujoco.mj_setConst(model, data)

    written = {
        geom_id: float(model.geom_friction[geom_id, 0]) for geom_id in friction_geoms
    }
    n_tail = min(len(zs), settle_window) if zs else 0
    z_tail = float(np.mean(zs[-n_tail:])) if n_tail else float(z_start)
    min_z_after_settle = float(np.min(zs[-n_tail:])) if n_tail else float(z_start)
    if z_settle is None:
        z_settle = z_tail
    if foot_slip_at_measure is None:
        foot_slip_at_measure = float(foot_slip_max)
    z_after_settle = float(z_settle)
    fall_kind = fail_reason if fail_reason in ("tilt", "contact_z", "upright_hold") else None
    fall = fall_kind is not None
    if fall:
        fail_reason = "fall"
    tilt_fail = fall
    stance_loss = False
    degradation = False
    z_thr = STANCE_THRESH.get("z_threshold")
    slip_thr = STANCE_THRESH.get("slip_threshold")
    z_ref_use = z_ref if z_ref is not None else STANCE_THRESH.get("z_ref")
    if not fall:
        z_bad = z_thr is not None and z_after_settle < float(z_thr)
        slip_bad = slip_thr is not None and foot_slip_at_measure > float(slip_thr)
        if z_bad or slip_bad:
            stance_loss = True
            fail_reason = "stance_loss"
            fail_step = int(measure_step)
        elif (
            z_ref_use is not None
            and float(z_ref_use) > 1e-9
            and z_after_settle < (1.0 - DEGRADATION_FRAC) * float(z_ref_use)
        ):
            degradation = True
            fail_reason = None
    failed = fall or stance_loss
    metrics = {
        "episode": int(episode),
        "seed": int(pert["seed"]),
        "friction": float(friction),
        "lateral_force": float(lateral_force),
        "impulse_start": impulse_start,
        "impulse_end": impulse_end,
        "mass_scale": float(pert["mass_scale"]),
        "joint_jitter_rms": float(np.sqrt(np.mean(np.square(pert["joint_jitter"])))),
        "failed": failed,
        "fail_reason": fail_reason,
        "fail_fall": fall,
        "fail_tilt": tilt_fail,
        "fail_stance_loss": stance_loss,
        "fall_kind": fall_kind,
        "degradation": degradation,
        "fail_step": fail_step,
        "fail_time": None if fail_step is None else float(fail_step) * dt,
        "impact_phase": None
        if fail_step is None
        else impact_phase(fail_step, impulse_start, impulse_end),
        "sag_below_0_20": sag_below,
        "sag_step": sag_step,
        "min_z": float(min_z),
        "z_after_settle": z_after_settle,
        "z_settle": float(z_after_settle),
        "min_z_after_settle": min_z_after_settle,
        "measure_step": int(measure_step),
        "max_abs_roll": float(max_abs_roll),
        "max_abs_pitch": float(max_abs_pitch),
        "max_tilt": float(max_tilt),
        "max_body_speed": float(max_body_speed),
        "foot_slip_max": float(foot_slip_at_measure),
        "foot_slip_rel": float(foot_slip_at_measure),
        "foot_slip_world_max": float(foot_slip_world_max),
        "body_speed_at_measure": body_speed_at_measure,
        "roll_at_measure": roll_at_measure,
        "settle_time_s": settle_time_s,
        "stance_ref_z": None if z_ref is None else float(z_ref),
        "stance_loss_frac": float(loss_frac),
        "z_start": float(z_start),
        "z_end": float(zs[-1]) if zs else float(z_start),
        "n_steps": len(zs),
    }
    result = {
        **metrics,
        "kp": float(kp),
        "kd": float(kd),
        "written_sliding": written,
        "contact_friction": contact_friction,
        "max_abs_ctrl": max_abs_ctrl,
        "max_abs_ctrl_raw": max_abs_ctrl_raw,
        "max_util_raw": max_util_raw,
        "max_tau_over_limit": float(max_tau_over_limit),
        "saturation_frac": (sat_steps / len(zs)) if zs else 0.0,
        "hit_limit": hit_limit,
        "y_end": ys[-1] if ys else 0.0,
        "max_abs_y": max((abs(y) for y in ys), default=0.0),
        "z": zs if record_trace else [],
        "y": ys if record_trace else [],
        "roll": rolls,
        "pitch": pitches,
        "speed": speed_trace,
        "foot_rel": foot_rel_trace,
        "final_z": float(zs[-1]) if zs else float(z_start),
    }
    if window is not None:
        result["window"] = list(window)
    return result


def run_controlled_episode(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    friction: float,
    lateral_force: float,
    torso_body_id: int,
    kp: float = KP,
    kd: float = KD,
    clip_ctrl: bool = False,
    action_delay: int = 0,
    obs_noise_q: float = 0.0,
    obs_noise_v: float = 0.0,
    obs_bias_q: float = 0.0,
    obs_drift_q: float = 0.0,
    noise_seed: int | None = None,
    gravity_ff: bool = False,
    ff_ctrl: np.ndarray | None = None,
    jitter: bool = False,
    record_trace: bool = True,
    episode: int = 1,
) -> dict:
    factor = None
    if clip_ctrl:
        factor = {"kind": "torque_limit", "scale": 1.0}
    elif action_delay:
        factor = {"kind": "delay", "steps": action_delay}
    elif obs_noise_q or obs_noise_v:
        factor = {"kind": "noise", "q": obs_noise_q, "v": obs_noise_v}
    elif obs_bias_q or obs_drift_q:
        factor = {"kind": "bias", "q": obs_bias_q, "drift": obs_drift_q}
    result = simulate_episode(
        model,
        data,
        target_qpos,
        friction_geoms,
        torso_body_id,
        friction,
        lateral_force,
        kp,
        kd,
        episode=episode if noise_seed is None else int(noise_seed),
        jitter=jitter,
        keep_window=False,
        record_trace=record_trace,
        factor=factor,
        gravity_ff=gravity_ff,
        ff_ctrl=ff_ctrl,
    )
    return result


def probe_friction_effect(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    lateral_force: float = COM_PROBE_FORCE,
) -> None:
    print(
        f"\nfriction probe: Fy={lateral_force:.1f} N on steps "
        f"{IMPULSE_START}..{IMPULSE_END}, friction={list(PROBE_FRICTIONS)}"
    )
    results = [
        run_controlled_episode(
            model,
            data,
            target_qpos,
            friction_geoms,
            mu,
            lateral_force,
            torso_body_id,
            kp=kp,
            kd=kd,
        )
        for mu in PROBE_FRICTIONS
    ]
    for result in results:
        status = (
            f"failed at step {result['fail_step']}"
            if result["fail_step"] is not None
            else "survived"
        )
        print(
            f"  mu={result['friction']:.2f}: written {result['written_sliding']}, "
            f"contact.friction={result['contact_friction']}, {status}, "
            f"z_end={result['z_end']:.4f}, z_min={result['min_z']:.4f}, "
            f"y_end={result['y_end']:.4f}, max|y|={result['max_abs_y']:.4f}"
        )

    z0, z1 = np.asarray(results[0]["z"]), np.asarray(results[1]["z"])
    y0, y1 = np.asarray(results[0]["y"]), np.asarray(results[1]["y"])
    n = min(len(z0), len(z1))
    z_diff = float(np.max(np.abs(z0[:n] - z1[:n])))
    y_diff = float(np.max(np.abs(y0[:n] - y1[:n])))
    print(f"  max |Δz|={z_diff:.6f}, max |Δy|={y_diff:.6f} (over the first {n} steps)")
    if z_diff < 1e-9 and y_diff < 1e-9:
        print("  warning: traces match, friction does not affect the contact")
    else:
        print("  traces differ, friction affects stability")


def probe_com_offset(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    lateral_force: float = COM_PROBE_FORCE,
) -> None:
    origin = read_body_ipos(model, torso_body_id)
    print(
        f"\ntorso COM offset probe: body_id={torso_body_id} "
        f"body_ipos0={origin.tolist()}  mu={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} N on steps {IMPULSE_START}..{IMPULSE_END}"
    )
    results: list[dict] = []
    try:
        for label, offset in COM_PROBE_OFFSETS:
            ipos = origin + np.asarray(offset, dtype=float)
            written = write_body_ipos(model, data, torso_body_id, ipos)
            result = run_controlled_episode(
                model,
                data,
                target_qpos,
                friction_geoms,
                COM_PROBE_FRICTION,
                lateral_force,
                torso_body_id,
                kp=kp,
                kd=kd,
            )
            result["label"] = label
            result["offset"] = list(offset)
            result["ipos"] = written.tolist()
            results.append(result)
            status = (
                f"failed at step {result['fail_step']}"
                if result["fail_step"] is not None
                else "survived"
            )
            print(
                f"  {label}: offset={list(offset)} ipos={written.tolist()}  "
                f"{status}, z_end={result['z_end']:.4f}, z_min={result['min_z']:.4f}"
            )
    finally:
        restored = write_body_ipos(model, data, torso_body_id, origin)
        if float(np.max(np.abs(restored - origin))) > 1e-12:
            raise RuntimeError(
                f"failed to restore body_ipos[{torso_body_id}]={origin.tolist()}, "
                f"current value {restored.tolist()}"
            )
        print(f"  COM restored: body_ipos={restored.tolist()}")

    if results:
        z0 = np.asarray(results[0]["z"])
        any_diff = False
        for result in results[1:]:
            z1 = np.asarray(result["z"])
            n = min(len(z0), len(z1))
            z_diff = float(np.max(np.abs(z0[:n] - z1[:n])))
            any_diff = any_diff or z_diff >= 1e-9
            print(
                f"  max |Δz| zero vs {result['label']}={z_diff:.6f} "
                f"(over the first {n} steps)"
            )
        if not any_diff:
            print("  warning: traces match, COM offset has no effect")
        else:
            print("  traces differ, COM offset affects stability")


def probe_torque_limit(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    lateral_force: float = COM_PROBE_FORCE,
) -> list[dict]:
    origin = stock_ctrlrange(model)
    print(
        f"\ntorque-limit probe: mu={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} N on steps {IMPULSE_START}..{IMPULSE_END}  "
        f"ctrlrange0 hip=[{origin[0, 0]:.2f}, {origin[0, 1]:.2f}]  "
        f"ctrllimited={bool(model.actuator_ctrllimited[0])}"
    )
    results: list[dict] = []
    try:
        for scale in TORQUE_PROBE_SCALES:
            written = apply_torque_limit(model, scale)
            result = run_controlled_episode(
                model,
                data,
                target_qpos,
                friction_geoms,
                COM_PROBE_FRICTION,
                lateral_force,
                torso_body_id,
                kp=kp,
                kd=kd,
                clip_ctrl=True,
            )
            label = f"{scale * 100:.0f}%"
            result["label"] = label
            result["scale"] = scale
            results.append(result)
            hip_limit = float(np.min(np.abs(written[0])))
            calf_limit = float(np.min(np.abs(written[2])))
            hit = "yes" if result["hit_limit"] else "no"
            sat = float(result["saturation_frac"])
            print(
                f"  {label} ctrlrange: hip=[{written[0, 0]:.2f}, {written[0, 1]:.2f}] "
                f"calf=[{written[2, 0]:.2f}, {written[2, 1]:.2f}]  "
                f"{episode_status(result)}, z_end={result['z_end']:.4f}, "
                f"z_min={result['min_z']:.4f}  "
                f"max|ctrl|_raw={result['max_abs_ctrl_raw']:.3f}  "
                f"max|tau|/limit={result['max_tau_over_limit']:.3f}  "
                f"sat={format_sat_pct(sat)} ({saturation_note(sat)})  "
                f"ceiling hip={hip_limit:.2f} calf={calf_limit:.2f}  hit={hit}"
            )
    finally:
        restored = apply_torque_limit(model, 1.0)
        if float(np.max(np.abs(restored - origin))) > 1e-12:
            raise RuntimeError("failed to restore actuator_ctrlrange")
        print(
            f"  ctrlrange restored: hip=[{restored[0, 0]:.2f}, {restored[0, 1]:.2f}]"
        )

    reduced_hit = [r for r in results if r["scale"] < 1.0 - 1e-12]
    if reduced_hit and not any(r["hit_limit"] for r in reduced_hit):
        print(
            "  warning: max|ctrl| never reached the limit, "
            "the torque ceiling did not engage on this scenario"
        )
    else:
        any_diff = print_z_deltas(results, "label")
        if any_diff:
            print("  traces differ, the torque ceiling affects stability")
        elif reduced_hit and any(r["hit_limit"] for r in reduced_hit):
            print("  ceiling engaged (ctrl hit the limit)")
    return [
        {
            "scale": r["scale"],
            "label": r["label"],
            **torque_episode(r),
            "status": episode_status(r),
        }
        for r in results
    ]


def probe_action_delay(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    lateral_force: float = COM_PROBE_FORCE,
) -> None:
    print(
        f"\ncommand-delay probe (#6): mu={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} N on steps {IMPULSE_START}..{IMPULSE_END}"
    )
    results: list[dict] = []
    for delay in DELAY_PROBE_STEPS:
        result = run_controlled_episode(
            model,
            data,
            target_qpos,
            friction_geoms,
            COM_PROBE_FRICTION,
            lateral_force,
            torso_body_id,
            kp=kp,
            kd=kd,
            action_delay=delay,
        )
        label = f"delay={delay}"
        result["label"] = label
        results.append(result)
        print(
            f"  {label}: {episode_status(result)}, "
            f"z_end={result['z_end']:.4f}, z_min={result['min_z']:.4f}"
        )
    any_diff = print_z_deltas(results, "label")
    if not any_diff:
        print("  warning: traces match, command delay has no effect")
    else:
        print("  traces differ, command delay affects stability")


def probe_joint_damping(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    lateral_force: float = COM_PROBE_FORCE,
) -> None:
    origin = read_dof_damping(model)
    legs = leg_dof_ids(model)
    print(
        f"\njoint-damping probe (#7): mu={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} N on steps {IMPULSE_START}..{IMPULSE_END}  "
        f"leg dof_damping 1x={origin[legs].tolist()}"
    )
    results: list[dict] = []
    try:
        for scale in DAMPING_PROBE_SCALES:
            damped = origin.copy()
            damped[legs] = origin[legs] * scale
            written = write_dof_damping(model, damped)
            result = run_controlled_episode(
                model,
                data,
                target_qpos,
                friction_geoms,
                COM_PROBE_FRICTION,
                lateral_force,
                torso_body_id,
                kp=kp,
                kd=kd,
            )
            label = f"{scale:g}x"
            result["label"] = label
            results.append(result)
            print(
                f"  {label}: leg_damping={written[legs].tolist()}  "
                f"{episode_status(result)}, z_end={result['z_end']:.4f}, "
                f"z_min={result['min_z']:.4f}"
            )
    finally:
        restored = write_dof_damping(model, origin)
        if float(np.max(np.abs(restored - origin))) > 1e-12:
            raise RuntimeError("failed to restore dof_damping")
        print(f"  dof_damping restored: legs={restored[legs].tolist()}")

    any_diff = print_z_deltas(results, "label")
    if not any_diff:
        print("  warning: traces match, joint damping has no effect")
    else:
        print("  traces differ, joint damping affects stability")


def probe_body_inertia(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    lateral_force: float = COM_PROBE_FORCE,
) -> None:
    origin = read_body_inertia(model, torso_body_id)
    print(
        f"\ntorso-inertia probe (#8): body_id={torso_body_id} "
        f"body_inertia0={origin.tolist()}  mu={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} N on steps {IMPULSE_START}..{IMPULSE_END}"
    )
    results: list[dict] = []
    try:
        for scale in INERTIA_PROBE_SCALES:
            written = write_body_inertia(model, data, torso_body_id, origin * scale)
            result = run_controlled_episode(
                model,
                data,
                target_qpos,
                friction_geoms,
                COM_PROBE_FRICTION,
                lateral_force,
                torso_body_id,
                kp=kp,
                kd=kd,
            )
            result["label"] = f"{scale:g}x"
            results.append(result)
            print_probe_line(
                result,
                extra=f"inertia={np.round(written, 5).tolist()}",
                ref=results[0],
            )
    finally:
        restored = write_body_inertia(model, data, torso_body_id, origin)
        if float(np.max(np.abs(restored - origin))) > 1e-12:
            raise RuntimeError("failed to restore body_inertia")
        print(f"  body_inertia restored: {restored.tolist()}")

    any_diff = print_z_deltas(results, "label")
    if not any_diff:
        print("  warning: traces match, torso inertia has no effect")
    else:
        print("  traces differ, torso inertia affects stability")


def probe_floor_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    lateral_force: float = COM_PROBE_FORCE,
) -> None:
    origin_ref, origin_imp = read_geom_contact(model, friction_geoms)
    print(
        f"\nfloor-contact probe (#9): geom_ids={friction_geoms} "
        f"solref0={origin_ref[0].tolist()} solimp0={origin_imp[0].tolist()}  "
        f"mu={COM_PROBE_FRICTION:g} Fy={lateral_force:.1f} N "
        f"on steps {IMPULSE_START}..{IMPULSE_END}"
    )
    results: list[dict] = []
    try:
        for label, solref, solimp in CONTACT_PROBE_VARIANTS:
            if solref is None:
                ref_mat, imp_mat = origin_ref, origin_imp
            else:
                ref_mat = np.tile(np.asarray(solref, dtype=float), (len(friction_geoms), 1))
                imp_mat = np.tile(np.asarray(solimp, dtype=float), (len(friction_geoms), 1))
            written_ref, written_imp = write_geom_contact(
                model, friction_geoms, ref_mat, imp_mat
            )
            result = run_controlled_episode(
                model,
                data,
                target_qpos,
                friction_geoms,
                COM_PROBE_FRICTION,
                lateral_force,
                torso_body_id,
                kp=kp,
                kd=kd,
            )
            result["label"] = label
            results.append(result)
            print_probe_line(
                result,
                extra=(
                    f"solref={written_ref[0].tolist()} "
                    f"solimp={written_imp[0].tolist()}"
                ),
                ref=results[0],
            )
    finally:
        restored_ref, restored_imp = write_geom_contact(
            model, friction_geoms, origin_ref, origin_imp
        )
        print(
            f"  contact restored: solref={restored_ref[0].tolist()} "
            f"solimp={restored_imp[0].tolist()}"
        )

    any_diff = print_z_deltas(results, "label")
    if not any_diff:
        print("  warning: traces match, contact softness has no effect")
    else:
        print("  traces differ, contact softness affects stability")


def probe_observation_noise(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    lateral_force: float = COM_PROBE_FORCE,
) -> None:
    print(
        f"\nmeasurement-noise probe (#10): mu={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} N on steps {IMPULSE_START}..{IMPULSE_END}  "
        f"seed={SEED}"
    )
    results: list[dict] = []
    for label, noise_q, noise_v in NOISE_PROBE_VARIANTS:
        result = run_controlled_episode(
            model,
            data,
            target_qpos,
            friction_geoms,
            COM_PROBE_FRICTION,
            lateral_force,
            torso_body_id,
            kp=kp,
            kd=kd,
            obs_noise_q=noise_q,
            obs_noise_v=noise_v,
            noise_seed=SEED,
        )
        result["label"] = label
        results.append(result)
        print_probe_line(
            result,
            extra=f"σq={noise_q:g} σv={noise_v:g}",
            ref=results[0],
        )
    any_diff = print_z_deltas(results, "label")
    if not any_diff:
        print("  warning: traces match, measurement noise has no effect")
    else:
        print("  traces differ, measurement noise affects stability")


def probe_sensor_bias(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    lateral_force: float = COM_PROBE_FORCE,
) -> None:
    print(
        f"\nsensor-bias probe: mu={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} N on steps {IMPULSE_START}..{IMPULSE_END}"
    )
    results: list[dict] = []
    for label, bias_q, drift_q in BIAS_PROBE_VARIANTS:
        result = run_controlled_episode(
            model,
            data,
            target_qpos,
            friction_geoms,
            COM_PROBE_FRICTION,
            lateral_force,
            torso_body_id,
            kp=kp,
            kd=kd,
            obs_bias_q=bias_q,
            obs_drift_q=drift_q,
        )
        result["label"] = label
        results.append(result)
        print_probe_line(
            result,
            extra=f"bias={bias_q:g} rad  drift={drift_q:g} rad/step",
            ref=results[0],
        )
    any_diff = print_z_deltas(results, "label")
    if not any_diff:
        print("  warning: traces match, sensor bias has no effect")
    else:
        print("  traces differ, sensor bias affects stability")


def probe_stance_hold(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
) -> dict:
    result = run_controlled_episode(
        model,
        data,
        target_qpos,
        friction_geoms,
        STANCE_CHECK_FRICTION,
        STANCE_CHECK_FORCE,
        torso_body_id,
        kp=kp,
        kd=kd,
    )
    status = "failed" if result["fail_step"] is not None else "survived"
    print(
        f"stance check kp={kp:g} kd={kd:g}: z_start={result['z_start']:.4f} "
        f"z_end={result['z_end']:.4f} {status}"
    )
    return result


def run_stress_batch(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    floor_id: int,
    specs: list[tuple[int, float, float]],
    kp: float,
    kd: float,
    keep_windows: bool,
    factor: dict | None = None,
    controller_name: str | None = None,
) -> dict:
    episodes: list[dict] = []
    failed_episodes: list[dict] = []
    failed_friction: list[float] = []
    failed_force: list[float] = []
    survived_friction: list[float] = []
    survived_force: list[float] = []
    extra_ctrl = factor.get("ctrl") if isinstance(factor, dict) else None
    gravity_ff = bool(extra_ctrl.get("gravity_ff")) if extra_ctrl else False
    keyframe_ff = bool(extra_ctrl.get("keyframe_ff")) if extra_ctrl else False
    ff_ctrl = extra_ctrl.get("ff_ctrl") if extra_ctrl else None
    if ff_ctrl is not None:
        ff_ctrl = np.asarray(ff_ctrl, dtype=float)
    target_override = extra_ctrl.get("target_qpos") if extra_ctrl else None
    if target_override is not None:
        target_override = np.asarray(target_override, dtype=float)
    stance_ref_z = STANCE_REF["z"]
    stance_loss_frac = STANCE_REF["frac"]
    foot_ids = find_foot_geom_ids(model)

    for episode, friction, lateral_force in specs:
        result = simulate_episode(
            model,
            data,
            target_qpos,
            friction_geoms,
            torso_body_id,
            friction,
            lateral_force,
            kp,
            kd,
            episode=int(episode),
            jitter=True,
            keep_window=keep_windows,
            record_trace=False,
            factor=factor,
            gravity_ff=gravity_ff,
            keyframe_ff=keyframe_ff,
            ff_ctrl=ff_ctrl,
            target_override=target_override,
            stance_ref_z=stance_ref_z,
            stance_loss_frac=stance_loss_frac,
            foot_ids=foot_ids,
            controller_name=controller_name or ACTIVE_CONTROLLER,
        )
        record = {
            **{
                k: v
                for k, v in result.items()
                if k not in ("z", "y", "roll", "pitch", "speed", "foot_rel")
            },
            "torso_body_id": int(torso_body_id),
            "floor_geom_id": int(floor_id),
            "kp": kp,
            "kd": kd,
        }
        if keep_windows and "window" in result:
            record["window"] = result["window"]
        elif "window" in record:
            record.pop("window", None)
        episodes.append({k: v for k, v in record.items() if k != "window"})
        if result["failed"]:
            failed_episodes.append(record)
            failed_friction.append(float(friction))
            failed_force.append(float(lateral_force))
        else:
            survived_friction.append(float(friction))
            survived_force.append(float(lateral_force))

    return summarize_episode_lists(
        kp,
        kd,
        episodes,
        failed_episodes,
        failed_friction,
        failed_force,
        survived_friction,
        survived_force,
    )


def summarize_episode_lists(
    kp: float,
    kd: float,
    episodes: list[dict],
    failed_episodes: list[dict],
    failed_friction: list[float],
    failed_force: list[float],
    survived_friction: list[float],
    survived_force: list[float],
) -> dict:
    n_episodes = len(episodes)
    n_failures = len(failed_episodes)
    n_sag = sum(1 for ep in episodes if ep.get("sag_below_0_20"))
    n_tilt = sum(1 for ep in episodes if ep.get("fail_fall") or ep.get("fail_tilt"))
    n_stance_loss = sum(1 for ep in episodes if ep.get("fail_stance_loss"))
    n_degradation = sum(1 for ep in episodes if ep.get("degradation"))
    n_before = sum(
        1
        for ep in failed_episodes
        if ep.get("impact_phase") == "before_impact"
        or (
            ep.get("fail_step") is not None
            and int(ep["fail_step"]) < int(ep.get("impulse_start") or IMPULSE_START)
        )
    )
    reasons: dict[str, int] = {}
    for ep in failed_episodes:
        key = str(ep.get("fail_reason") or "unknown")
        reasons[key] = reasons.get(key, 0) + 1
    lo, hi = wilson_ci(n_failures, n_episodes)
    sag_lo, sag_hi = wilson_ci(n_sag, n_episodes)
    min_zs = [float(ep["min_z"]) for ep in episodes if ep.get("min_z") is not None]
    max_rolls = [
        float(ep["max_abs_roll"]) for ep in episodes if ep.get("max_abs_roll") is not None
    ]
    return {
        "kp": kp,
        "kd": kd,
        "n_episodes": n_episodes,
        "n_failures": n_failures,
        "n_survived": n_episodes - n_failures,
        "failure_rate": n_failures / n_episodes if n_episodes else 0.0,
        "failure_rate_ci95": [lo, hi],
        "n_sag": n_sag,
        "sag_rate": n_sag / n_episodes if n_episodes else 0.0,
        "sag_rate_ci95": [sag_lo, sag_hi],
        "n_tilt": n_tilt,
        "tilt_rate": n_tilt / n_episodes if n_episodes else 0.0,
        "tilt_rate_ci95": list(wilson_ci(n_tilt, n_episodes)),
        "n_fall": n_tilt,
        "fall_rate": n_tilt / n_episodes if n_episodes else 0.0,
        "fall_rate_ci95": list(wilson_ci(n_tilt, n_episodes)),
        "n_stance_loss": n_stance_loss,
        "stance_loss_rate": n_stance_loss / n_episodes if n_episodes else 0.0,
        "stance_loss_rate_ci95": list(wilson_ci(n_stance_loss, n_episodes)),
        "n_degradation": n_degradation,
        "degradation_rate": n_degradation / n_episodes if n_episodes else 0.0,
        "degradation_rate_ci95": list(wilson_ci(n_degradation, n_episodes)),
        "stance_ref_z": STANCE_REF.get("z"),
        "stance_loss_frac": STANCE_REF.get("frac"),
        "n_before_impact": n_before,
        "n_at_or_after_impact": n_failures - n_before,
        "fail_reasons": reasons,
        "failed_episodes": failed_episodes,
        "failed_friction": failed_friction,
        "failed_force": failed_force,
        "survived_friction": survived_friction,
        "survived_force": survived_force,
        "episodes": episodes,
        "min_z_mean": float(np.mean(min_zs)) if min_zs else None,
        "max_abs_roll_mean": float(np.mean(max_rolls)) if max_rolls else None,
        "max_tau_over_limit_mean": float(
            np.mean([float(ep["max_tau_over_limit"]) for ep in episodes if ep.get("max_tau_over_limit") is not None])
        ) if any(ep.get("max_tau_over_limit") is not None for ep in episodes) else None,
        "saturation_frac_mean": float(
            np.mean([float(ep["saturation_frac"]) for ep in episodes if ep.get("saturation_frac") is not None])
        ) if any(ep.get("saturation_frac") is not None for ep in episodes) else None,
    }


def sample_uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = float(bounds[0]), float(bounds[1])
    if abs(hi - lo) <= 1e-15:
        return lo
    return float(rng.uniform(lo, hi))


def generate_episode_specs(
    n_episodes: int,
    seed: int,
    friction_range: tuple[float, float] | None = None,
    force_range: tuple[float, float] | None = None,
) -> list[tuple[int, float, float]]:
    mu_range = friction_range or FRICTION_RANGE
    fy_range = force_range or LATERAL_FORCE_RANGE
    rng = np.random.default_rng(seed)
    return [
        (
            episode,
            sample_uniform(rng, mu_range),
            sample_uniform(rng, fy_range),
        )
        for episode in range(1, n_episodes + 1)
    ]


def split_chunks(
    specs: list[tuple[int, float, float]], chunk_size: int
) -> list[list[tuple[int, float, float]]]:
    size = max(1, int(chunk_size))
    return [specs[i : i + size] for i in range(0, len(specs), size)]


def _run_chunk(payload: dict) -> dict:
    controllers.apply_import_root(payload.get("controller_import_root"))
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    floor_id = find_floor_geom_id(model)
    foot_ids = find_foot_geom_ids(model)
    torso_body_id = find_torso_body_id(model)
    friction_geoms = [floor_id] + [geom_id for geom_id in foot_ids if geom_id != floor_id]
    target_qpos = np.asarray(payload["target_qpos"], dtype=float)
    factor = payload.get("factor")
    apply_model_factor(model, data, friction_geoms, torso_body_id, factor)
    if payload.get("stance_ref_z") is not None:
        STANCE_REF["z"] = float(payload["stance_ref_z"])
    if payload.get("stance_loss_frac") is not None:
        STANCE_REF["frac"] = float(payload["stance_loss_frac"])
    if payload.get("stance_thresh"):
        STANCE_THRESH.update(payload["stance_thresh"])
    if payload.get("impulse_cfg"):
        IMPULSE_CFG.update(payload["impulse_cfg"])
    return run_stress_batch(
        model,
        data,
        target_qpos,
        friction_geoms,
        torso_body_id,
        floor_id,
        payload["specs"],
        float(payload["kp"]),
        float(payload["kd"]),
        bool(payload["keep_windows"]),
        factor=factor,
        controller_name=payload.get("controller") or ACTIVE_CONTROLLER,
    )


def merge_batch_parts(parts: list[dict], kp: float, kd: float) -> dict:
    episodes: list[dict] = []
    failed_episodes: list[dict] = []
    failed_friction: list[float] = []
    failed_force: list[float] = []
    survived_friction: list[float] = []
    survived_force: list[float] = []
    for part in parts:
        episodes.extend(part.get("episodes") or [])
        failed_episodes.extend(part["failed_episodes"])
        failed_friction.extend(part["failed_friction"])
        failed_force.extend(part["failed_force"])
        survived_friction.extend(part["survived_friction"])
        survived_force.extend(part["survived_force"])
    episodes.sort(key=lambda ep: int(ep["episode"]))
    failed_episodes.sort(key=lambda ep: int(ep["episode"]))
    return summarize_episode_lists(
        kp,
        kd,
        episodes,
        failed_episodes,
        failed_friction,
        failed_force,
        survived_friction,
        survived_force,
    )


def run_stress_batch_parallel(
    specs: list[tuple[int, float, float]],
    kp: float,
    kd: float,
    keep_windows: bool,
    target_qpos: np.ndarray,
    factor: dict | None = None,
) -> dict:
    chunks = split_chunks(specs, CHUNK_SIZE)
    n_workers = min(len(chunks), os.cpu_count() or 1)
    controller_spec = controllers.canonical_controller_spec(ACTIVE_CONTROLLER)
    import_root = controllers.CONTROLLER_IMPORT_ROOT
    payloads = [
        {
            "specs": chunk,
            "kp": kp,
            "kd": kd,
            "keep_windows": keep_windows,
            "target_qpos": np.asarray(target_qpos, dtype=float).tolist(),
            "factor": factor,
            "stance_ref_z": STANCE_REF.get("z"),
            "stance_loss_frac": STANCE_REF.get("frac"),
            "stance_thresh": dict(STANCE_THRESH),
            "impulse_cfg": dict(IMPULSE_CFG),
            "controller": controller_spec,
            "controller_import_root": import_root,
        }
        for chunk in chunks
    ]
    started = time.perf_counter()
    use_pool = len(chunks) > 1
    if not use_pool:
        parts = [_run_chunk(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=max(1, n_workers)) as pool:
            parts = list(pool.map(_run_chunk, payloads))
    elapsed = time.perf_counter() - started
    batch = merge_batch_parts(parts, kp, kd)
    batch["elapsed_s"] = float(elapsed)
    batch["n_workers"] = int(max(1, n_workers) if use_pool else 1)
    batch["n_chunks"] = len(chunks)
    batch["used_process_pool"] = bool(use_pool)
    return batch


def batch_rate_summary(batch: dict) -> dict:
    return {
        "n_episodes": int(batch["n_episodes"]),
        "n_failures": int(batch["n_failures"]),
        "n_survived": int(batch["n_survived"]),
        "failure_rate": float(batch["failure_rate"]),
        "failure_rate_ci95": list(batch.get("failure_rate_ci95") or wilson_ci(
            int(batch["n_failures"]), int(batch["n_episodes"])
        )),
        "n_sag": int(batch.get("n_sag") or 0),
        "sag_rate": float(batch.get("sag_rate") or 0.0),
        "sag_rate_ci95": list(batch.get("sag_rate_ci95") or wilson_ci(
            int(batch.get("n_sag") or 0), int(batch["n_episodes"])
        )),
        "fail_reasons": dict(batch.get("fail_reasons") or {}),
        "n_tilt": int(batch.get("n_tilt") or 0),
        "tilt_rate": float(batch.get("tilt_rate") or 0.0),
        "tilt_rate_ci95": list(batch.get("tilt_rate_ci95") or wilson_ci(
            int(batch.get("n_tilt") or 0), int(batch["n_episodes"])
        )),
        "n_stance_loss": int(batch.get("n_stance_loss") or 0),
        "stance_loss_rate": float(batch.get("stance_loss_rate") or 0.0),
        "stance_loss_rate_ci95": list(batch.get("stance_loss_rate_ci95") or wilson_ci(
            int(batch.get("n_stance_loss") or 0), int(batch["n_episodes"])
        )),
        "n_before_impact": int(batch["n_before_impact"]),
        "n_at_or_after_impact": int(batch["n_at_or_after_impact"]),
        "min_z_mean": batch.get("min_z_mean"),
        "max_abs_roll_mean": batch.get("max_abs_roll_mean"),
        "elapsed_s": batch.get("elapsed_s"),
        "n_workers": batch.get("n_workers"),
        "n_chunks": batch.get("n_chunks"),
        "outcomes": [outcome_record(ep) for ep in batch.get("episodes") or []],
    }


def run_parameter_sweeps(
    target_qpos: np.ndarray,
    kp: float,
    kd: float,
    scenario: dict,
    n_episodes: int | None = None,
) -> dict:
    n_ep = int(N_SWEEP_EPISODES if n_episodes is None else n_episodes)
    specs = generate_episode_specs(
        n_ep,
        SEED,
        scenario["friction_range"],
        scenario["lateral_force_range"],
    )
    print(
        f"\nOFAT parameter check: {n_ep} episodes per level, "
        f"seed={SEED}, scenario={scenario['id']}, parallel workers"
    )
    print(
        "  exact paired McNemar vs baseline on shared seeds; "
        f"{HOLM_FAMILY_NOTE}"
    )
    parameters: list[dict] = []
    for spec in PARAMETER_SWEEPS:
        print(f"  {spec['name']} ({spec['id']}):")
        levels_out: list[dict] = []
        for level in spec["levels"]:
            batch = run_stress_batch_parallel(
                specs,
                kp,
                kd,
                False,
                target_qpos,
                factor=level["factor"],
            )
            row = {
                "id": level["id"],
                "label": level["label"],
                "factor": level["factor"],
                **batch_rate_summary(batch),
            }
            levels_out.append(row)
        baseline = next(
            (row for row in levels_out if row["id"] == spec["baseline"]),
            levels_out[0],
        )
        base_rate = float(baseline["failure_rate"])
        for row in levels_out:
            row["delta_vs_baseline"] = float(row["failure_rate"]) - base_rate
            if row["id"] == baseline["id"]:
                row["mcnemar"] = None
                row["p_mcnemar"] = None
                row["p_holm"] = None
                row["significant"] = False
                row["effect_label"] = "baseline"
            else:
                mc = mcnemar_exact_paired(baseline["outcomes"], row["outcomes"])
                row["mcnemar"] = mc
                row["p_mcnemar"] = mc["pvalue"]
                row["p_holm"] = None
                row["significant"] = False
                row["effect_label"] = "within noise"
        parameters.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "baseline": spec["baseline"],
                "levels": levels_out,
                "baseline_failure_rate": base_rate,
            }
        )
    family: list[tuple[int, int, float]] = []
    for pi, param in enumerate(parameters):
        for li, row in enumerate(param["levels"]):
            if row.get("p_mcnemar") is not None:
                family.append((pi, li, float(row["p_mcnemar"])))
    p_adj, sigs = holm_adjust([p for _, _, p in family])
    for (pi, li, _), padj, sig in zip(family, p_adj, sigs):
        row = parameters[pi]["levels"][li]
        row["p_holm"] = padj
        row["significant"] = bool(sig)
        row["effect_label"] = "significant" if sig else "within noise"
    print(
        f"  Holm: {len(family)} McNemar p-values "
        f"(non-baseline levels of every factor vs its baseline, scenario {scenario['id']})"
    )
    for param in parameters:
        print(f"  {param['name']} ({param['id']}):")
        levels_out = param["levels"]
        baseline = next(row for row in levels_out if row["id"] == param["baseline"])
        worst = max(levels_out, key=lambda row: float(row["failure_rate"]))
        sig_rows = [row for row in levels_out if row.get("significant")]
        for row in levels_out:
            lo, hi = row.get("failure_rate_ci95") or (0.0, 0.0)
            if row["id"] == baseline["id"]:
                extra = "baseline"
            else:
                p_raw = row.get("p_mcnemar")
                p_h = row.get("p_holm")
                extra = (
                    f"p_mcnemar={p_raw:.4g}  p_holm={p_h:.4g}  {row['effect_label']}"
                    if p_raw is not None and p_h is not None
                    else row["effect_label"]
                )
            print(
                f"    {row['label']}: failures={row['failure_rate'] * 100:.1f}% "
                f"({row['n_failures']}/{row['n_episodes']}, "
                f"Wilson 95% CI[{lo * 100:.1f},{hi * 100:.1f}])  "
                f"delta vs baseline={row['delta_vs_baseline'] * 100:+.1f} pp  "
                f"{extra}  t={row['elapsed_s']:.2f}s"
            )
        param["worst_level"] = worst["id"]
        param["worst_label"] = worst["label"]
        param["worst_failure_rate"] = float(worst["failure_rate"])
        param["worst_delta_vs_baseline"] = float(worst["failure_rate"]) - float(
            param["baseline_failure_rate"]
        )
        param["n_significant_levels"] = len(sig_rows)
        if sig_rows:
            leader = max(sig_rows, key=lambda row: float(row["failure_rate"]))
            param["significant_leader_id"] = leader["id"]
            param["significant_leader_label"] = leader["label"]
            print(
                f"    significant vs baseline (Holm): "
                + ", ".join(
                    f"{row['label']} (delta {row['delta_vs_baseline'] * 100:+.1f} pp)"
                    for row in sig_rows
                )
            )
        else:
            param["significant_leader_id"] = None
            param["significant_leader_label"] = None
            print("    no significant difference from baseline (Holm) — shifts are within noise")
    return {
        "n_episodes": n_ep,
        "seed": SEED,
        "kp": kp,
        "kd": kd,
        "scenario": scenario["id"],
        "scenario_label": scenario["label"],
        "friction_range": list(scenario["friction_range"]),
        "lateral_force_range": list(scenario["lateral_force_range"]),
        "failure_criterion": "fall OR stance_loss",
        "multiple_testing": {
            "method": "holm",
            "alpha": 0.05,
            "family": HOLM_FAMILY_NOTE,
            "n_tests": len(family),
            "scenario": scenario["id"],
        },
        "parameters": parameters,
    }


def build_config_meta(
    *,
    kp: float,
    kd: float,
    scenario: dict | None = None,
    extra: dict | None = None,
) -> dict:
    config = {
        "model_path": MODEL_PATH,
        "seed": SEED,
        "kp": float(kp),
        "kd": float(kd),
        "fall_height": FALL_HEIGHT,
        "contact_z": CONTACT_Z,
        "sag_z": SAG_Z,
        "tilt_fail_deg": math.degrees(TILT_FAIL_RAD),
        "upright_hold_s": UPRIGHT_HOLD_S,
        "joint_jitter_rad": JOINT_JITTER_RAD,
        "impulse_jitter_steps": IMPULSE_JITTER_STEPS,
        "mass_jitter": MASS_JITTER,
        "max_steps": MAX_STEPS,
        "impulse_start": IMPULSE_START,
        "impulse_end": IMPULSE_END,
        "n_envs": N_ENVS,
        "n_sweep_episodes": N_SWEEP_EPISODES,
        "n_grid_episodes": N_GRID_EPISODES,
        "chunk_size": CHUNK_SIZE,
    }
    if scenario is not None:
        config["scenario"] = scenario["id"]
        config["friction_range"] = list(scenario["friction_range"])
        config["lateral_force_range"] = list(scenario["lateral_force_range"])
    if extra:
        config.update(extra)
    blob = json.dumps(config, sort_keys=True, ensure_ascii=True, default=str)
    return {
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": config,
        "config_hash": hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16],
    }


def resolve_ofat_scenarios(run_mode: str, scenario: dict) -> list[dict]:
    if run_mode not in ("full", "ofat"):
        return []
    if scenario["id"] in ("extreme", "slip_only", "heavy_impact"):
        return [scenario]
    return [SCENARIOS["nominal"], SCENARIOS["moderate"]]


def print_batch_rates(prefix: str, batch: dict) -> None:
    n = int(batch["n_episodes"])
    k = int(batch["n_failures"])
    ks = int(batch.get("n_sag") or 0)
    lo, hi = batch.get("failure_rate_ci95") or wilson_ci(k, n)
    slo, shi = batch.get("sag_rate_ci95") or wilson_ci(ks, n)
    reasons = batch.get("fail_reasons") or {}
    reason_txt = ", ".join(f"{name}={count}" for name, count in sorted(reasons.items())) or "none"
    nt = int(batch.get("n_tilt") or 0)
    ns = int(batch.get("n_stance_loss") or 0)
    print(
        f"{prefix}failures={k}/{n}={batch['failure_rate'] * 100:.1f}% "
        f"CI95[{lo * 100:.1f},{hi * 100:.1f}]  "
        f"fall={nt}/{n}={float(batch.get('fall_rate') or batch.get('tilt_rate') or 0.0) * 100:.1f}%  "
        f"stance_loss={ns}/{n}={float(batch.get('stance_loss_rate') or 0.0) * 100:.1f}%  "
        f"degradation={int(batch.get('n_degradation') or 0)}/{n}="
        f"{float(batch.get('degradation_rate') or 0.0) * 100:.1f}%  "
        f"sag z<0.20={ks}/{n}={float(batch.get('sag_rate') or 0.0) * 100:.1f}% "
        f"CI95[{slo * 100:.1f},{shi * 100:.1f}]  "
        f"reasons: {reason_txt}"
        + (f"  t={batch['elapsed_s']:.2f}s" if batch.get("elapsed_s") is not None else "")
    )


def grid_row_from_batch(mu: float, fy: float, batch: dict, meta: dict) -> dict:
    lo, hi = batch.get("failure_rate_ci95") or wilson_ci(
        int(batch["n_failures"]), int(batch["n_episodes"])
    )
    slo, shi = batch.get("sag_rate_ci95") or wilson_ci(
        int(batch.get("n_sag") or 0), int(batch["n_episodes"])
    )
    return {
        "mu": float(mu),
        "fy": float(fy),
        "n_episodes": int(batch["n_episodes"]),
        "n_failures": int(batch["n_failures"]),
        "failure_rate": float(batch["failure_rate"]),
        "failure_ci_lo": float(lo),
        "failure_ci_hi": float(hi),
        "n_sag": int(batch.get("n_sag") or 0),
        "sag_rate": float(batch.get("sag_rate") or 0.0),
        "sag_ci_lo": float(slo),
        "sag_ci_hi": float(shi),
        "n_tilt": int(batch.get("n_tilt") or 0),
        "tilt_rate": float(batch.get("tilt_rate") or 0.0),
        "n_stance_loss": int(batch.get("n_stance_loss") or 0),
        "stance_loss_rate": float(batch.get("stance_loss_rate") or 0.0),
        "fail_reasons": json.dumps(batch.get("fail_reasons") or {}, ensure_ascii=False),
        "n_before_impact": int(batch["n_before_impact"]),
        "n_at_or_after_impact": int(batch["n_at_or_after_impact"]),
        "min_z_mean": batch.get("min_z_mean"),
        "max_abs_roll_mean": batch.get("max_abs_roll_mean"),
        "seed": SEED,
        "config_hash": meta["config_hash"],
        "created_at": meta["created_at"],
    }


GRID_CSV_FIELDS = [
    "mu",
    "fy",
    "n_episodes",
    "n_failures",
    "failure_rate",
    "failure_ci_lo",
    "failure_ci_hi",
    "n_sag",
    "sag_rate",
    "sag_ci_lo",
    "sag_ci_hi",
    "n_tilt",
    "tilt_rate",
    "n_stance_loss",
    "stance_loss_rate",
    "fail_reasons",
    "n_before_impact",
    "n_at_or_after_impact",
    "min_z_mean",
    "max_abs_roll_mean",
    "seed",
    "config_hash",
    "created_at",
]


def print_grid_table(rows: list[dict], mus: tuple, fys: tuple, rate_key: str, title: str) -> None:
    print(f"\n{title} (rows mu, columns Fy, %):")
    print("μ\\Fy" + "".join(f"{fy:8.0f}" for fy in fys))
    for mu in mus:
        cells = [
            next(r for r in rows if r["mu"] == mu and r["fy"] == fy)[rate_key]
            for fy in fys
        ]
        print(f"{mu:4.1f}" + "".join(f"{rate * 100:8.1f}" for rate in cells))


def run_force_mu_grid(
    target_qpos: np.ndarray,
    kp: float,
    kd: float,
    mus: tuple,
    fys: tuple,
    n_episodes: int,
    title: str,
) -> tuple[list[dict], dict]:
    meta = build_config_meta(
        kp=kp,
        kd=kd,
        extra={"grid_mu": list(mus), "grid_fy": list(fys), "n_grid_episodes": n_episodes},
    )
    print(
        f"\n{title}: {len(mus)}x{len(fys)} cells, "
        f"{n_episodes} episodes per cell, seed={SEED}, "
        f"config_hash={meta['config_hash']}"
    )
    rows: list[dict] = []
    for mu in mus:
        for fy in fys:
            specs = [
                (episode, float(mu), float(fy))
                for episode in range(1, n_episodes + 1)
            ]
            batch = run_stress_batch_parallel(specs, kp, kd, False, target_qpos)
            row = grid_row_from_batch(float(mu), float(fy), batch, meta)
            rows.append(row)
            print_batch_rates(f"  mu={mu:g} Fy={fy:6.1f} N: ", batch)
    return rows, meta


def measure_stance_reference(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    frac: float,
    announce_frac_rule: bool = True,
) -> float:
    result = simulate_episode(
        model,
        data,
        target_qpos,
        friction_geoms,
        torso_body_id,
        1.0,
        0.0,
        kp,
        kd,
        episode=1,
        jitter=False,
        keep_window=False,
        record_trace=True,
        stance_ref_z=None,
        stance_loss_frac=frac,
    )
    z_ref = float(result["z_after_settle"])
    STANCE_REF["z"] = z_ref
    STANCE_REF["frac"] = float(frac)
    STANCE_REF["kp"] = float(kp)
    STANCE_REF["kd"] = float(kd)
    z_lim = float(frac) * z_ref
    print(
        f"\nstance reference: controller {ACTIVE_CONTROLLER}, "
        f"PD kp={kp:g} kd={kd:g}, mu=1 Fy=0, no jitter\n"
        f"  z_start={result['z_start']:.4f} z_end={result['z_end']:.4f} "
        f"z_ref={z_ref:.4f} m"
    )
    if announce_frac_rule:
        print(
            f"  stance_loss if z_after_settle < {frac:.2f}x{z_ref:.4f} = {z_lim:.4f} m"
        )
    return z_ref


def run_bias_offset_check(target_qpos: np.ndarray, kp: float, kd: float) -> dict:
    scenario = SCENARIOS["moderate"]
    specs = generate_episode_specs(
        N_SWEEP_EPISODES,
        SEED,
        scenario["friction_range"],
        scenario["lateral_force_range"],
    )
    print(
        f"\nsensor-bias +0.05 rad probe on moderate "
        f"({N_SWEEP_EPISODES} episodes)"
    )
    base = run_stress_batch_parallel(specs, kp, kd, False, target_qpos)
    biased = run_stress_batch_parallel(
        specs,
        kp,
        kd,
        False,
        target_qpos,
        factor={"kind": "bias", "q": 0.05, "drift": 0.0},
    )
    print_batch_rates("  no bias: ", base)
    print_batch_rates("  +0.05 rad:    ", biased)
    return {"none": batch_rate_summary(base), "offset": batch_rate_summary(biased)}


def run_stance_hold_batch(target_qpos: np.ndarray, kp: float, kd: float) -> dict:
    specs = [(episode, 1.0, 0.0) for episode in range(1, N_STANCE_EPISODES + 1)]
    print(
        f"\nstance without a push: mu=1.0 Fy=0, {N_STANCE_EPISODES} episodes, "
        f"kp={kp:g} kd={kd:g} seed={SEED}"
    )
    batch = run_stress_batch_parallel(specs, kp, kd, False, target_qpos)
    print_batch_rates("  ", batch)
    if batch["n_failures"] != 0:
        print("  warning: the stance at mu=1 Fy=0 fails the current criterion")
    else:
        print("  stance holds under the current criterion: 0 failures")
    return batch


def run_scenario_rate_check(target_qpos: np.ndarray, kp: float, kd: float) -> list[dict]:
    rows: list[dict] = []
    print("\nfailure rates by scenario (current criterion vs sag z<0.20):")
    for name in ("nominal", "moderate", "extreme"):
        scenario = SCENARIOS[name]
        specs = generate_episode_specs(
            100, SEED, scenario["friction_range"], scenario["lateral_force_range"]
        )
        batch = run_stress_batch_parallel(specs, kp, kd, False, target_qpos)
        print_batch_rates(f"  {name}: ", batch)
        rows.append(
            {
                "scenario": name,
                "label": scenario["label"],
                **batch_rate_summary(batch),
            }
        )
    return rows


def run_baseline_grid(target_qpos: np.ndarray, kp: float, kd: float) -> list[dict]:
    rows, meta = run_force_mu_grid(
        target_qpos, kp, kd, GRID_MU, GRID_FY, N_GRID_EPISODES, "Fy/mu grid"
    )
    write_csv(GRID_PATH, rows, GRID_CSV_FIELDS)
    print(f"grid saved: {GRID_PATH}")
    print_grid_table(rows, GRID_MU, GRID_FY, "failure_rate", "failure rate (tilt OR stance_loss)")
    print_grid_table(rows, GRID_MU, GRID_FY, "tilt_rate", "tilt rate")
    print_grid_table(rows, GRID_MU, GRID_FY, "stance_loss_rate", "stance_loss rate")
    print_grid_table(rows, GRID_MU, GRID_FY, "sag_rate", "sag rate z<0.20 (metric)")
    unique_rates = {round(float(r["failure_rate"]), 6) for r in rows}
    if unique_rates <= {0.0, 1.0}:
        print("  warning: every grid cell is still 0% or 100%")
    else:
        print("  grid cells are not all binary 0/100")
    write_json(
        GRID_JSON_PATH,
        {
            "seed": SEED,
            "created_at": meta["created_at"],
            "config_hash": meta["config_hash"],
            "config": meta["config"],
            "path_csv": str(GRID_PATH),
            "rows": rows,
        },
    )
    return rows


def run_cliff_grid(target_qpos: np.ndarray, kp: float, kd: float) -> list[dict]:
    rows, meta = run_force_mu_grid(
        target_qpos, kp, kd, CLIFF_MU, CLIFF_FY, N_GRID_EPISODES, "fine grid near the cliff"
    )
    write_csv(CLIFF_PATH, rows, GRID_CSV_FIELDS)
    print(f"fine grid saved: {CLIFF_PATH}")
    series = []
    for mu in CLIFF_MU:
        subset = [r for r in rows if abs(r["mu"] - mu) < 1e-12]
        subset.sort(key=lambda r: r["fy"])
        series.append(
            {
                "label": f"mu={mu:g} failure",
                "x": [r["fy"] for r in subset],
                "y": [r["failure_rate"] for r in subset],
                "y_lo": [r["failure_ci_lo"] for r in subset],
                "y_hi": [r["failure_ci_hi"] for r in subset],
            }
        )
        series.append(
            {
                "label": f"mu={mu:g} sag z<0.20",
                "x": [r["fy"] for r in subset],
                "y": [r["sag_rate"] for r in subset],
                "y_lo": [r["sag_ci_lo"] for r in subset],
                "y_hi": [r["sag_ci_hi"] for r in subset],
            }
        )
    write_line_svg(
        CLIFF_SVG_PATH,
        series,
        title="Failure rate vs Fy (Wilson 95%)",
        xlabel="Fy, N",
        ylabel="rate",
        y_min=0.0,
        y_max=1.0,
    )
    print(f"plot saved: {CLIFF_SVG_PATH}")
    write_json(
        Path("reports") / "cliff_grid.json",
        {
            "seed": SEED,
            "created_at": meta["created_at"],
            "config_hash": meta["config_hash"],
            "config": meta["config"],
            "path_csv": str(CLIFF_PATH),
            "path_svg": str(CLIFF_SVG_PATH),
            "rows": rows,
        },
    )
    return rows


def run_mu02_traces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
) -> dict:
    forces = (0.0, 100.0, 150.0, 200.0, 250.0)
    traces: list[dict] = []
    print("\ntraces at mu=0.2: z(t) and roll for Fy=0/100/150/200/250")
    for fy in forces:
        for episode in range(1, N_TRACE_EPISODES + 1):
            result = simulate_episode(
                model,
                data,
                target_qpos,
                friction_geoms,
                torso_body_id,
                0.2,
                float(fy),
                kp,
                kd,
                episode=episode,
                jitter=True,
                keep_window=False,
                record_trace=True,
            )
            dt = float(model.opt.timestep)
            traces.append(
                {
                    "episode": episode,
                    "seed": result["seed"],
                    "mu": 0.2,
                    "fy": float(fy),
                    "failed": result["failed"],
                    "fail_reason": result["fail_reason"],
                    "fail_step": result["fail_step"],
                    "sag_below_0_20": result["sag_below_0_20"],
                    "sag_step": result["sag_step"],
                    "min_z": result["min_z"],
                    "max_abs_roll": result["max_abs_roll"],
                    "max_abs_pitch": result["max_abs_pitch"],
                    "t": [dt * i for i in range(1, result["n_steps"] + 1)],
                    "z": result["z"],
                    "roll_deg": [math.degrees(v) for v in result["roll"]],
                }
            )
            print(
                f"  Fy={fy:5.0f} ep={episode}: "
                f"fail={result['failed']} reason={result['fail_reason']} "
                f"sag={result['sag_below_0_20']} "
                f"min_z={result['min_z']:.4f} "
                f"max|roll|={math.degrees(result['max_abs_roll']):.1f}°"
            )
    z_series = []
    roll_series = []
    for fy, color in (
        (0.0, "#7f8c8d"),
        (100.0, "#c0392b"),
        (250.0, "#1f4e79"),
    ):
        for item in traces:
            if abs(item["fy"] - fy) > 1e-12 or item["episode"] != 1:
                continue
            z_series.append(
                {
                    "label": f"Fy={fy:g} ep1",
                    "x": item["t"],
                    "y": item["z"],
                    "color": color,
                }
            )
            roll_series.append(
                {
                    "label": f"Fy={fy:g} ep1",
                    "x": item["t"],
                    "y": item["roll_deg"],
                    "color": color,
                }
            )
    write_line_svg(
        MU02_Z_SVG_PATH,
        z_series,
        title="mu=0.2: torso height z(t)",
        xlabel="t, s",
        ylabel="z, m",
    )
    write_line_svg(
        MU02_ROLL_SVG_PATH,
        roll_series,
        title="mu=0.2: roll(t)",
        xlabel="t, s",
        ylabel="roll, deg",
    )
    payload = {
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mu": 0.2,
        "forces": list(forces),
        "n_trace_episodes": N_TRACE_EPISODES,
        "path_z_svg": str(MU02_Z_SVG_PATH),
        "path_roll_svg": str(MU02_ROLL_SVG_PATH),
        "traces": traces,
    }
    write_json(MU02_TRACES_PATH, payload)
    return payload


def measure_stance_physics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    *,
    gravity_ff: bool = False,
    keyframe_ff: bool = False,
) -> dict:
    result = simulate_episode(
        model,
        data,
        target_qpos,
        friction_geoms,
        torso_body_id,
        1.0,
        0.0,
        kp,
        kd,
        episode=1,
        jitter=False,
        keep_window=False,
        record_trace=True,
        gravity_ff=gravity_ff,
        keyframe_ff=keyframe_ff,
    )
    reset_to_stance(model, data)
    mujoco.mj_forward(model, data)
    q_err = np.asarray(data.qpos[7:], dtype=float) - np.asarray(target_qpos, dtype=float)
    return {
        "z_start": result["z_start"],
        "z_end": result["z_end"],
        "min_z": result["min_z"],
        "max_abs_roll_deg": math.degrees(result["max_abs_roll"]),
        "failed": result["failed"],
        "sag_below_0_20": result["sag_below_0_20"],
        "qfrc_bias_abs_mean": float(np.mean(np.abs(data.qfrc_bias[6:]))),
        "q_err_abs_mean_start": float(np.mean(np.abs(q_err))),
        "total_mass": float(np.sum(model.body_mass)),
        "torso_mass": float(model.body_mass[torso_body_id]),
    }


def run_sag_compare(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
) -> list[dict]:
    variants = (
        {
            "id": "baseline",
            "label": "current PD kp=25 kd=1.5, no FF",
            "kp": 25.0,
            "kd": 1.5,
            "factor": None,
        },
        {
            "id": "stiff",
            "label": "raised kp=40 kd=3.0",
            "kp": 40.0,
            "kd": 3.0,
            "factor": None,
        },
        {
            "id": "gravity_ff",
            "label": "kp=25 kd=1.5 + feed-forward qfrc_bias",
            "kp": 25.0,
            "kd": 1.5,
            "factor": {"kind": "pd_variant", "ctrl": {"gravity_ff": True}},
        },
        {
            "id": "keyframe_ff",
            "label": "kp=25 kd=1.5 + keyframe-home ctrl as FF",
            "kp": 25.0,
            "kd": 1.5,
            "factor": {"kind": "pd_variant", "ctrl": {"keyframe_ff": True}},
        },
    )
    print("\nstance-variant comparison (default PD is unchanged):")
    rows: list[dict] = []
    for variant in variants:
        physics = measure_stance_physics(
            model,
            data,
            target_qpos,
            friction_geoms,
            torso_body_id,
            variant["kp"],
            variant["kd"],
            gravity_ff=bool((variant["factor"] or {}).get("ctrl", {}).get("gravity_ff")),
            keyframe_ff=bool((variant["factor"] or {}).get("ctrl", {}).get("keyframe_ff")),
        )
        print(
            f"  {variant['id']}: z {physics['z_start']:.4f}→{physics['z_end']:.4f} "
            f"min_z={physics['min_z']:.4f} |roll|={physics['max_abs_roll_deg']:.2f}° "
            f"|qfrc_bias|={physics['qfrc_bias_abs_mean']:.3f} N*m"
        )
        scenario_rates = {}
        for name in ("nominal", "moderate"):
            scenario = SCENARIOS[name]
            specs = generate_episode_specs(
                N_SAG_EPISODES,
                SEED,
                scenario["friction_range"],
                scenario["lateral_force_range"],
            )
            batch = run_stress_batch_parallel(
                specs,
                variant["kp"],
                variant["kd"],
                False,
                target_qpos,
                factor=variant["factor"],
            )
            print_batch_rates(f"    {name}: ", batch)
            scenario_rates[name] = batch_rate_summary(batch)
        row = {
            "id": variant["id"],
            "label": variant["label"],
            "kp": variant["kp"],
            "kd": variant["kd"],
            **physics,
            "nominal": scenario_rates["nominal"],
            "moderate": scenario_rates["moderate"],
        }
        rows.append(row)
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                "id": row["id"],
                "label": row["label"],
                "kp": row["kp"],
                "kd": row["kd"],
                "z_start": row["z_start"],
                "z_end": row["z_end"],
                "min_z": row["min_z"],
                "nominal_fail": row["nominal"]["failure_rate"],
                "nominal_sag": row["nominal"]["sag_rate"],
                "moderate_fail": row["moderate"]["failure_rate"],
                "moderate_sag": row["moderate"]["sag_rate"],
            }
        )
    write_csv(
        SAG_CSV_PATH,
        csv_rows,
        [
            "id",
            "label",
            "kp",
            "kd",
            "z_start",
            "z_end",
            "min_z",
            "nominal_fail",
            "nominal_sag",
            "moderate_fail",
            "moderate_sag",
        ],
    )
    write_json(
        SAG_JSON_PATH,
        {
            "seed": SEED,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "note": "Default PD is unchanged. The operator chooses the variant.",
            "rows": rows,
        },
    )
    return rows


def compare_ofat_sweeps(sweeps: list[dict]) -> None:
    if len(sweeps) < 2:
        return
    by_id = {item["scenario"]: item for item in sweeps}
    if "nominal" not in by_id or "moderate" not in by_id:
        return
    print("\nOFAT comparison: nominal (control) vs moderate (sensitivity)")
    nom_params = {p["id"]: p for p in by_id["nominal"]["parameters"]}
    for param in by_id["moderate"]["parameters"]:
        left = nom_params.get(param["id"])
        if left is None:
            continue
        nom_sig = left.get("significant_leader_label") or "none significant"
        mod_sig = param.get("significant_leader_label") or "none significant"
        print(
            f"  {param['name']}: nominal worst {left['worst_label']} "
            f"{left['worst_failure_rate'] * 100:.1f}% "
            f"(delta {left['worst_delta_vs_baseline'] * 100:+.1f}, "
            f"Holm: {nom_sig}); "
            f"moderate worst {param['worst_label']} "
            f"{param['worst_failure_rate'] * 100:.1f}% "
            f"(delta {param['worst_delta_vs_baseline'] * 100:+.1f}, "
            f"Holm: {mod_sig})"
        )


def load_existing_report() -> dict:
    if not REPORT_PATH.exists():
        return {}
    try:
        payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def save_report(report: dict) -> None:
    merged = load_existing_report()
    merged.update(report)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"results saved: {REPORT_PATH}")


def run_single_probes(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    lateral_force: float = COM_PROBE_FORCE,
) -> None:
    probe_friction_effect(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=kp, kd=kd, lateral_force=lateral_force
    )
    probe_com_offset(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=kp, kd=kd, lateral_force=lateral_force
    )
    probe_torque_limit(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=kp, kd=kd, lateral_force=lateral_force
    )
    probe_action_delay(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=kp, kd=kd, lateral_force=lateral_force
    )
    probe_joint_damping(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=kp, kd=kd, lateral_force=lateral_force
    )
    probe_body_inertia(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=kp, kd=kd, lateral_force=lateral_force
    )
    probe_floor_contact(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=kp, kd=kd, lateral_force=lateral_force
    )
    probe_observation_noise(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=kp, kd=kd, lateral_force=lateral_force
    )
    probe_sensor_bias(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=kp, kd=kd, lateral_force=lateral_force
    )


def run_pd_comparison(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    specs: list[tuple[int, float, float]],
    pd_name: str,
) -> dict:
    print("\nPD comparison (stance: mu=0.8, Fy=0; batch on the same episodes):")
    main_batch = None
    compare_names = ("baseline", "weak_pd")
    for name in compare_names:
        if name not in PD_MODES:
            continue
        gains = PD_MODES[name]
        kp = float(gains["kp"])
        kd = float(gains["kd"])
        stance = probe_stance_hold(
            model, data, target_qpos, friction_geoms, torso_body_id, kp=kp, kd=kd
        )
        keep_windows = name == pd_name
        batch = run_stress_batch_parallel(
            specs,
            kp,
            kd,
            keep_windows,
            target_qpos,
        )
        print(
            f"  {name}: kp={kp:g} kd={kd:g}  "
            f"z_end={stance['z_end']:.4f}  "
            f"failures={batch['failure_rate'] * 100:.1f}% "
            f"({batch['n_failures']}/{batch['n_episodes']})  "
            f"before step {IMPULSE_START}: {batch['n_before_impact']}  "
            f"at/after {IMPULSE_START}: {batch['n_at_or_after_impact']}  "
            f"time={batch['elapsed_s']:.2f}s  "
            f"workers={batch['n_workers']} chunks={batch['n_chunks']}"
        )
        if keep_windows:
            main_batch = batch
    if main_batch is None:
        raise RuntimeError("failed to run the main PD batch")
    return main_batch


def main_batch_heading(controller: str, pd_name: str, kp: float, kd: float) -> str:
    return (
        f"main run (controller {controller}, "
        f"PD {pd_name}, kp={kp:g}, kd={kd:g}):"
    )


def stance_reference_record(
    controller: str,
    pd_name: str,
    kp: float,
    kd: float,
    z_ref,
    frac,
) -> dict:
    z_lim = None
    if z_ref is not None and frac is not None:
        z_lim = float(frac) * float(z_ref)
    return {
        "controller": controller,
        "pd": pd_name,
        "kp": float(kp),
        "kd": float(kd),
        "z_ref": None if z_ref is None else float(z_ref),
        "stance_loss_frac": None if frac is None else float(frac),
        "stance_loss_z_limit": z_lim,
        "z_threshold": STANCE_THRESH.get("z_threshold"),
        "slip_threshold": STANCE_THRESH.get("slip_threshold"),
    }


def print_main_batch_summary(
    pd_name: str,
    batch: dict,
    controller: str | None = None,
    stance: dict | None = None,
) -> None:
    failed_friction_stats = value_stats(batch["failed_friction"])
    failed_force_stats = value_stats(batch["failed_force"])
    survived_friction_stats = value_stats(batch["survived_friction"])
    survived_force_stats = value_stats(batch["survived_force"])
    ctrl = controller or ACTIVE_CONTROLLER
    print(f"\n{main_batch_heading(ctrl, pd_name, batch['kp'], batch['kd'])}")
    if stance:
        z_ref = stance.get("z_ref")
        z_thr = stance.get("z_threshold")
        slip = stance.get("slip_threshold")
        z_txt = f"{float(z_ref):.4f} m" if z_ref is not None else "n/a"
        thr_txt = f"z<{float(z_thr):.4f} m" if z_thr is not None else "z n/a"
        slip_txt = f"slip>{float(slip):.4f} m" if slip is not None else "slip n/a"
        print(f"reference z_ref={z_txt}  stance_loss threshold: {thr_txt}  {slip_txt}")
    print(f"episodes: {batch['n_episodes']}  seed={SEED}")
    print(
        f"parallel: {batch['n_workers']} workers, "
        f"{batch['n_chunks']} chunks, time={batch['elapsed_s']:.2f}s"
    )
    print(f"failed (current criterion): {batch['n_failures']}")
    print(f"survived: {batch['n_survived']}")
    print_batch_rates("  ", batch)
    print(f"failed, friction: {format_stats(failed_friction_stats)}")
    print(f"failed, lateral force: {format_stats(failed_force_stats)}")
    print(f"survived, friction: {format_stats(survived_friction_stats)}")
    print(f"survived, lateral force: {format_stats(survived_force_stats)}")


def report_from_batch(
    pd_name: str,
    run_mode: str,
    batch: dict,
    model: mujoco.MjModel,
    floor_id: int,
    friction_geoms: list[int],
    torso_body_id: int,
    target_qpos: np.ndarray,
) -> dict:
    return {
        "n_episodes": batch["n_episodes"],
        "n_envs": N_ENVS,
        "n_failures": batch["n_failures"],
        "n_survived": batch["n_survived"],
        "failure_rate": batch["failure_rate"],
        "failure_rate_ci95": list(batch.get("failure_rate_ci95") or wilson_ci(
            int(batch["n_failures"]), int(batch["n_episodes"])
        )),
        "n_sag": int(batch.get("n_sag") or 0),
        "sag_rate": float(batch.get("sag_rate") or 0.0),
        "sag_rate_ci95": list(batch.get("sag_rate_ci95") or wilson_ci(
            int(batch.get("n_sag") or 0), int(batch["n_episodes"])
        )),
        "fail_reasons": dict(batch.get("fail_reasons") or {}),
        "n_tilt": int(batch.get("n_tilt") or 0),
        "tilt_rate": float(batch.get("tilt_rate") or 0.0),
        "n_stance_loss": int(batch.get("n_stance_loss") or 0),
        "stance_loss_rate": float(batch.get("stance_loss_rate") or 0.0),
        "episodes": batch.get("episodes") or [],
        "run_mode": run_mode,
        "mode": pd_name,
        "pd": pd_name,
        "seed": SEED,
        "elapsed_s": batch.get("elapsed_s"),
        "n_workers": batch.get("n_workers"),
        "n_chunks": batch.get("n_chunks"),
        "kp": batch["kp"],
        "kd": batch["kd"],
        "floor_geom_id": int(floor_id),
        "floor_geom_name": geom_name(model, floor_id),
        "floor_geom_type": geom_type_name(model, floor_id),
        "geom_friction_shape": list(model.geom_friction.shape),
        "friction_geom_ids": friction_geoms,
        "torso_body_id": int(torso_body_id),
        "impulse_start": IMPULSE_START,
        "impulse_end": IMPULSE_END,
        "target_qpos": target_qpos.tolist(),
        "failed": {
            "friction": value_stats(batch["failed_friction"]),
            "lateral_force": value_stats(batch["failed_force"]),
        },
        "survived": {
            "friction": value_stats(batch["survived_friction"]),
            "lateral_force": value_stats(batch["survived_force"]),
        },
        "failures": batch["failed_episodes"],
    }


def hold_stance_for_ff(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    n_steps: int,
    ff_mode: str,
) -> dict:
    reset_to_stance(model, data)
    set_sliding_friction(model, friction_geoms, 1.0)
    mujoco.mj_forward(model, data)
    start_z = float(data.qpos[2])
    start_bias = np.array(data.qfrc_bias, copy=True)
    start_qerr = np.asarray(data.qpos[7:], dtype=float) - np.asarray(target_qpos)
    for _ in range(n_steps):
        apply_pd(model, data, target_qpos, kp=kp, kd=kd, gravity_ff=False)
        if ff_mode == "legacy":
            add_legacy_gravity_ff(model, data)
        elif ff_mode == "mapped":
            add_mapped_gravity_ff(model, data)
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    rows = actuator_joint_rows(model)
    joints = []
    for row in rows:
        dof = int(row["dof"])
        tau_pd = float(data.qfrc_actuator[dof])
        tau_bias = float(data.qfrc_bias[dof])
        joints.append(
            {
                **row,
                "tau_actuator": tau_pd,
                "tau_bias": tau_bias,
                "rel_err": (
                    abs(tau_pd - tau_bias) / max(abs(tau_bias), 1e-9)
                    if abs(tau_bias) > 1e-6
                    else abs(tau_pd - tau_bias)
                ),
            }
        )
    qerr = np.asarray(data.qpos[7:], dtype=float) - np.asarray(target_qpos)
    return {
        "ff_mode": ff_mode,
        "z_start": start_z,
        "z_end": float(data.qpos[2]),
        "qerr_abs_mean_start": float(np.mean(np.abs(start_qerr))),
        "qerr_abs_mean": float(np.mean(np.abs(qerr))),
        "bias_abs_mean_start_legs": float(np.mean(np.abs(start_bias[6:]))),
        "bias_abs_max_start_legs": float(np.max(np.abs(start_bias[6:]))),
        "bias_abs_mean_settled_legs": float(np.mean(np.abs(data.qfrc_bias[6:]))),
        "bias_abs_max_settled_legs": float(np.max(np.abs(data.qfrc_bias[6:]))),
        "bias_free_z_start": float(start_bias[2]),
        "bias_free_z_settled": float(data.qfrc_bias[2]),
        "joints": joints,
    }


def run_gravity_ff_diagnosis(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
) -> dict:
    n_steps = int(round(1.5 / max(float(model.opt.timestep), 1e-9)))
    print("\ngravity feed-forward diagnosis")
    print("actuator -> joint -> dof:")
    mapping = actuator_joint_rows(model)
    print(
        f"{'act':>3} {'name':<14} {'joint':<16} {'jid':>3} {'dof':>3} {'gear':>6}"
    )
    for row in mapping:
        print(
            f"{row['actuator']:3d} {row['actuator_name']:<14} "
            f"{row['joint_name']:<16} {row['joint']:3d} {row['dof']:3d} "
            f"{row['gear']:6.3f}"
        )
    snapshots = []
    for mode in ("none", "legacy", "mapped"):
        snap = hold_stance_for_ff(
            model,
            data,
            target_qpos,
            friction_geoms,
            torso_body_id,
            kp,
            kd,
            n_steps,
            mode,
        )
        snapshots.append(snap)
        print(
            f"\n{mode}: z {snap['z_start']:.4f}→{snap['z_end']:.4f}  "
            f"|qerr|={snap['qerr_abs_mean']:.4f} rad  "
            f"|bias_legs| start mean={snap['bias_abs_mean_start_legs']:.3f} "
            f"max={snap['bias_abs_max_start_legs']:.3f}  "
            f"settled mean={snap['bias_abs_mean_settled_legs']:.3f} "
            f"max={snap['bias_abs_max_settled_legs']:.3f}  "
            f"qfrc_bias free-z start={snap['bias_free_z_start']:.2f} "
            f"settled={snap['bias_free_z_settled']:.2f}"
        )
        if mode == "none":
            print(
                f"{'joint':<16} {'dof':>3} {'tau_act':>10} {'tau_bias':>10} {'rel.error':>10}"
            )
            for j in snap["joints"]:
                print(
                    f"{j['joint_name']:<16} {j['dof']:3d} "
                    f"{j['tau_actuator']:10.4f} {j['tau_bias']:10.4f} "
                    f"{j['rel_err']*100:9.1f}%"
                )
    none, legacy, mapped = snapshots
    lift = mapped["z_end"] - none["z_end"]
    print(
        f"\nsettled height: none={none['z_end']:.4f}  "
        f"legacy={legacy['z_end']:.4f}  mapped={mapped['z_end']:.4f}  "
        f"delta mapped-none={lift:+.4f} m"
    )
    if lift > 0.005:
        print(
            "Mapped actuator->dof FF lifts the stance. "
            "Proposal: make PD+gravity_ff the default controller. "
            "Left off until that is decided."
        )
    else:
        print("mapped FF did not lift the stance by more than 5 mm. Baseline PD stays without FF.")
    payload = {
        "kp": kp,
        "kd": kd,
        "n_steps": n_steps,
        "mapping": mapping,
        "snapshots": [
            {k: v for k, v in snap.items() if k != "joints"} | {"joints": snap["joints"]}
            for snap in snapshots
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
    }
    write_json(Path("reports") / "gravity_ff_diagnosis.json", payload)
    return payload


def compact_episode(ep: dict) -> dict:
    keep = (
        "episode",
        "seed",
        "friction",
        "lateral_force",
        "failed",
        "fail_reason",
        "fail_fall",
        "fail_tilt",
        "fail_stance_loss",
        "z_settle",
        "z_after_settle",
        "min_z",
        "foot_slip_rel",
        "foot_slip_max",
        "max_abs_roll",
        "max_abs_pitch",
        "max_body_speed",
        "max_tau_over_limit",
        "saturation_frac",
        "hit_limit",
        "fall_kind",
        "impulse_start",
        "impulse_end",
    )
    return {k: ep.get(k) for k in keep}


def torque_episode(ep: dict) -> dict:
    sat = ep.get("saturation_frac")
    tau = ep.get("max_tau_over_limit")
    return {
        "episode": ep.get("episode"),
        "seed": ep.get("seed"),
        "friction": ep.get("friction"),
        "lateral_force": ep.get("lateral_force"),
        "fail": bool(ep.get("failed")),
        "fall": bool(ep.get("fail_fall") or ep.get("fail_tilt")),
        "stance_loss": bool(ep.get("fail_stance_loss")),
        "fail_reason": ep.get("fail_reason"),
        "fall_kind": ep.get("fall_kind"),
        "z_settle": ep.get("z_settle", ep.get("z_after_settle")),
        "min_z": ep.get("min_z"),
        "z_end": ep.get("z_end"),
        "max_tau_over_limit": None if tau is None else float(tau),
        "saturation_frac": None if sat is None else float(sat),
        "limit_reached": bool(sat is not None and float(sat) > 1e-12),
        "limit_note": saturation_note(None if sat is None else float(sat)),
        "hit_limit": bool(ep.get("hit_limit")),
    }


def criteria_episode(ep: dict) -> dict:
    slip = ep.get("foot_slip_rel", ep.get("foot_slip_max"))
    return {
        "episode": ep.get("episode"),
        "seed": ep.get("seed"),
        "friction": ep.get("friction"),
        "lateral_force": ep.get("lateral_force"),
        "fail": bool(ep.get("failed")),
        "fall": bool(ep.get("fail_fall") or ep.get("fail_tilt")),
        "stance_loss": bool(ep.get("fail_stance_loss")),
        "z_settle": ep.get("z_settle", ep.get("z_after_settle")),
        "slip": None if slip is None else float(slip),
        "min_z": ep.get("min_z"),
        "max_abs_roll": ep.get("max_abs_roll"),
        "max_abs_pitch": ep.get("max_abs_pitch"),
        "max_body_speed": ep.get("max_body_speed"),
    }


def fit_outcome_thresholds(episodes: list[dict], z_ref: float) -> dict:
    standing = [
        ep
        for ep in episodes
        if not ep.get("fail_fall") and not ep.get("fail_tilt")
    ]
    zs = np.array(
        [
            float(ep.get("z_settle", ep.get("z_after_settle")))
            for ep in standing
            if ep.get("z_settle", ep.get("z_after_settle")) is not None
        ],
        dtype=float,
    )
    slips = np.array(
        [
            float(ep.get("foot_slip_rel", ep.get("foot_slip_max") or 0.0))
            for ep in standing
        ],
        dtype=float,
    )
    if zs.size == 0:
        raise RuntimeError("no standing nominal episodes for the stance_loss threshold")
    z_std = float(zs.std(ddof=1)) if zs.size > 1 else 0.0
    slip_std = float(slips.std(ddof=1)) if slips.size > 1 else 0.0
    z_min = float(zs.min())
    slip_max = float(slips.max())
    derivation = {
        "n_nominal": int(len(episodes)),
        "n_standing": int(zs.size),
        "z_ref": float(z_ref),
        "z_settle_mean": float(zs.mean()),
        "z_settle_min": z_min,
        "z_settle_max": float(zs.max()),
        "z_settle_std": z_std,
        "z_threshold_rule": "min(nominal) - 3σ",
        "z_threshold": z_min - 3.0 * z_std,
        "slip_mean": float(slips.mean()),
        "slip_min": float(slips.min()),
        "slip_max": slip_max,
        "slip_std": slip_std,
        "slip_threshold_rule": "max(nominal) + 3σ",
        "slip_threshold": slip_max + 3.0 * slip_std,
        "failure_criterion": {
            "fail": "fall OR stance_loss",
            "fall": "abs(roll)>45° or abs(pitch)>45° or body z<0.12 m",
            "stance_loss": "z_settle < z_threshold OR slip > slip_threshold",
            "slip": "max foot xy displacement relative to body",
            "measure": f"{POST_IMPACT_MEASURE_S:.2f}s after impulse_end",
        },
        "post_impact_measure_s": POST_IMPACT_MEASURE_S,
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    STANCE_THRESH.update(
        {
            "z_threshold": derivation["z_threshold"],
            "slip_threshold": derivation["slip_threshold"],
            "z_ref": float(z_ref),
            "n": int(zs.size),
            "derivation": derivation,
        }
    )
    STANCE_REF["z"] = float(z_ref)
    return derivation


def relabel_outcomes(episodes: list[dict]) -> None:
    z_thr = STANCE_THRESH.get("z_threshold")
    slip_thr = STANCE_THRESH.get("slip_threshold")
    for ep in episodes:
        fall = bool(ep.get("fail_fall") or ep.get("fail_reason") in ("fall", "tilt"))
        z_s = ep.get("z_settle", ep.get("z_after_settle"))
        slip = ep.get("foot_slip_rel", ep.get("foot_slip_max"))
        stance_loss = False
        if not fall:
            z_bad = (
                z_thr is not None
                and z_s is not None
                and float(z_s) < float(z_thr)
            )
            slip_bad = (
                slip_thr is not None
                and slip is not None
                and float(slip) > float(slip_thr)
            )
            if z_bad or slip_bad:
                stance_loss = True
        ep["fail_fall"] = fall
        ep["fail_tilt"] = fall
        ep["fail_stance_loss"] = stance_loss
        ep["failed"] = fall or stance_loss
        if fall:
            ep["fail_reason"] = "fall"
        elif stance_loss:
            ep["fail_reason"] = "stance_loss"
        else:
            ep["fail_reason"] = None


def outcome_counts(episodes: list[dict]) -> dict:
    n = len(episodes)
    n_fall = sum(1 for ep in episodes if ep.get("fail_fall"))
    n_sl = sum(1 for ep in episodes if ep.get("fail_stance_loss"))
    n_fail = sum(1 for ep in episodes if ep.get("failed"))
    return {
        "n": n,
        "n_fail": n_fail,
        "fail_rate": n_fail / n if n else 0.0,
        "n_fall": n_fall,
        "fall_rate": n_fall / n if n else 0.0,
        "n_stance_loss": n_sl,
        "stance_loss_rate": n_sl / n if n else 0.0,
        "fail_rate_ci95": list(wilson_ci(n_fail, n)),
        "fall_rate_ci95": list(wilson_ci(n_fall, n)),
        "stance_loss_rate_ci95": list(wilson_ci(n_sl, n)),
        "episodes": [criteria_episode(ep) for ep in episodes],
    }


def print_criteria_rates(prefix: str, counts: dict) -> None:
    n = int(counts["n"])
    lo, hi = counts.get("fail_rate_ci95") or wilson_ci(int(counts["n_fail"]), n)
    print(
        f"{prefix}fail={counts['n_fail']}/{n}={counts['fail_rate'] * 100:.1f}% "
        f"CI95[{lo * 100:.1f},{hi * 100:.1f}]  "
        f"fall={counts['n_fall']}/{n}={counts['fall_rate'] * 100:.1f}%  "
        f"stance_loss={counts['n_stance_loss']}/{n}={counts['stance_loss_rate'] * 100:.1f}%"
    )


def frac_fail_rate(episodes: list[dict], frac: float, z_ref: float) -> dict:
    z_lim = float(frac) * float(z_ref)
    n = 0
    n_fail = 0
    n_z = 0
    for ep in episodes:
        fall = bool(ep.get("fail_fall"))
        z_s = ep.get("z_settle", ep.get("z_after_settle"))
        z_hit = z_s is not None and float(z_s) < z_lim
        n += 1
        if z_hit:
            n_z += 1
        if fall or z_hit:
            n_fail += 1
    return {
        "frac": frac,
        "z_lim": z_lim,
        "n": n,
        "fail_rate": n_fail / n if n else 0.0,
        "z_hit_rate": n_z / n if n else 0.0,
        "fail_rate_ci95": list(wilson_ci(n_fail, n)),
    }


def run_named_batch(
    target_qpos: np.ndarray,
    kp: float,
    kd: float,
    specs: list[tuple[int, float, float]],
    factor: dict | None = None,
) -> dict:
    return run_stress_batch_parallel(specs, kp, kd, False, target_qpos, factor=factor)


def run_criteria_analysis(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
) -> dict:
    z_ref = float(STANCE_REF["z"] or 0.0)
    print("\nfailure criterion: fail = fall OR stance_loss")
    print("  fall: |roll| or |pitch| > 45 deg, or torso contact with the floor (z < 0.12 m)")
    print("  stance_loss: z_settle or slip (feet relative to the torso) beyond the nominal thresholds")
    print("  degradation is unused: -3% of z_ref sits inside the stance_loss threshold")
    print(f"\nstance_loss calibration: {N_THRESHOLD_EPISODES} nominal episodes")
    STANCE_THRESH["z_threshold"] = None
    STANCE_THRESH["slip_threshold"] = None
    specs = generate_episode_specs(
        N_THRESHOLD_EPISODES,
        SEED,
        SCENARIOS["nominal"]["friction_range"],
        SCENARIOS["nominal"]["lateral_force_range"],
    )
    nom = run_named_batch(target_qpos, kp, kd, specs)
    derivation = fit_outcome_thresholds(nom["episodes"], z_ref)
    relabel_outcomes(nom["episodes"])
    nom_counts = outcome_counts(nom["episodes"])
    print(
        f"  z_settle min={derivation['z_settle_min']:.4f} σ={derivation['z_settle_std']:.4f} "
        f"threshold={derivation['z_threshold']:.4f}"
    )
    print(
        f"  slip max={derivation['slip_max']:.4f} σ={derivation['slip_std']:.4f} "
        f"threshold={derivation['slip_threshold']:.4f}"
    )
    print_criteria_rates("  nominal after the threshold: ", nom_counts)
    write_json(THRESHOLDS_PATH, derivation)

    cells = {"nominal": nom}
    ice_specs = {}
    for mu in (0.1, 0.2):
        for fy in (0.0, 100.0, 200.0):
            key = f"ice_mu{mu}_fy{fy:.0f}"
            specs_i = [(ep, mu, fy) for ep in range(1, 101)]
            batch = run_named_batch(target_qpos, kp, kd, specs_i)
            relabel_outcomes(batch["episodes"])
            cells[key] = batch
            c = outcome_counts(batch["episodes"])
            print_criteria_rates(f"  {key}: ", c)
            ice_specs[key] = c
    bias_specs = generate_episode_specs(
        100,
        SEED,
        SCENARIOS["nominal"]["friction_range"],
        SCENARIOS["nominal"]["lateral_force_range"],
    )
    bias = run_named_batch(
        target_qpos,
        kp,
        kd,
        bias_specs,
        factor={"kind": "bias", "q": 0.05, "drift": 0.0},
    )
    relabel_outcomes(bias["episodes"])
    cells["bias_0.05"] = bias
    bias_counts = outcome_counts(bias["episodes"])
    print_criteria_rates("  bias+0.05: ", bias_counts)

    fracs = [round(x, 2) for x in np.arange(0.85, 0.991, 0.02)]
    sensitivity = []
    series = []
    names = ["nominal", "ice_mu0.1_fy0", "ice_mu0.1_fy100", "ice_mu0.1_fy200",
             "ice_mu0.2_fy0", "ice_mu0.2_fy100", "ice_mu0.2_fy200", "bias_0.05"]
    print("\nz-threshold diagnosis (not the failure criterion): fall OR z_settle < frac*z_ref")
    header = "frac  " + "  ".join(f"{n:>16}" for n in names)
    print(header)
    for frac in fracs:
        row = {"frac": frac}
        for name in names:
            stats = frac_fail_rate(cells[name]["episodes"], frac, z_ref)
            row[name] = stats["fail_rate"]
            row[f"{name}_ci"] = stats["fail_rate_ci95"]
        sensitivity.append(row)
        print(
            f"{frac:4.2f}  "
            + "  ".join(f"{row[n]*100:15.1f}%" for n in names)
        )
    ice_names = [n for n in names if n.startswith("ice_")]
    plateau = None
    for row in sensitivity:
        if row["nominal"] <= 1e-12 and all(row[k] <= 1e-12 for k in ice_names):
            plateau = row["frac"]
    last_nominal_zero = None
    for row in sensitivity:
        if row["nominal"] <= 1e-12:
            last_nominal_zero = row["frac"]
    if plateau is not None:
        print(
            f"z-criterion plateau: nominal=0 and ice=0 up to frac={plateau:.2f}; "
            f"nominal=0 up to frac={last_nominal_zero:.2f}"
        )
    for name, color in (
        ("nominal", "#1f4e79"),
        ("ice_mu0.2_fy0", "#c0392b"),
        ("ice_mu0.2_fy200", "#8e44ad"),
        ("bias_0.05", "#1e8449"),
    ):
        series.append(
            {
                "label": name,
                "x": [r["frac"] for r in sensitivity],
                "y": [r[name] for r in sensitivity],
                "y_lo": [r[f"{name}_ci"][0] for r in sensitivity],
                "y_hi": [r[f"{name}_ci"][1] for r in sensitivity],
                "color": color,
            }
        )
    write_line_svg(
        CRITERIA_SVG_PATH,
        series,
        title="Failure rate vs frac*z_ref (Wilson 95%)",
        xlabel="frac",
        ylabel="failure rate",
        y_min=0.0,
        y_max=1.0,
    )
    traces = run_ice_nonmonotonic_traces(
        model, data, target_qpos, friction_geoms, torso_body_id, kp, kd
    )
    payload = {
        "failure_criterion": {
            "fail": "fall OR stance_loss",
            "fall": "abs(roll)>45° or abs(pitch)>45° or body z<0.12 m",
            "stance_loss": "z_settle < z_threshold OR slip > slip_threshold",
            "slip": "max foot xy displacement relative to body",
            "measure": f"{POST_IMPACT_MEASURE_S:.2f}s after impulse_end",
        },
        "degradation": {
            "used": False,
            "reason": (
                "the -3% of z_ref threshold sits below z_threshold (min-3 sigma), "
                "so the event is inside stance_loss and is not a separate fail"
            ),
        },
        "z_ref": z_ref,
        "thresholds": derivation,
        "nominal": nom_counts,
        "ice": ice_specs,
        "bias_0.05": bias_counts,
        "sensitivity": sensitivity,
        "plateau_nominal_zero_and_ice_zero_frac": plateau,
        "last_nominal_zero_frac": last_nominal_zero,
        "sensitivity_svg": str(CRITERIA_SVG_PATH),
        "ice_traces": traces,
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "kp": kp,
        "kd": kd,
        "post_impact_measure_s": POST_IMPACT_MEASURE_S,
    }
    write_json(Path("reports") / "criteria_analysis.json", payload)
    return payload


def run_ice_nonmonotonic_traces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
) -> dict:
    foot_ids = find_foot_geom_ids(model)
    forces = (150.0, 250.0, 350.0)
    print("\nmu=0.2 diagnosis: z, roll, feet/torso, speed at Fy=150/250/350")
    traces = []
    for fy in forces:
        for ep in range(1, 4):
            result = simulate_episode(
                model,
                data,
                target_qpos,
                friction_geoms,
                torso_body_id,
                0.2,
                float(fy),
                kp,
                kd,
                episode=ep,
                jitter=True,
                record_trace=True,
                foot_ids=foot_ids,
            )
            dt = float(model.opt.timestep)
            t = [dt * i for i in range(1, result["n_steps"] + 1)]
            rel = np.asarray(result["foot_rel"], dtype=float) if result["foot_rel"] else np.zeros((0, 8))
            spread = []
            if rel.size:
                for row in rel:
                    d = [
                        math.hypot(row[i], row[i + 1])
                        for i in range(0, min(8, row.size), 2)
                    ]
                    spread.append(float(np.mean(d)) if d else 0.0)
            traces.append(
                {
                    **criteria_episode(result),
                    "fy": fy,
                    "t": t,
                    "z": result["z"],
                    "roll_deg": [math.degrees(v) for v in result["roll"]],
                    "speed": result["speed"],
                    "foot_spread": spread,
                }
            )
            print(
                f"  Fy={fy:.0f} ep={ep}: fall={result.get('fail_fall')} "
                f"stance_loss={result.get('fail_stance_loss')} "
                f"z_settle={result.get('z_settle')} "
                f"slip_rel={result.get('foot_slip_rel')} "
                f"min_z={result['min_z']:.4f}"
            )
    colors = {150.0: "#1f4e79", 250.0: "#c0392b", 350.0: "#1e8449"}
    def series_for(key, ylabel, path, scale=None):
        ser = []
        for fy in forces:
            item = next(tr for tr in traces if tr["fy"] == fy and tr["episode"] == 1)
            ys = item[key]
            if scale:
                ys = [scale(v) for v in ys]
            ser.append(
                {
                    "label": f"Fy={fy:g} ep1",
                    "x": item["t"],
                    "y": ys,
                    "color": colors[fy],
                }
            )
        write_line_svg(path, ser, title=ylabel, xlabel="t, s", ylabel=ylabel)
        return str(path)

    paths = {
        "z": series_for("z", "mu=0.2 z(t)", ICE_TRACE_DIR / "ice_mu02_z.svg"),
        "roll": series_for("roll_deg", "mu=0.2 roll, deg", ICE_TRACE_DIR / "ice_mu02_roll.svg"),
        "speed": series_for("speed", "mu=0.2 torso speed, m/s", ICE_TRACE_DIR / "ice_mu02_speed.svg"),
        "spread": series_for(
            "foot_spread",
            "mu=0.2 mean |foot-torso| xy, m",
            ICE_TRACE_DIR / "ice_mu02_foot_rel.svg",
        ),
    }
    compact_traces = []
    for tr in traces:
        row = {k: v for k, v in tr.items() if k not in ("t", "z", "roll_deg", "speed", "foot_spread")}
        compact_traces.append(row)
    return {"episodes": compact_traces, "paths": paths}


def evaluate_fall_flags(
    target_qpos: np.ndarray,
    kp: float,
    kd: float,
    mu: float,
    fy_mag: float,
    n_episodes: int,
    factor: dict | None = None,
) -> list[int]:
    specs = [(ep, float(mu), float(fy_mag)) for ep in range(1, n_episodes + 1)]
    batch = run_named_batch(target_qpos, kp, kd, specs, factor=factor)
    by_ep = {int(ep["episode"]): ep for ep in batch["episodes"]}
    flags = []
    for ep in range(1, n_episodes + 1):
        row = by_ep.get(ep, {})
        flags.append(1 if row.get("fail_fall") or row.get("fail_tilt") else 0)
    return flags


def interpolate_crossing(points: list[tuple[float, float]], p_target: float) -> float:
    pts = sorted(points, key=lambda p: p[0])
    if not pts:
        return float("nan")
    if pts[0][1] >= p_target:
        return pts[0][0]
    if pts[-1][1] < p_target:
        return pts[-1][0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 <= p_target <= y1 or y1 <= p_target <= y0:
            if abs(y1 - y0) < 1e-12:
                return 0.5 * (x0 + x1)
            t = (p_target - y0) / (y1 - y0)
            return x0 + t * (x1 - x0)
    return pts[-1][0]


def fy_crit_search(
    target_qpos: np.ndarray,
    kp: float,
    kd: float,
    *,
    mu: float = 0.8,
    p_target: float = 0.5,
    n_episodes: int = FY_CRIT_N_EPISODES,
    n_iter: int = FY_CRIT_ITERS,
    factor: dict | None = None,
) -> dict:
    lo, hi = FY_CRIT_LO, FY_CRIT_HI
    probes: list[dict] = []
    print(f"  Fy_crit p={p_target:.2f} μ={mu:g} n={n_episodes} iter={n_iter}")
    for i in range(n_iter):
        mid = 0.5 * (lo + hi)
        flags = evaluate_fall_flags(
            target_qpos, kp, kd, mu, mid, n_episodes, factor=factor
        )
        p = float(np.mean(flags))
        probes.append({"fy": mid, "p": p, "flags": flags})
        print(f"    iter {i+1}: |Fy|={mid:.2f} N  p_fall={p*100:.1f}%")
        if p < p_target:
            lo = mid
        else:
            hi = mid
    points = [(float(pr["fy"]), float(pr["p"])) for pr in probes]
    fy_hat = interpolate_crossing(points, p_target)
    rng = np.random.default_rng(SEED + 17)
    n = n_episodes
    boot = []
    flag_mat = np.array([pr["flags"] for pr in probes], dtype=float)
    fys = np.array([pr["fy"] for pr in probes], dtype=float)
    for _ in range(FY_CRIT_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_pts = [
            (float(fys[j]), float(flag_mat[j, idx].mean()))
            for j in range(len(probes))
        ]
        boot.append(interpolate_crossing(boot_pts, p_target))
    boot_arr = np.asarray(boot, dtype=float)
    ci = (float(np.percentile(boot_arr, 2.5)), float(np.percentile(boot_arr, 97.5)))
    return {
        "p_target": p_target,
        "mu": mu,
        "fy_crit": fy_hat,
        "ci95": list(ci),
        "n_episodes": n_episodes,
        "n_iter": n_iter,
        "n_bootstrap": FY_CRIT_BOOTSTRAP,
        "probes": [{"fy": pr["fy"], "p": pr["p"]} for pr in probes],
        "impulse_realistic": bool(IMPULSE_CFG.get("realistic")),
    }


def run_fycrit_ofat(
    target_qpos: np.ndarray,
    kp: float,
    kd: float,
) -> dict:
    print("\ntransition width before a realistic impulse (force along Y)")
    IMPULSE_CFG["realistic"] = False
    before_10 = fy_crit_search(target_qpos, kp, kd, p_target=0.1)
    before_50 = fy_crit_search(target_qpos, kp, kd, p_target=0.5)
    before_90 = fy_crit_search(target_qpos, kp, kd, p_target=0.9)
    width_before = before_90["fy_crit"] - before_10["fy_crit"]
    print(
        f"  before: Fy10={before_10['fy_crit']:.1f} Fy50={before_50['fy_crit']:.1f} "
        f"Fy90={before_90['fy_crit']:.1f} width={width_before:.1f} N"
    )
    print("\nrealistic impulse: random direction, height +/-3 cm, duration +/-20%")
    IMPULSE_CFG["realistic"] = True
    after_10 = fy_crit_search(target_qpos, kp, kd, p_target=0.1)
    after_50 = fy_crit_search(target_qpos, kp, kd, p_target=0.5)
    after_90 = fy_crit_search(target_qpos, kp, kd, p_target=0.9)
    width_after = after_90["fy_crit"] - after_10["fy_crit"]
    print(
        f"  after: Fy10={after_10['fy_crit']:.1f} Fy50={after_50['fy_crit']:.1f} "
        f"Fy90={after_90['fy_crit']:.1f} width={width_after:.1f} N"
    )
    base = after_50
    print("\nOFAT Fy_crit, baseline mu=0.8, realistic impulse")
    parameters = []
    no_effect = []
    for spec in PARAMETER_SWEEPS:
        if spec["id"] == "torque_limit":
            print("  torque_limit: skipped (left unchanged on this step)")
            continue
        print(f"  {spec['name']}:")
        levels_out = []
        for level in spec["levels"]:
            est = fy_crit_search(
                target_qpos, kp, kd, factor=level["factor"]
            )
            delta = est["fy_crit"] - base["fy_crit"]
            pct = 100.0 * delta / base["fy_crit"] if abs(base["fy_crit"]) > 1e-9 else 0.0
            overlap = not (
                est["ci95"][1] < base["ci95"][0] or est["ci95"][0] > base["ci95"][1]
            )
            significant = not overlap
            row = {
                "id": level["id"],
                "label": level["label"],
                "fy_crit": est["fy_crit"],
                "ci95": est["ci95"],
                "delta_n": delta,
                "delta_pct": pct,
                "significant": significant,
                "ci_overlap_with_base": overlap,
            }
            levels_out.append(row)
            mark = "significant" if significant else "within noise"
            print(
                f"    {level['label']}: Fy_crit={est['fy_crit']:.1f} N "
                f"CI[{est['ci95'][0]:.1f},{est['ci95'][1]:.1f}] "
                f"delta={delta:+.1f} N ({pct:+.1f}%) {mark}"
            )
        baseline_level = next(
            (lv for lv in levels_out if lv["id"] == spec["baseline"]), levels_out[0]
        )
        if not any(lv["significant"] and lv["id"] != baseline_level["id"] for lv in levels_out):
            no_effect.append(spec["id"])
        parameters.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "baseline": spec["baseline"],
                "levels": levels_out,
            }
        )
    payload = {
        "mu": 0.8,
        "kp": kp,
        "kd": kd,
        "seed": SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "impulse": dict(IMPULSE_CFG),
        "transition_before": {
            "fy10": before_10,
            "fy50": before_50,
            "fy90": before_90,
            "width_n": width_before,
        },
        "transition_after": {
            "fy10": after_10,
            "fy50": after_50,
            "fy90": after_90,
            "width_n": width_after,
        },
        "base_fy_crit": base,
        "parameters": parameters,
        "no_significant_effect": no_effect,
    }
    write_json(FYCRIT_PATH, payload)
    print("no significant effect:", ", ".join(no_effect) or "none")
    IMPULSE_CFG["realistic"] = False
    return payload


def compute_contact_ff(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    friction_geoms: list[int],
) -> np.ndarray:
    reset_to_stance(model, data)
    set_sliding_friction(model, friction_geoms, 1.0)
    mujoco.mj_forward(model, data)
    nu = int(model.nu)
    return np.asarray(data.qfrc_bias[6 : 6 + nu] - data.qfrc_constraint[6 : 6 + nu], dtype=float)


def run_pd_ff_variant(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
) -> dict:
    tau = compute_contact_ff(model, data, friction_geoms)
    print("\npd_ff variant: tau_ff = qfrc_bias - qfrc_constraint (leg dofs), baseline unchanged")
    print(f"  |tau_ff| mean={float(np.mean(np.abs(tau))):.3f} max={float(np.max(np.abs(tau))):.3f} N*m")
    hold = hold_stance_for_ff(
        model, data, target_qpos, friction_geoms, torso_body_id, kp, kd, 750, "none"
    )
    reset_to_stance(model, data)
    set_sliding_friction(model, friction_geoms, 1.0)
    mujoco.mj_forward(model, data)
    for _ in range(750):
        apply_pd(model, data, target_qpos, kp=kp, kd=kd, gravity_ff=False, ff_ctrl=tau)
        mujoco.mj_step(model, data)
    z_ff = float(data.qpos[2])
    print(f"  baseline height={hold['z_end']:.4f}  pd_ff={z_ff:.4f}")
    factor = {"kind": "pd_variant", "ctrl": {"ff_ctrl": tau.tolist()}}
    IMPULSE_CFG["realistic"] = True
    base_crit = fy_crit_search(target_qpos, kp, kd, p_target=0.5)
    ff_crit = fy_crit_search(target_qpos, kp, kd, p_target=0.5, factor=factor)
    IMPULSE_CFG["realistic"] = False
    payload = {
        "kp": kp,
        "kd": kd,
        "tau_ff_abs_mean": float(np.mean(np.abs(tau))),
        "tau_ff_abs_max": float(np.max(np.abs(tau))),
        "z_base": hold["z_end"],
        "z_pd_ff": z_ff,
        "fy_crit_base": base_crit,
        "fy_crit_pd_ff": ff_crit,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "note": "controller variant; baseline kp=40 kd=3 without FF is unchanged",
    }
    write_json(Path("reports") / "pd_ff_compare.json", payload)
    print(
        f"  Fy_crit baseline={base_crit['fy_crit']:.1f} "
        f"pd_ff={ff_crit['fy_crit']:.1f} N"
    )
    return payload


def load_outcome_thresholds() -> dict | None:
    if not THRESHOLDS_PATH.exists():
        print(f"missing {THRESHOLDS_PATH}, stance_loss has no 3-sigma calibration")
        return None
    derivation = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
    STANCE_THRESH.update(
        {
            "z_threshold": derivation["z_threshold"],
            "slip_threshold": derivation["slip_threshold"],
            "z_ref": derivation["z_ref"],
            "n": int(derivation.get("n_nominal") or 0),
            "derivation": derivation,
        }
    )
    STANCE_REF["z"] = float(derivation["z_ref"])
    print(
        f"stance_loss thresholds from {THRESHOLDS_PATH}: "
        f"z<{derivation['z_threshold']:.4f} m  slip>{derivation['slip_threshold']:.4f} m"
    )
    return derivation


def torque_level_stats(batch: dict) -> dict:
    episodes = batch.get("episodes") or []
    compact = [torque_episode(ep) for ep in episodes]
    n = len(compact) if compact else int(batch.get("n_episodes") or 0)
    n_fail = sum(1 for ep in compact if ep["fail"]) if compact else int(batch.get("n_failures") or 0)
    n_fall = sum(1 for ep in compact if ep["fall"]) if compact else int(batch.get("n_fall") or batch.get("n_tilt") or 0)
    n_sl = sum(1 for ep in compact if ep["stance_loss"]) if compact else int(batch.get("n_stance_loss") or 0)
    tau = [float(ep["max_tau_over_limit"]) for ep in compact if ep.get("max_tau_over_limit") is not None]
    sat = [float(ep["saturation_frac"]) for ep in compact if ep.get("saturation_frac") is not None]
    z_s = [float(ep["z_settle"]) for ep in compact if ep.get("z_settle") is not None]
    sat_mean = float(np.mean(sat)) if sat else 0.0
    return {
        "n": n,
        "n_fail": n_fail,
        "fail_rate": n_fail / n if n else 0.0,
        "n_fall": n_fall,
        "fall_rate": n_fall / n if n else 0.0,
        "n_stance_loss": n_sl,
        "stance_loss_rate": n_sl / n if n else 0.0,
        "fail_rate_ci95": list(wilson_ci(n_fail, n)),
        "fall_rate_ci95": list(wilson_ci(n_fall, n)),
        "stance_loss_rate_ci95": list(wilson_ci(n_sl, n)),
        "max_tau_over_limit_mean": float(np.mean(tau)) if tau else None,
        "max_tau_over_limit_max": float(np.max(tau)) if tau else None,
        "saturation_frac_mean": sat_mean if sat else None,
        "saturation_frac_max": float(np.max(sat)) if sat else None,
        "limit_reached": bool(sat) and sat_mean > 1e-12,
        "limit_note": saturation_note(sat_mean if sat else 0.0),
        "z_settle_mean": float(np.mean(z_s)) if z_s else None,
        "n_hit_limit": sum(1 for ep in compact if ep.get("hit_limit")),
        "episodes": compact,
    }


def run_torque_scale_sample(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    scale: float,
    n_episodes: int = 3,
    friction: float = 0.8,
    lateral_force: float = 0.0,
) -> list[dict]:
    apply_torque_limit(model, scale)
    foot_ids = find_foot_geom_ids(model)
    rows = []
    for ep in range(1, int(n_episodes) + 1):
        result = simulate_episode(
            model,
            data,
            target_qpos,
            friction_geoms,
            torso_body_id,
            float(friction),
            float(lateral_force),
            kp,
            kd,
            episode=ep,
            jitter=True,
            record_trace=False,
            factor={"kind": "torque_limit", "scale": float(scale)},
            foot_ids=foot_ids,
        )
        rows.append(torque_episode(result))
    apply_torque_limit(model, 1.0)
    return rows


def run_torque_limit_test(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
) -> dict:
    stock = stock_ctrlrange(model)
    print("\ntorque_limit check: clip ctrl to actuator_ctrlrange, ctrllimited=1")
    print(
        f"  stock ctrlrange hip=[{stock[0, 0]:.2f},{stock[0, 1]:.2f}] "
        f"calf=[{stock[2, 0]:.2f},{stock[2, 1]:.2f}]  "
        f"ctrllimited={bool(np.all(model.actuator_ctrllimited))}"
    )
    derivation = load_outcome_thresholds()
    probe_rows = probe_torque_limit(
        model, data, target_qpos, friction_geoms, torso_body_id, kp, kd
    )
    specs = generate_episode_specs(
        N_TORQUE_EPISODES,
        SEED,
        SCENARIOS["nominal"]["friction_range"],
        SCENARIOS["nominal"]["lateral_force_range"],
    )
    print(
        f"\ntorque_limit batch: {N_TORQUE_EPISODES} nominal episodes, "
        f"shared seeds, scales {list(TORQUE_TEST_SCALES)}"
    )
    levels = []
    for scale in TORQUE_TEST_SCALES:
        written = apply_torque_limit(model, scale)
        batch = run_named_batch(
            target_qpos,
            kp,
            kd,
            specs,
            factor={"kind": "torque_limit", "scale": scale},
        )
        stats = torque_level_stats(batch)
        row = {
            "scale": scale,
            "label": f"{scale * 100:.0f}%",
            "ctrlrange_hip": [float(written[0, 0]), float(written[0, 1])],
            "ctrlrange_calf": [float(written[2, 0]), float(written[2, 1])],
            **stats,
        }
        levels.append(row)
        tau = row["max_tau_over_limit_mean"]
        sat = row["saturation_frac_mean"]
        print(
            f"  {row['label']:>4}  hip_lim={abs(written[0, 0]):.2f} N*m  "
            f"calf_lim={abs(written[2, 0]):.2f} N*m  "
            f"max|tau|/limit={0.0 if tau is None else tau:.3f}  "
            f"sat={format_sat_pct(sat)} ({row['limit_note']})  "
            f"fail={row['n_fail']}/{row['n']}={row['fail_rate']*100:.1f}%  "
            f"fall={row['n_fall']}/{row['n']}={row['fall_rate']*100:.1f}%  "
            f"stance_loss={row['n_stance_loss']}/{row['n']}={row['stance_loss_rate']*100:.1f}%  "
            f"z_settle={0.0 if row['z_settle_mean'] is None else row['z_settle_mean']:.4f}"
        )
    apply_torque_limit(model, 1.0)
    base = next(lv for lv in levels if abs(lv["scale"] - 1.0) < 1e-12)
    harsh = next(lv for lv in levels if abs(lv["scale"] - 0.05) < 1e-12)
    changed = (
        abs((harsh["fail_rate"] or 0.0) - (base["fail_rate"] or 0.0)) > 1e-9
        or abs((harsh["saturation_frac_mean"] or 0.0) - (base["saturation_frac_mean"] or 0.0)) > 1e-4
        or abs((harsh["z_settle_mean"] or 0.0) - (base["z_settle_mean"] or 0.0)) > 0.005
    )
    if changed:
        print("  5% changes saturation and/or failures relative to 100%")
    else:
        print("  warning: 5% did not differ from 100% on fail/sat/z")
    payload = {
        "kp": kp,
        "kd": kd,
        "seed": SEED,
        "n_episodes": N_TORQUE_EPISODES,
        "scenario": "nominal",
        "failure_criterion": "fall OR stance_loss",
        "stock_ctrlrange_hip": [float(stock[0, 0]), float(stock[0, 1])],
        "stock_ctrlrange_calf": [float(stock[2, 0]), float(stock[2, 1])],
        "ctrllimited": True,
        "thresholds": derivation,
        "probe": probe_rows,
        "levels": levels,
        "changed_at_5pct": changed,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(TORQUE_TEST_PATH, payload)
    return payload


def main() -> None:
    global ACTIVE_CONTROLLER, REPORT_PATH
    args = parse_args()
    if args.report:
        REPORT_PATH = Path(args.report)
    if args.compare:
        from paired_compare import run_compare_files

        run_compare_files(args.compare[0], args.compare[1], args.compare_out)
        return
    run_mode = args.mode
    pd_name = args.pd
    scenario = SCENARIOS[args.scenario]
    mode = PD_MODES[pd_name]
    main_kp = float(mode["kp"])
    main_kd = float(mode["kd"])
    ACTIVE_CONTROLLER, _import_root = prepare_controller_spec(str(args.controller))
    make_controller(ACTIVE_CONTROLLER, main_kp, main_kd)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    floor_id = find_floor_geom_id(model)
    foot_ids = find_foot_geom_ids(model)
    torso_body_id = find_torso_body_id(model)
    torso_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, torso_body_id) or ""
    friction_geoms = [floor_id] + [geom_id for geom_id in foot_ids if geom_id != floor_id]
    print_floor_selection(model, floor_id, foot_ids)
    print(
        f"impulse body: id={torso_body_id} name={torso_name!r} "
        f"xfrc_applied[:, 1], steps {IMPULSE_START}..{IMPULSE_END}"
    )
    print(f"run mode: {run_mode}")
    print(f"scenario: {scenario['id']} — {scenario['label']}")
    print(
        f"  Fy range={list(scenario['lateral_force_range'])}, "
        f"probe Fy={scenario['probe_force']:.1f} N"
    )
    print(f"main PD: {pd_name} kp={main_kp:g} kd={main_kd:g}")
    print(f"controller: {ACTIVE_CONTROLLER}")
    n_episodes = max(1, int(args.episodes))
    print(
        f"seed={SEED}  N_ENVS={N_ENVS}  CHUNK_SIZE={CHUNK_SIZE}  "
        f"--episodes={n_episodes}"
    )

    reset_to_stance(model, data)
    target_qpos = np.array(data.qpos[7:], copy=True)
    if model.nu != len(target_qpos):
        raise RuntimeError(
            f"actuator count ({model.nu}) does not match joint count ({len(target_qpos)})"
        )
    STANCE_REF["frac"] = float(args.stance_loss_frac)
    measure_stance_reference(
        model,
        data,
        target_qpos,
        friction_geoms,
        torso_body_id,
        float(PD_MODES["baseline"]["kp"]),
        float(PD_MODES["baseline"]["kd"]),
        float(args.stance_loss_frac),
        announce_frac_rule=(run_mode not in ("criteria", "torque", "ofat")),
    )
    measured_z_ref = STANCE_REF.get("z")
    measured_frac = STANCE_REF.get("frac")
    extra_config = {
        "stance_ref_z": STANCE_REF["z"],
        "n_episodes": n_episodes,
    }
    if run_mode in ("criteria", "torque", "ofat", "full", "quick"):
        extra_config["failure_criterion"] = "fall OR stance_loss"
    if run_mode not in ("criteria", "torque"):
        extra_config["stance_loss_frac"] = STANCE_REF["frac"]
    if run_mode in ("full", "quick", "ofat"):
        load_outcome_thresholds()
    config_meta = build_config_meta(
        kp=main_kp, kd=main_kd, scenario=scenario, extra=extra_config
    )
    print(f"config_hash={config_meta['config_hash']}  created_at={config_meta['created_at']}")

    if run_mode == "ff":
        run_gravity_ff_diagnosis(
            model, data, target_qpos, friction_geoms, torso_body_id, main_kp, main_kd
        )
        return
    if run_mode == "criteria":
        run_criteria_analysis(
            model, data, target_qpos, friction_geoms, torso_body_id, main_kp, main_kd
        )
        return
    if run_mode == "fycrit":
        run_fycrit_ofat(target_qpos, main_kp, main_kd)
        return
    if run_mode == "pd_ff":
        run_pd_ff_variant(
            model, data, target_qpos, friction_geoms, torso_body_id, main_kp, main_kd
        )
        return
    if run_mode == "torque":
        run_torque_limit_test(
            model, data, target_qpos, friction_geoms, torso_body_id, main_kp, main_kd
        )
        return

    if run_mode == "grid":
        stance = run_stance_hold_batch(target_qpos, main_kp, main_kd)
        scenario_rates = run_scenario_rate_check(target_qpos, main_kp, main_kd)
        bias_check = run_bias_offset_check(target_qpos, main_kp, main_kd)
        rows = run_baseline_grid(target_qpos, main_kp, main_kd)
        cliff = run_cliff_grid(target_qpos, main_kp, main_kd)
        traces = run_mu02_traces(
            model, data, target_qpos, friction_geoms, torso_body_id, main_kp, main_kd
        )
        report = {
            "run_mode": run_mode,
            "scenario": scenario["id"],
            "scenario_label": scenario["label"],
            "seed": SEED,
            "kp": main_kp,
            "kd": main_kd,
            "failure_criterion": {
                "contact_z": CONTACT_Z,
                "tilt_deg": math.degrees(TILT_FAIL_RAD),
                "upright_hold_s": UPRIGHT_HOLD_S,
                "sag_z": SAG_Z,
                "stance_loss_frac": STANCE_REF.get("frac"),
                "stance_ref_z": STANCE_REF.get("z"),
            },
            "stance_hold": {
                "friction": 1.0,
                "lateral_force": 0.0,
                **batch_rate_summary(stance),
            },
            "scenario_rates": scenario_rates,
            "bias_offset_check": bias_check,
            "stance_ref_z": STANCE_REF.get("z"),
            "stance_loss_frac": STANCE_REF.get("frac"),
            "baseline_grid_path": str(GRID_PATH),
            "baseline_grid_json": str(GRID_JSON_PATH),
            "baseline_grid": rows,
            "cliff_grid_path": str(CLIFF_PATH),
            "cliff_grid": cliff,
            "mu02_traces_path": str(MU02_TRACES_PATH),
            "mu02_traces_summary": {
                "n": len(traces.get("traces") or []),
                "path_z_svg": traces.get("path_z_svg"),
                "path_roll_svg": traces.get("path_roll_svg"),
            },
            **config_meta,
        }
        write_json(GRID_JSON_PATH, report)
        print("grid does not overwrite reports/batch_failures.json")
        return
    if run_mode == "sag":
        rows = run_sag_compare(
            model, data, target_qpos, friction_geoms, torso_body_id
        )
        write_json(
            SAG_JSON_PATH,
            {
                "run_mode": run_mode,
                "seed": SEED,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "note": "Default PD is unchanged. The operator chooses the variant.",
                "rows": rows,
            },
        )
        print("sag does not overwrite reports/batch_failures.json")
        return
    if run_mode == "full":
        run_single_probes(
            model,
            data,
            target_qpos,
            friction_geoms,
            torso_body_id,
            main_kp,
            main_kd,
            lateral_force=float(scenario["probe_force"]),
        )
    elif run_mode == "quick":
        print("\nquick run: main batch only, no probes and no OFAT")
    else:
        print("\nOFAT run: parallel parameter check, main batch skipped")

    episode_specs_kwargs = {
        "friction_range": scenario["friction_range"],
        "force_range": scenario["lateral_force_range"],
    }
    main_batch = None
    if run_mode == "full":
        specs = generate_episode_specs(n_episodes, SEED, **episode_specs_kwargs)
        main_batch = run_pd_comparison(
            model,
            data,
            target_qpos,
            friction_geoms,
            torso_body_id,
            specs,
            pd_name,
        )
    elif run_mode == "quick":
        specs = generate_episode_specs(n_episodes, SEED, **episode_specs_kwargs)
        main_batch = run_stress_batch_parallel(
            specs,
            main_kp,
            main_kd,
            True,
            target_qpos,
        )

    sweeps = None
    sweeps_by_scenario: list[dict] = []
    if run_mode in ("full", "ofat"):
        ofat_scenarios = resolve_ofat_scenarios(run_mode, scenario)
        for ofat_scenario in ofat_scenarios:
            if ofat_scenario["id"] != scenario["id"]:
                print(
                    f"OFAT from scenario {ofat_scenario['id']} "
                    f"({ofat_scenario['label']}), not from {scenario['id']}"
                )
            one = run_parameter_sweeps(
                target_qpos, main_kp, main_kd, ofat_scenario, n_episodes=n_episodes
            )
            sweeps_by_scenario.append(one)
        if sweeps_by_scenario:
            sweeps = sweeps_by_scenario[0]
            compare_ofat_sweeps(sweeps_by_scenario)

    report: dict = {
        "run_mode": run_mode,
        "scenario": scenario["id"],
        "scenario_label": scenario["label"],
        "friction_range": list(scenario["friction_range"]),
        "lateral_force_range": list(scenario["lateral_force_range"]),
        "mode": pd_name,
        "pd": pd_name,
        "seed": SEED,
        **config_meta,
        "kp": main_kp,
        "kd": main_kd,
        "floor_geom_id": int(floor_id),
        "floor_geom_name": geom_name(model, floor_id),
        "floor_geom_type": geom_type_name(model, floor_id),
        "geom_friction_shape": list(model.geom_friction.shape),
        "friction_geom_ids": friction_geoms,
        "torso_body_id": int(torso_body_id),
        "impulse_start": IMPULSE_START,
        "impulse_end": IMPULSE_END,
        "target_qpos": target_qpos.tolist(),
    }
    stance_meta = stance_reference_record(
        ACTIVE_CONTROLLER,
        pd_name,
        main_kp,
        main_kd,
        measured_z_ref,
        measured_frac,
    )
    report["controller"] = ACTIVE_CONTROLLER
    report["stance_reference"] = stance_meta
    if main_batch is not None:
        print_main_batch_summary(
            pd_name,
            main_batch,
            controller=ACTIVE_CONTROLLER,
            stance=stance_meta,
        )
        report.update(
            report_from_batch(
                pd_name,
                run_mode,
                main_batch,
                model,
                floor_id,
                friction_geoms,
                torso_body_id,
                target_qpos,
            )
        )
    if sweeps is not None:
        report["parameter_sweeps"] = sweeps
    if sweeps_by_scenario:
        report["parameter_sweeps_by_scenario"] = {
            item["scenario"]: item for item in sweeps_by_scenario
        }
        moderate = next(
            (item for item in sweeps_by_scenario if item["scenario"] == "moderate"),
            None,
        )
        if moderate is not None:
            report["parameter_sweeps_moderate"] = moderate
    save_report(report)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
