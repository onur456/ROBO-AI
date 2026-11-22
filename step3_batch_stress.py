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
SCENARIOS = {
    "default": {
        "id": "default",
        "label": "базовый: трение и боковой удар (Fy до ±350 Н)",
        "friction_range": FRICTION_RANGE,
        "lateral_force_range": LATERAL_FORCE_RANGE,
        "probe_force": PROBE_LATERAL_FORCE,
    },
    "nominal": {
        "id": "nominal",
        "label": "nominal: μ 0.6–1.0, Fy до ±100 Н",
        "friction_range": NOMINAL_FRICTION_RANGE,
        "lateral_force_range": NOMINAL_FORCE_RANGE,
        "probe_force": 80.0,
    },
    "moderate": {
        "id": "moderate",
        "label": "moderate: μ 0.4–1.0, Fy до ±250 Н",
        "friction_range": MODERATE_FRICTION_RANGE,
        "lateral_force_range": MODERATE_FORCE_RANGE,
        "probe_force": 180.0,
    },
    "extreme": {
        "id": "extreme",
        "label": "extreme: μ 0.05–1.0, Fy до ±350 Н (прежний default)",
        "friction_range": FRICTION_RANGE,
        "lateral_force_range": LATERAL_FORCE_RANGE,
        "probe_force": PROBE_LATERAL_FORCE,
    },
    "slip_only": {
        "id": "slip_only",
        "label": "скольжение и стойка без бокового удара (Fy=0)",
        "friction_range": FRICTION_RANGE,
        "lateral_force_range": (0.0, 0.0),
        "probe_force": 0.0,
    },
    "heavy_impact": {
        "id": "heavy_impact",
        "label": "усиленный боковой удар (Fy до ±500 Н)",
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
TORQUE_PROBE_SCALES = (1.0, 0.5, 0.25)
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
        "name": "смещение COM по X",
        "baseline": "zero",
        "levels": (
            {
                "id": "zero",
                "label": "базовый 0 м",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.0, 0.0]},
            },
            {
                "id": "neg",
                "label": "назад −0.03 м",
                "factor": {"kind": "com_offset", "offset": [-0.03, 0.0, 0.0]},
            },
            {
                "id": "pos",
                "label": "вперёд +0.03 м",
                "factor": {"kind": "com_offset", "offset": [0.03, 0.0, 0.0]},
            },
        ),
    },
    {
        "id": "com_y",
        "name": "смещение COM по Y",
        "baseline": "zero",
        "levels": (
            {
                "id": "zero",
                "label": "базовый 0 м",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.0, 0.0]},
            },
            {
                "id": "weak_neg",
                "label": "слабый −0.02 м",
                "factor": {"kind": "com_offset", "offset": [0.0, -0.02, 0.0]},
            },
            {
                "id": "weak_pos",
                "label": "слабый +0.02 м",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.02, 0.0]},
            },
            {
                "id": "harsh_neg",
                "label": "жёсткий −0.04 м",
                "factor": {"kind": "com_offset", "offset": [0.0, -0.04, 0.0]},
            },
            {
                "id": "harsh_pos",
                "label": "жёсткий +0.04 м",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.04, 0.0]},
            },
        ),
    },
    {
        "id": "com_z",
        "name": "смещение COM по Z",
        "baseline": "zero",
        "levels": (
            {
                "id": "zero",
                "label": "базовый 0 м",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.0, 0.0]},
            },
            {
                "id": "weak",
                "label": "слабый +0.015 м",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.0, 0.015]},
            },
            {
                "id": "harsh",
                "label": "жёсткий +0.03 м",
                "factor": {"kind": "com_offset", "offset": [0.0, 0.0, 0.03]},
            },
        ),
    },
    {
        "id": "torque_limit",
        "name": "ограничение момента моторов",
        "baseline": "full",
        "levels": (
            {
                "id": "full",
                "label": "базовый 100% ctrlrange",
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
        "name": "задержка команд",
        "baseline": "zero",
        "levels": (
            {
                "id": "zero",
                "label": "базовая 0 шагов",
                "factor": {"kind": "delay", "steps": 0},
            },
            {
                "id": "weak",
                "label": "слабая 5 шагов",
                "factor": {"kind": "delay", "steps": 5},
            },
            {
                "id": "harsh",
                "label": "жёсткая 15 шагов",
                "factor": {"kind": "delay", "steps": 15},
            },
        ),
    },
    {
        "id": "joint_damping",
        "name": "демпфирование суставов",
        "baseline": "base",
        "levels": (
            {
                "id": "base",
                "label": "базовое 1x",
                "factor": {"kind": "damping", "scale": 1.0},
            },
            {
                "id": "weak",
                "label": "слабое 0x",
                "factor": {"kind": "damping", "scale": 0.0},
            },
            {
                "id": "harsh",
                "label": "жёсткое 10x",
                "factor": {"kind": "damping", "scale": 10.0},
            },
        ),
    },
    {
        "id": "floor_contact",
        "name": "мягкость контакта пола",
        "baseline": "baseline",
        "levels": (
            {
                "id": "baseline",
                "label": "базовый solref",
                "factor": {"kind": "contact", "preset": "baseline"},
            },
            {
                "id": "stiff",
                "label": "жёсткий пол",
                "factor": {
                    "kind": "contact",
                    "preset": "stiff",
                    "solref": [0.002, 1.0],
                    "solimp": [0.9, 0.99, 0.0001, 0.5, 2.0],
                },
            },
            {
                "id": "soft",
                "label": "податливый пол",
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
        "name": "шум датчиков",
        "baseline": "none",
        "levels": (
            {
                "id": "none",
                "label": "без шума",
                "factor": {"kind": "noise", "q": 0.0, "v": 0.0},
            },
            {
                "id": "medium",
                "label": "средний шум",
                "factor": {"kind": "noise", "q": 0.03, "v": 0.3},
            },
            {
                "id": "strong",
                "label": "сильный шум",
                "factor": {"kind": "noise", "q": 0.12, "v": 1.2},
            },
        ),
    },
    {
        "id": "control_hold",
        "name": "частота обновления команды",
        "baseline": "every",
        "levels": (
            {
                "id": "every",
                "label": "каждый шаг (500 Гц)",
                "factor": {"kind": "ctrl_hold", "steps": 1},
            },
            {
                "id": "weak",
                "label": "каждые 5 шагов (100 Гц)",
                "factor": {"kind": "ctrl_hold", "steps": 5},
            },
            {
                "id": "harsh",
                "label": "каждые 20 шагов (25 Гц)",
                "factor": {"kind": "ctrl_hold", "steps": 20},
            },
        ),
    },
    {
        "id": "sensor_bias",
        "name": "смещение нуля датчиков",
        "baseline": "none",
        "levels": (
            {
                "id": "none",
                "label": "без смещения",
                "factor": {"kind": "bias", "q": 0.0, "drift": 0.0},
            },
            {
                "id": "weak_neg",
                "label": "слабый −0.015 рад",
                "factor": {"kind": "bias", "q": -0.015, "drift": 0.0},
            },
            {
                "id": "weak_pos",
                "label": "слабый +0.015 рад",
                "factor": {"kind": "bias", "q": 0.015, "drift": 0.0},
            },
            {
                "id": "offset",
                "label": "смещение нуля +0.05 рад",
                "factor": {"kind": "bias", "q": 0.05, "drift": 0.0},
            },
            {
                "id": "drift",
                "label": "дрейф +0.0002 рад/шаг",
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
STANCE_REF = {"z": None, "frac": STANCE_LOSS_FRAC, "kp": None, "kd": None}
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
        choices=("full", "quick", "ofat", "grid", "sag", "ff"),
        default="full",
        help=(
            "full: единичные проверки, сравнение PD, OFAT и основной батч; "
            "quick: только основной батч в reports/batch_failures.json; "
            "ofat: OFAT на nominal (контроль) и moderate (чувствительность); "
            "grid: стойка, сетка Fy/μ, обрыв и траектории μ=0.2 (не трогает batch_failures.json); "
            "sag: сравнение вариантов стойки; "
            "ff: диагностика gravity feed-forward"
        ),
    )
    parser.add_argument(
        "--pd",
        choices=tuple(PD_MODES),
        default=MAIN_MODE,
        help="PD-режим: baseline=kp40/kd3, weak_pd=kp25/kd1.5 (старый), stiff=алиас baseline",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default="default",
        help=(
            "default/extreme: μ 0.05–1, Fy до ±350 Н; "
            "nominal: μ 0.6–1, Fy до ±100 Н; "
            "moderate: μ 0.4–1, Fy до ±250 Н; "
            "slip_only: скольжение/стойка при Fy=0; "
            "heavy_impact: усиленный удар (Fy до ±500 Н)"
        ),
    )
    parser.add_argument(
        "--stance-loss-frac",
        type=float,
        default=STANCE_LOSS_FRAC,
        help="stance_loss: z после успокоения < frac * высота эталонной стойки (по умолчанию 0.85)",
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
) -> None:
    data.xfrc_applied[torso_body_id] = 0.0
    if impulse_start <= step <= impulse_end:
        data.xfrc_applied[torso_body_id, 1] = force


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


def episode_seed(episode: int) -> int:
    return int(SEED) * 1_000_003 + int(episode)


def sample_episode_perturbation(episode: int, n_joints: int) -> dict:
    rng = np.random.default_rng(episode_seed(episode))
    shift = int(rng.integers(-IMPULSE_JITTER_STEPS, IMPULSE_JITTER_STEPS + 1))
    return {
        "seed": episode_seed(episode),
        "joint_jitter": rng.uniform(-JOINT_JITTER_RAD, JOINT_JITTER_RAD, size=n_joints),
        "impulse_start": IMPULSE_START + shift,
        "impulse_end": IMPULSE_END + shift,
        "mass_scale": 1.0 + float(rng.uniform(-MASS_JITTER, MASS_JITTER)),
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
    print(f"результаты сохранены: {path}")


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
                f"не удалось записать geom_friction[{geom_id}, 0]={friction}, сейчас {actual}"
            )


def print_floor_selection(
    model: mujoco.MjModel, floor_id: int, foot_ids: list[int]
) -> None:
    print("геометрия пола:")
    print(f"  index={floor_id}")
    print(f"  name={geom_name(model, floor_id)!r}")
    print(f"  type={geom_type_name(model, floor_id)}")
    print(f"  priority={int(model.geom_priority[floor_id])}")
    print(f"  condim={int(model.geom_condim[floor_id])}")
    print(f"  geom_friction.shape={tuple(model.geom_friction.shape)}")
    print(f"  geom_friction[{floor_id}]={model.geom_friction[floor_id].tolist()}")
    print("контактные geom лап (выше priority, от них MuJoCo берёт friction контакта):")
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
            f"не удалось записать body_ipos[{body_id}]={np.asarray(ipos).tolist()}, "
            f"сейчас {written.tolist()}"
        )
    mujoco.mj_setConst(model, data)
    return written


def read_ctrlrange(model: mujoco.MjModel) -> np.ndarray:
    return np.array(model.actuator_ctrlrange, copy=True)


def write_ctrlrange(model: mujoco.MjModel, ctrlrange: np.ndarray) -> np.ndarray:
    model.actuator_ctrlrange[:] = ctrlrange
    written = read_ctrlrange(model)
    if float(np.max(np.abs(written - np.asarray(ctrlrange, dtype=float)))) > 1e-12:
        raise RuntimeError("не удалось записать actuator_ctrlrange")
    return written


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
    q_obs = np.asarray(data.qpos[7:], dtype=float)
    v_obs = np.asarray(data.qvel[6:], dtype=float)
    if rng is not None and (obs_noise_q > 0.0 or obs_noise_v > 0.0):
        if obs_noise_q > 0.0:
            q_obs = q_obs + rng.normal(0.0, obs_noise_q, size=q_obs.shape)
        if obs_noise_v > 0.0:
            v_obs = v_obs + rng.normal(0.0, obs_noise_v, size=v_obs.shape)
    if abs(obs_bias_q) > 0.0:
        q_obs = q_obs + obs_bias_q
    data.ctrl[:] = kp * (target_qpos - q_obs) - kd * v_obs
    if gravity_ff:
        add_mapped_gravity_ff(model, data)
    if ff_ctrl is not None:
        data.ctrl[:] += np.asarray(ff_ctrl, dtype=float)
    if clip_ctrl:
        np.clip(
            data.ctrl,
            model.actuator_ctrlrange[:, 0],
            model.actuator_ctrlrange[:, 1],
            out=data.ctrl,
        )


def episode_status(result: dict) -> str:
    if result["fail_step"] is None:
        return "устоял"
    return f"упал на шаге {result['fail_step']}"


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
        f"z_end={result['z_end']:.4f}, z_min={result['z_min']:.4f}{dz}"
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
            f"(по первым {n} шагам)"
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
        raise RuntimeError("не удалось записать dof_damping")
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
            f"не удалось записать body_inertia[{body_id}]={np.asarray(inertia).tolist()}, "
            f"сейчас {written.tolist()}"
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
        raise RuntimeError("не удалось записать geom_solref")
    if float(np.max(np.abs(written_imp - solimp))) > 1e-12:
        raise RuntimeError("не удалось записать geom_solimp")
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
        origin = read_ctrlrange(model)
        scale = float(factor.get("scale", 1.0))
        write_ctrlrange(model, origin * scale)


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
        return "нет данных"
    return (
        f"среднее={stats['mean']:.4f}, "
        f"диапазон=[{stats['min']:.4f}, {stats['max']:.4f}]"
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
        foot_xy0 = {gid: np.array(data.geom_xpos[gid][:2], copy=True) for gid in feet}
        contact_friction = None
        if data.ncon > 0:
            contact_friction = np.asarray(data.contact[0].friction, dtype=float).tolist()

        window: deque[dict] | None = deque(maxlen=WINDOW_SIZE) if keep_window else None
        ctrl_buf: deque[np.ndarray] = deque(maxlen=delay + 1)
        last_desired: np.ndarray | None = None
        limits = np.array(model.actuator_ctrlrange, copy=True)
        limit_abs = np.maximum(np.abs(limits[:, 0]), np.abs(limits[:, 1]))
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
        z_ref = STANCE_REF["z"] if stance_ref_z is None else stance_ref_z
        loss_frac = STANCE_LOSS_FRAC if stance_loss_frac is None else float(stance_loss_frac)
        settle_window = max(1, int(math.ceil(UPRIGHT_HOLD_S / max(dt, 1e-9))))

        for step in range(1, MAX_STEPS + 1):
            if last_desired is None or (step - 1) % hold == 0:
                apply_pd(
                    model,
                    data,
                    q_target,
                    kp=kp,
                    kd=kd,
                    rng=obs_rng,
                    obs_noise_q=noise_q,
                    obs_noise_v=noise_v,
                    obs_bias_q=bias_q + drift_q * (step - 1),
                    gravity_ff=gravity_ff,
                    ff_ctrl=applied_ff,
                )
                last_desired = np.array(data.ctrl, copy=True)
            ctrl_buf.append(np.array(last_desired, copy=True))
            raw = np.array(ctrl_buf[0], copy=True)
            data.ctrl[:] = raw
            max_abs_ctrl_raw = max(max_abs_ctrl_raw, float(np.max(np.abs(raw))))
            util = float(np.max(np.abs(raw) / np.maximum(limit_abs, 1e-9)))
            max_util_raw = max(max_util_raw, util)
            exceeded = np.any(raw < limits[:, 0] - 1e-12) | np.any(
                raw > limits[:, 1] + 1e-12
            )
            if clip_ctrl:
                if bool(exceeded):
                    hit_limit = True
                np.clip(data.ctrl, limits[:, 0], limits[:, 1], out=data.ctrl)
            max_abs_ctrl = max(max_abs_ctrl, float(np.max(np.abs(data.ctrl))))
            apply_torso_impulse(
                data,
                torso_body_id,
                step,
                lateral_force,
                impulse_start=impulse_start,
                impulse_end=impulse_end,
            )
            mujoco.mj_step(model, data)
            z = float(data.qpos[2])
            roll, pitch = freejoint_roll_pitch(data.qpos)
            tilt = tilt_from_vertical(data.qpos)
            speed = float(np.linalg.norm(data.qvel[:3]))
            min_z = min(min_z, z)
            max_abs_roll = max(max_abs_roll, abs(roll))
            max_abs_pitch = max(max_abs_pitch, abs(pitch))
            max_tilt = max(max_tilt, tilt)
            max_body_speed = max(max_body_speed, speed)
            for gid in feet:
                xy = data.geom_xpos[gid][:2]
                foot_slip_max = max(
                    foot_slip_max,
                    float(np.linalg.norm(xy - foot_xy0[gid])),
                )
            zs.append(z)
            ys.append(float(data.qpos[1]))
            if record_trace:
                rolls.append(roll)
                pitches.append(pitch)
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
                fail_reason = "tilt"
                break
    finally:
        model.body_mass[torso_body_id] = orig_mass
        mujoco.mj_setConst(model, data)

    written = {
        geom_id: float(model.geom_friction[geom_id, 0]) for geom_id in friction_geoms
    }
    n_tail = min(len(zs), settle_window) if zs else 0
    z_after_settle = float(np.mean(zs[-n_tail:])) if n_tail else float(z_start)
    min_z_after_settle = float(np.min(zs[-n_tail:])) if n_tail else float(z_start)
    tilt_fail = fail_reason == "tilt"
    stance_loss = False
    stance_loss_step = None
    if z_ref is not None and z_ref > 1e-9:
        z_lim = float(loss_frac) * float(z_ref)
        if z_after_settle < z_lim:
            stance_loss = True
            stance_loss_step = len(zs)
            if fail_reason is None:
                fail_reason = "stance_loss"
                fail_step = stance_loss_step
            elif fail_reason == "tilt":
                fail_reason = "tilt+stance_loss"
    failed = tilt_fail or stance_loss
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
        "fail_tilt": tilt_fail,
        "fail_stance_loss": stance_loss,
        "fail_step": fail_step,
        "fail_time": None if fail_step is None else float(fail_step) * dt,
        "impact_phase": None
        if fail_step is None
        else impact_phase(fail_step, impulse_start, impulse_end),
        "sag_below_0_20": sag_below,
        "sag_step": sag_step,
        "min_z": float(min_z),
        "z_after_settle": z_after_settle,
        "min_z_after_settle": min_z_after_settle,
        "max_abs_roll": float(max_abs_roll),
        "max_abs_pitch": float(max_abs_pitch),
        "max_tilt": float(max_tilt),
        "max_body_speed": float(max_body_speed),
        "foot_slip_max": float(foot_slip_max),
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
        "hit_limit": hit_limit,
        "y_end": ys[-1] if ys else 0.0,
        "max_abs_y": max((abs(y) for y in ys), default=0.0),
        "z": zs if record_trace else [],
        "y": ys if record_trace else [],
        "roll": rolls,
        "pitch": pitches,
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
        f"\nпроверка трения: Fy={lateral_force:.1f} Н на шагах "
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
            f"упал на шаге {result['fail_step']}"
            if result["fail_step"] is not None
            else "устоял"
        )
        print(
            f"  mu={result['friction']:.2f}: записано {result['written_sliding']}, "
            f"contact.friction={result['contact_friction']}, {status}, "
            f"z_end={result['z_end']:.4f}, z_min={result['z_min']:.4f}, "
            f"y_end={result['y_end']:.4f}, max|y|={result['max_abs_y']:.4f}"
        )

    z0, z1 = np.asarray(results[0]["z"]), np.asarray(results[1]["z"])
    y0, y1 = np.asarray(results[0]["y"]), np.asarray(results[1]["y"])
    n = min(len(z0), len(z1))
    z_diff = float(np.max(np.abs(z0[:n] - z1[:n])))
    y_diff = float(np.max(np.abs(y0[:n] - y1[:n])))
    print(f"  max |Δz|={z_diff:.6f}, max |Δy|={y_diff:.6f} (по первым {n} шагам)")
    if z_diff < 1e-9 and y_diff < 1e-9:
        print("  предупреждение: траектории совпали, трение не влияет на контакт")
    else:
        print("  траектории различаются, трение влияет на устойчивость")


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
        f"\nпроверка смещения COM корпуса: body_id={torso_body_id} "
        f"body_ipos0={origin.tolist()}  μ={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} Н на шагах {IMPULSE_START}..{IMPULSE_END}"
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
                f"упал на шаге {result['fail_step']}"
                if result["fail_step"] is not None
                else "устоял"
            )
            print(
                f"  {label}: offset={list(offset)} ipos={written.tolist()}  "
                f"{status}, z_end={result['z_end']:.4f}, z_min={result['z_min']:.4f}"
            )
    finally:
        restored = write_body_ipos(model, data, torso_body_id, origin)
        if float(np.max(np.abs(restored - origin))) > 1e-12:
            raise RuntimeError(
                f"не удалось восстановить body_ipos[{torso_body_id}]={origin.tolist()}, "
                f"сейчас {restored.tolist()}"
            )
        print(f"  COM восстановлен: body_ipos={restored.tolist()}")

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
                f"(по первым {n} шагам)"
            )
        if not any_diff:
            print("  предупреждение: траектории совпали, смещение COM не влияет")
        else:
            print("  траектории различаются, смещение COM влияет на устойчивость")


def probe_torque_limit(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
    kp: float,
    kd: float,
    lateral_force: float = COM_PROBE_FORCE,
) -> None:
    origin = read_ctrlrange(model)
    print(
        f"\nпроверка ограничения момента: μ={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} Н на шагах {IMPULSE_START}..{IMPULSE_END}  "
        f"ctrlrange0 hip=[{origin[0, 0]:.2f}, {origin[0, 1]:.2f}]"
    )
    results: list[dict] = []
    try:
        for scale in TORQUE_PROBE_SCALES:
            written = write_ctrlrange(model, origin * scale)
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
            hit = "да" if result["hit_limit"] else "нет"
            print(
                f"  {label} ctrlrange: hip=[{written[0, 0]:.2f}, {written[0, 1]:.2f}] "
                f"calf=[{written[2, 0]:.2f}, {written[2, 1]:.2f}]  "
                f"{episode_status(result)}, z_end={result['z_end']:.4f}, "
                f"z_min={result['z_min']:.4f}  "
                f"max|ctrl|_raw={result['max_abs_ctrl_raw']:.3f}  "
                f"max|ctrl|={result['max_abs_ctrl']:.3f}  "
                f"потолок hip={hip_limit:.2f} calf={calf_limit:.2f}  упёрся={hit}"
            )
    finally:
        restored = write_ctrlrange(model, origin)
        if float(np.max(np.abs(restored - origin))) > 1e-12:
            raise RuntimeError("не удалось восстановить actuator_ctrlrange")
        print(
            f"  ctrlrange восстановлен: hip=[{restored[0, 0]:.2f}, {restored[0, 1]:.2f}]"
        )

    reduced_hit = [r for r in results if r["scale"] < 1.0 - 1e-12]
    if reduced_hit and not any(r["hit_limit"] for r in reduced_hit):
        print(
            "  предупреждение: max|ctrl| так и не дошёл до лимита, "
            "потолок момента на этом сценарии не сработал"
        )
    else:
        any_diff = print_z_deltas(results, "label")
        if any_diff:
            print("  траектории различаются, потолок момента влияет на устойчивость")
        elif reduced_hit and any(r["hit_limit"] for r in reduced_hit):
            print("  потолок сработал (ctrl упёрся в лимит)")


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
        f"\nпроверка задержки команд (#6): μ={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} Н на шагах {IMPULSE_START}..{IMPULSE_END}"
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
            f"z_end={result['z_end']:.4f}, z_min={result['z_min']:.4f}"
        )
    any_diff = print_z_deltas(results, "label")
    if not any_diff:
        print("  предупреждение: траектории совпали, задержка команд не влияет")
    else:
        print("  траектории различаются, задержка команд влияет на устойчивость")


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
        f"\nпроверка демпфирования суставов (#7): μ={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} Н на шагах {IMPULSE_START}..{IMPULSE_END}  "
        f"dof_damping ног 1x={origin[legs].tolist()}"
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
                f"  {label}: damping_ног={written[legs].tolist()}  "
                f"{episode_status(result)}, z_end={result['z_end']:.4f}, "
                f"z_min={result['z_min']:.4f}"
            )
    finally:
        restored = write_dof_damping(model, origin)
        if float(np.max(np.abs(restored - origin))) > 1e-12:
            raise RuntimeError("не удалось восстановить dof_damping")
        print(f"  dof_damping восстановлен: ног={restored[legs].tolist()}")

    any_diff = print_z_deltas(results, "label")
    if not any_diff:
        print("  предупреждение: траектории совпали, демпфирование суставов не влияет")
    else:
        print("  траектории различаются, демпфирование суставов влияет на устойчивость")


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
        f"\nпроверка инерции корпуса (#8): body_id={torso_body_id} "
        f"body_inertia0={origin.tolist()}  μ={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} Н на шагах {IMPULSE_START}..{IMPULSE_END}"
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
            raise RuntimeError("не удалось восстановить body_inertia")
        print(f"  body_inertia восстановлена: {restored.tolist()}")

    any_diff = print_z_deltas(results, "label")
    if not any_diff:
        print("  предупреждение: траектории совпали, инерция корпуса не влияет")
    else:
        print("  траектории различаются, инерция корпуса влияет на устойчивость")


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
        f"\nпроверка контакта пола (#9): geom_ids={friction_geoms} "
        f"solref0={origin_ref[0].tolist()} solimp0={origin_imp[0].tolist()}  "
        f"μ={COM_PROBE_FRICTION:g} Fy={lateral_force:.1f} Н "
        f"на шагах {IMPULSE_START}..{IMPULSE_END}"
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
            f"  контакт восстановлен: solref={restored_ref[0].tolist()} "
            f"solimp={restored_imp[0].tolist()}"
        )

    any_diff = print_z_deltas(results, "label")
    if not any_diff:
        print("  предупреждение: траектории совпали, мягкость контакта не влияет")
    else:
        print("  траектории различаются, мягкость контакта влияет на устойчивость")


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
        f"\nпроверка шума измерений (#10): μ={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} Н на шагах {IMPULSE_START}..{IMPULSE_END}  "
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
        print("  предупреждение: траектории совпали, шум измерений не влияет")
    else:
        print("  траектории различаются, шум измерений влияет на устойчивость")


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
        f"\nпроверка смещения нуля датчиков: μ={COM_PROBE_FRICTION:g} "
        f"Fy={lateral_force:.1f} Н на шагах {IMPULSE_START}..{IMPULSE_END}"
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
            extra=f"bias={bias_q:g} рад  drift={drift_q:g} рад/шаг",
            ref=results[0],
        )
    any_diff = print_z_deltas(results, "label")
    if not any_diff:
        print("  предупреждение: траектории совпали, смещение нуля датчиков не влияет")
    else:
        print("  траектории различаются, смещение нуля датчиков влияет на устойчивость")


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
    status = "упал" if result["fail_step"] is not None else "устоял"
    print(
        f"контроль стойки kp={kp:g} kd={kd:g}: z_start={result['z_start']:.4f} "
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
            target_override=target_override,
            stance_ref_z=stance_ref_z,
            stance_loss_frac=stance_loss_frac,
            foot_ids=foot_ids,
        )
        record = {
            **{k: v for k, v in result.items() if k not in ("z", "y", "roll", "pitch")},
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
    n_tilt = sum(1 for ep in episodes if ep.get("fail_tilt"))
    n_stance_loss = sum(1 for ep in episodes if ep.get("fail_stance_loss"))
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
        "n_stance_loss": n_stance_loss,
        "stance_loss_rate": n_stance_loss / n_episodes if n_episodes else 0.0,
        "stance_loss_rate_ci95": list(wilson_ci(n_stance_loss, n_episodes)),
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
        }
        for chunk in chunks
    ]
    started = time.perf_counter()
    if n_workers <= 1:
        parts = [_run_chunk(payload) for payload in payloads]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            parts = list(pool.map(_run_chunk, payloads))
    elapsed = time.perf_counter() - started
    batch = merge_batch_parts(parts, kp, kd)
    batch["elapsed_s"] = float(elapsed)
    batch["n_workers"] = int(n_workers)
    batch["n_chunks"] = len(chunks)
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
    }


def run_parameter_sweeps(
    target_qpos: np.ndarray,
    kp: float,
    kd: float,
    scenario: dict,
) -> dict:
    specs = generate_episode_specs(
        N_SWEEP_EPISODES,
        SEED,
        scenario["friction_range"],
        scenario["lateral_force_range"],
    )
    print(
        f"\nOFAT-проверка параметров: {N_SWEEP_EPISODES} эпизодов на уровень, "
        f"seed={SEED}, сценарий={scenario['id']}, параллельные воркеры"
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
        worst = max(levels_out, key=lambda row: float(row["failure_rate"]))
        for row in levels_out:
            delta = float(row["failure_rate"]) - base_rate
            row["delta_vs_baseline"] = delta
            mark = "  (норма)" if row["id"] == baseline["id"] else ""
            lo, hi = row.get("failure_rate_ci95") or (0.0, 0.0)
            print(
                f"    {row['label']}: сбои={row['failure_rate'] * 100:.1f}% "
                f"({row['n_failures']}/{row['n_episodes']}, "
                f"CI95[{lo * 100:.1f},{hi * 100:.1f}])  "
                f"просадка={float(row.get('sag_rate') or 0.0) * 100:.1f}%  "
                f"Δ vs норма={delta * 100:+.1f} п.п.  "
                f"до удара: {row['n_before_impact']}  "
                f"при/после: {row['n_at_or_after_impact']}  "
                f"время={row['elapsed_s']:.2f}с{mark}"
            )
        print(
            f"    больше всего сбоев: {worst['label']} "
            f"({worst['failure_rate'] * 100:.1f}%, "
            f"{(float(worst['failure_rate']) - base_rate) * 100:+.1f} п.п. к норме)"
        )
        parameters.append(
            {
                "id": spec["id"],
                "name": spec["name"],
                "baseline": spec["baseline"],
                "levels": levels_out,
                "worst_level": worst["id"],
                "worst_label": worst["label"],
                "baseline_failure_rate": base_rate,
                "worst_failure_rate": float(worst["failure_rate"]),
                "worst_delta_vs_baseline": float(worst["failure_rate"]) - base_rate,
            }
        )
    return {
        "n_episodes": N_SWEEP_EPISODES,
        "seed": SEED,
        "kp": kp,
        "kd": kd,
        "scenario": scenario["id"],
        "scenario_label": scenario["label"],
        "friction_range": list(scenario["friction_range"]),
        "lateral_force_range": list(scenario["lateral_force_range"]),
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
    reason_txt = ", ".join(f"{name}={count}" for name, count in sorted(reasons.items())) or "нет"
    nt = int(batch.get("n_tilt") or 0)
    ns = int(batch.get("n_stance_loss") or 0)
    print(
        f"{prefix}сбои={k}/{n}={batch['failure_rate'] * 100:.1f}% "
        f"CI95[{lo * 100:.1f},{hi * 100:.1f}]  "
        f"tilt={nt}/{n}={float(batch.get('tilt_rate') or 0.0) * 100:.1f}%  "
        f"stance_loss={ns}/{n}={float(batch.get('stance_loss_rate') or 0.0) * 100:.1f}%  "
        f"просадка z<0.20={ks}/{n}={float(batch.get('sag_rate') or 0.0) * 100:.1f}% "
        f"CI95[{slo * 100:.1f},{shi * 100:.1f}]  "
        f"причины: {reason_txt}"
        + (f"  t={batch['elapsed_s']:.2f}с" if batch.get("elapsed_s") is not None else "")
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
    print(f"\n{title} (строки μ, столбцы Fy, %):")
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
        f"\n{title}: {len(mus)}×{len(fys)} узлов, "
        f"{n_episodes} эпизодов на узел, seed={SEED}, "
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
            print_batch_rates(f"  μ={mu:g} Fy={fy:6.1f} Н: ", batch)
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
        f"\nэталон стойки: опорный PD kp={kp:g} kd={kd:g}, μ=1 Fy=0, без джиттера\n"
        f"  z_start={result['z_start']:.4f} z_end={result['z_end']:.4f} "
        f"z_after_settle={z_ref:.4f} м\n"
        f"  stance_loss если z_after_settle < {frac:.2f}×{z_ref:.4f} = {z_lim:.4f} м"
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
        f"\nпроверка смещения нуля +0.05 рад на moderate "
        f"({N_SWEEP_EPISODES} эпизодов)"
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
    print_batch_rates("  без смещения: ", base)
    print_batch_rates("  +0.05 рад:     ", biased)
    return {"none": batch_rate_summary(base), "offset": batch_rate_summary(biased)}


def run_stance_hold_batch(target_qpos: np.ndarray, kp: float, kd: float) -> dict:
    specs = [(episode, 1.0, 0.0) for episode in range(1, N_STANCE_EPISODES + 1)]
    print(
        f"\nстойка без толчка: μ=1.0 Fy=0, {N_STANCE_EPISODES} эпизодов, "
        f"kp={kp:g} kd={kd:g} seed={SEED}"
    )
    batch = run_stress_batch_parallel(specs, kp, kd, False, target_qpos)
    print_batch_rates("  ", batch)
    if batch["n_failures"] != 0:
        print("  предупреждение: стойка при μ=1 Fy=0 не держится по новому критерию")
    else:
        print("  стойка держится по новому критерию: 0 сбоев")
    return batch


def run_scenario_rate_check(target_qpos: np.ndarray, kp: float, kd: float) -> list[dict]:
    rows: list[dict] = []
    print("\nдоли сбоев по сценариям (новый критерий vs просадка z<0.20):")
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
        target_qpos, kp, kd, GRID_MU, GRID_FY, N_GRID_EPISODES, "сетка Fy/μ"
    )
    write_csv(GRID_PATH, rows, GRID_CSV_FIELDS)
    print(f"сетка сохранена: {GRID_PATH}")
    print_grid_table(rows, GRID_MU, GRID_FY, "failure_rate", "доля сбоев (tilt ИЛИ stance_loss)")
    print_grid_table(rows, GRID_MU, GRID_FY, "tilt_rate", "доля tilt")
    print_grid_table(rows, GRID_MU, GRID_FY, "stance_loss_rate", "доля stance_loss")
    print_grid_table(rows, GRID_MU, GRID_FY, "sag_rate", "доля просадки z<0.20 (метрика)")
    unique_rates = {round(float(r["failure_rate"]), 6) for r in rows}
    if unique_rates <= {0.0, 1.0}:
        print("  предупреждение: все узлы сетки всё ещё 0% или 100%")
    else:
        print("  узлы сетки не все бинарные 0/100")
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
        target_qpos, kp, kd, CLIFF_MU, CLIFF_FY, N_GRID_EPISODES, "тонкая сетка около обрыва"
    )
    write_csv(CLIFF_PATH, rows, GRID_CSV_FIELDS)
    print(f"тонкая сетка сохранена: {CLIFF_PATH}")
    series = []
    for mu in CLIFF_MU:
        subset = [r for r in rows if abs(r["mu"] - mu) < 1e-12]
        subset.sort(key=lambda r: r["fy"])
        series.append(
            {
                "label": f"μ={mu:g} сбой",
                "x": [r["fy"] for r in subset],
                "y": [r["failure_rate"] for r in subset],
                "y_lo": [r["failure_ci_lo"] for r in subset],
                "y_hi": [r["failure_ci_hi"] for r in subset],
            }
        )
        series.append(
            {
                "label": f"μ={mu:g} просадка z<0.20",
                "x": [r["fy"] for r in subset],
                "y": [r["sag_rate"] for r in subset],
                "y_lo": [r["sag_ci_lo"] for r in subset],
                "y_hi": [r["sag_ci_hi"] for r in subset],
            }
        )
    write_line_svg(
        CLIFF_SVG_PATH,
        series,
        title="Доля сбоев против Fy (Уилсон 95%)",
        xlabel="Fy, Н",
        ylabel="доля",
        y_min=0.0,
        y_max=1.0,
    )
    print(f"график сохранён: {CLIFF_SVG_PATH}")
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
    print("\nтраектории μ=0.2: z(t) и крен для Fy=0/100/150/200/250")
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
        title="μ=0.2: высота корпуса z(t)",
        xlabel="t, с",
        ylabel="z, м",
    )
    write_line_svg(
        MU02_ROLL_SVG_PATH,
        roll_series,
        title="μ=0.2: крен roll(t)",
        xlabel="t, с",
        ylabel="крен, град",
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
            "label": "текущий PD kp=25 kd=1.5, без FF",
            "kp": 25.0,
            "kd": 1.5,
            "factor": None,
        },
        {
            "id": "stiff",
            "label": "повышенный kp=40 kd=3.0",
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
            "label": "kp=25 kd=1.5 + ctrl из keyframe home как FF",
            "kp": 25.0,
            "kd": 1.5,
            "factor": {"kind": "pd_variant", "ctrl": {"keyframe_ff": True}},
        },
    )
    print("\nсравнение вариантов стойки (PD по умолчанию не меняется):")
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
            f"|qfrc_bias|={physics['qfrc_bias_abs_mean']:.3f} Н·м"
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
            "note": "PD по умолчанию не изменён. Выбор варианта — за оператором.",
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
    print("\nсравнение OFAT: nominal (контроль) vs moderate (чувствительность)")
    nom_params = {p["id"]: p for p in by_id["nominal"]["parameters"]}
    for param in by_id["moderate"]["parameters"]:
        left = nom_params.get(param["id"])
        if left is None:
            continue
        print(
            f"  {param['name']}: nominal худший {left['worst_label']} "
            f"{left['worst_failure_rate'] * 100:.1f}% "
            f"(Δ{left['worst_delta_vs_baseline'] * 100:+.1f}); "
            f"moderate худший {param['worst_label']} "
            f"{param['worst_failure_rate'] * 100:.1f}% "
            f"(Δ{param['worst_delta_vs_baseline'] * 100:+.1f})"
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
    print(f"результаты сохранены: {REPORT_PATH}")


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
    print("\nсравнение PD (стойка: μ=0.8, Fy=0; батч на тех же эпизодах):")
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
            f"сбои={batch['failure_rate'] * 100:.1f}% "
            f"({batch['n_failures']}/{batch['n_episodes']})  "
            f"до шага {IMPULSE_START}: {batch['n_before_impact']}  "
            f"при/после {IMPULSE_START}: {batch['n_at_or_after_impact']}  "
            f"время={batch['elapsed_s']:.2f}с  "
            f"workers={batch['n_workers']} chunks={batch['n_chunks']}"
        )
        if keep_windows:
            main_batch = batch
    if main_batch is None:
        raise RuntimeError("не удалось выполнить основной прогон PD")
    return main_batch


def print_main_batch_summary(pd_name: str, batch: dict) -> None:
    failed_friction_stats = value_stats(batch["failed_friction"])
    failed_force_stats = value_stats(batch["failed_force"])
    survived_friction_stats = value_stats(batch["survived_friction"])
    survived_force_stats = value_stats(batch["survived_force"])
    print(f"\nосновной прогон ({pd_name}, kp={batch['kp']:g}, kd={batch['kd']:g}):")
    print(f"всего прогонов: {batch['n_episodes']}  seed={SEED}")
    print(
        f"параллельно: {batch['n_workers']} workers, "
        f"{batch['n_chunks']} chunks, время={batch['elapsed_s']:.2f}с"
    )
    print(f"упало (новый критерий): {batch['n_failures']}")
    print(f"устояло: {batch['n_survived']}")
    print_batch_rates("  ", batch)
    print(f"упавшие, трение: {format_stats(failed_friction_stats)}")
    print(f"упавшие, боковая сила: {format_stats(failed_force_stats)}")
    print(f"устоявшие, трение: {format_stats(survived_friction_stats)}")
    print(f"устоявшие, боковая сила: {format_stats(survived_force_stats)}")


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
    print("\nдиагностика gravity feed-forward")
    print("соответствие актуатор → сустав → dof:")
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
            f"|qerr|={snap['qerr_abs_mean']:.4f} рад  "
            f"|bias_legs| start mean={snap['bias_abs_mean_start_legs']:.3f} "
            f"max={snap['bias_abs_max_start_legs']:.3f}  "
            f"settled mean={snap['bias_abs_mean_settled_legs']:.3f} "
            f"max={snap['bias_abs_max_settled_legs']:.3f}  "
            f"qfrc_bias free-z start={snap['bias_free_z_start']:.2f} "
            f"settled={snap['bias_free_z_settled']:.2f}"
        )
        if mode == "none":
            print(
                f"{'joint':<16} {'dof':>3} {'tau_act':>10} {'tau_bias':>10} {'отн.ошибка':>10}"
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
        f"\nвысота settled: none={none['z_end']:.4f}  "
        f"legacy={legacy['z_end']:.4f}  mapped={mapped['z_end']:.4f}  "
        f"Δ mapped−none={lift:+.4f} м"
    )
    if lift > 0.005:
        print(
            "FF с маппингом актуатор→dof поднимает стойку. "
            "Предложение: сделать PD+gravity_ff базовым контроллером. "
            "Не включаю без решения."
        )
    else:
        print("mapped FF не дал заметного подъёма (>5 мм). Базовый PD без FF.")
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


def main() -> None:
    args = parse_args()
    run_mode = args.mode
    pd_name = args.pd
    scenario = SCENARIOS[args.scenario]
    mode = PD_MODES[pd_name]
    main_kp = float(mode["kp"])
    main_kd = float(mode["kd"])

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    floor_id = find_floor_geom_id(model)
    foot_ids = find_foot_geom_ids(model)
    torso_body_id = find_torso_body_id(model)
    torso_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, torso_body_id) or ""
    friction_geoms = [floor_id] + [geom_id for geom_id in foot_ids if geom_id != floor_id]
    print_floor_selection(model, floor_id, foot_ids)
    print(
        f"корпус для импульса: id={torso_body_id} name={torso_name!r} "
        f"xfrc_applied[:, 1], шаги {IMPULSE_START}..{IMPULSE_END}"
    )
    print(f"режим прогона: {run_mode}")
    print(f"сценарий: {scenario['id']} — {scenario['label']}")
    print(
        f"  Fy диапазон={list(scenario['lateral_force_range'])}, "
        f"проба Fy={scenario['probe_force']:.1f} Н"
    )
    print(f"основной PD: {pd_name} kp={main_kp:g} kd={main_kd:g}")
    print(f"seed={SEED}  N_ENVS={N_ENVS}  CHUNK_SIZE={CHUNK_SIZE}")

    reset_to_stance(model, data)
    target_qpos = np.array(data.qpos[7:], copy=True)
    if model.nu != len(target_qpos):
        raise RuntimeError(
            f"число актуаторов ({model.nu}) не совпадает с числом суставов ({len(target_qpos)})"
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
    )
    extra_config = {
        "stance_ref_z": STANCE_REF["z"],
        "stance_loss_frac": STANCE_REF["frac"],
    }
    config_meta = build_config_meta(
        kp=main_kp, kd=main_kd, scenario=scenario, extra=extra_config
    )
    print(f"config_hash={config_meta['config_hash']}  created_at={config_meta['created_at']}")

    if run_mode == "ff":
        run_gravity_ff_diagnosis(
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
        print("grid не перезаписывает reports/batch_failures.json")
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
                "note": "PD по умолчанию не изменён. Выбор варианта — за оператором.",
                "rows": rows,
            },
        )
        print("sag не перезаписывает reports/batch_failures.json")
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
        print("\nбыстрый прогон: только основной батч, без проб и OFAT")
    else:
        print("\nOFAT-прогон: параллельная проверка параметров, основной батч пропущен")

    episode_specs_kwargs = {
        "friction_range": scenario["friction_range"],
        "force_range": scenario["lateral_force_range"],
    }
    main_batch = None
    if run_mode == "full":
        specs = generate_episode_specs(N_ENVS, SEED, **episode_specs_kwargs)
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
        specs = generate_episode_specs(N_ENVS, SEED, **episode_specs_kwargs)
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
                    f"OFAT от сценария {ofat_scenario['id']} "
                    f"({ofat_scenario['label']}), не от {scenario['id']}"
                )
            one = run_parameter_sweeps(target_qpos, main_kp, main_kd, ofat_scenario)
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
    if main_batch is not None:
        print_main_batch_summary(pd_name, main_batch)
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
