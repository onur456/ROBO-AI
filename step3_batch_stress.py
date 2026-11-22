import argparse
import json
import os
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
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
FALL_HEIGHT = 0.20
FRICTION_RANGE = (0.05, 1.0)
LATERAL_FORCE_RANGE = (-350.0, 350.0)
HEAVY_FORCE_RANGE = (-500.0, 500.0)
IMPULSE_START = 450
IMPULSE_END = 500
PROBE_FRICTIONS = (0.05, 1.0)
PROBE_LATERAL_FORCE = 180.0
SCENARIOS = {
    "default": {
        "id": "default",
        "label": "базовый: трение и боковой удар (Fy до ±350 Н)",
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
    "baseline": {"kp": 25.0, "kd": 1.5},
    "stiff": {"kp": 40.0, "kd": 3.0},
}
MAIN_MODE = "baseline"
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
        choices=("full", "quick", "ofat"),
        default="full",
        help=(
            "full: единичные проверки, сравнение PD, OFAT и основной батч; "
            "quick: только основной батч в reports/batch_failures.json; "
            "ofat: только параллельная OFAT-проверка"
        ),
    )
    parser.add_argument(
        "--pd",
        choices=tuple(PD_MODES),
        default=MAIN_MODE,
        help="PD-режим основного прогона и JSON (в full сравнение всегда baseline vs stiff)",
    )
    parser.add_argument(
        "--scenario",
        choices=tuple(SCENARIOS),
        default="default",
        help=(
            "default: трение и боковой удар (Fy до ±350 Н); "
            "slip_only: скольжение/стойка при Fy=0; "
            "heavy_impact: усиленный удар (Fy до ±500 Н)"
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
    data: mujoco.MjData, torso_body_id: int, step: int, force: float
) -> None:
    data.xfrc_applied[torso_body_id] = 0.0
    if IMPULSE_START <= step <= IMPULSE_END:
        data.xfrc_applied[torso_body_id, 1] = force


def impact_phase(fail_step: int) -> str:
    if fail_step < IMPULSE_START:
        return "before_impact"
    if fail_step <= IMPULSE_END:
        return "during_impact"
    return "after_impact"


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


def snapshot(step: int, data: mujoco.MjData, z: float) -> dict:
    return {
        "step": step,
        "time": float(data.time),
        "z": z,
        "qpos_head": data.qpos[:7].tolist(),
        "qvel_head": data.qvel[:6].tolist(),
        "ctrl": data.ctrl.tolist(),
    }


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
) -> dict:
    reset_to_stance(model, data)
    set_sliding_friction(model, friction_geoms, friction)
    mujoco.mj_forward(model, data)
    z_start = float(data.qpos[2])

    contact_friction = None
    if data.ncon > 0:
        contact_friction = np.asarray(data.contact[0].friction, dtype=float).tolist()

    zs: list[float] = []
    ys: list[float] = []
    fail_step = None
    max_abs_ctrl = 0.0
    max_abs_ctrl_raw = 0.0
    max_util_raw = 0.0
    hit_limit = False
    delay = max(0, int(action_delay))
    ctrl_buf: deque[np.ndarray] = deque(maxlen=delay + 1)
    limits = np.array(model.actuator_ctrlrange, copy=True)
    limit_abs = np.maximum(np.abs(limits[:, 0]), np.abs(limits[:, 1]))
    obs_rng = None
    if obs_noise_q > 0.0 or obs_noise_v > 0.0:
        obs_rng = np.random.default_rng(SEED if noise_seed is None else noise_seed)
    for step in range(1, MAX_STEPS + 1):
        apply_pd(
            model,
            data,
            target_qpos,
            kp=kp,
            kd=kd,
            clip_ctrl=False,
            rng=obs_rng,
            obs_noise_q=obs_noise_q,
            obs_noise_v=obs_noise_v,
            obs_bias_q=obs_bias_q + obs_drift_q * (step - 1),
        )
        desired = np.array(data.ctrl, copy=True)
        ctrl_buf.append(desired)
        raw = np.array(ctrl_buf[0], copy=True)
        data.ctrl[:] = raw
        max_abs_ctrl_raw = max(max_abs_ctrl_raw, float(np.max(np.abs(raw))))
        util = float(np.max(np.abs(raw) / np.maximum(limit_abs, 1e-9)))
        max_util_raw = max(max_util_raw, util)
        exceeded = np.any(raw < limits[:, 0] - 1e-12) | np.any(raw > limits[:, 1] + 1e-12)
        if clip_ctrl:
            if bool(exceeded):
                hit_limit = True
            np.clip(data.ctrl, limits[:, 0], limits[:, 1], out=data.ctrl)
        max_abs_ctrl = max(max_abs_ctrl, float(np.max(np.abs(data.ctrl))))
        apply_torso_impulse(data, torso_body_id, step, lateral_force)
        mujoco.mj_step(model, data)
        z = float(data.qpos[2])
        zs.append(z)
        ys.append(float(data.qpos[1]))
        if z < FALL_HEIGHT:
            fail_step = step
            break

    written = {
        geom_id: float(model.geom_friction[geom_id, 0]) for geom_id in friction_geoms
    }
    return {
        "z_start": z_start,
        "friction": friction,
        "lateral_force": lateral_force,
        "written_sliding": written,
        "contact_friction": contact_friction,
        "fail_step": fail_step,
        "n_steps": len(zs),
        "z_end": zs[-1],
        "z_min": min(zs),
        "max_abs_ctrl": max_abs_ctrl,
        "max_abs_ctrl_raw": max_abs_ctrl_raw,
        "max_util_raw": max_util_raw,
        "hit_limit": hit_limit,
        "y_end": ys[-1],
        "max_abs_y": max(abs(y) for y in ys),
        "z": zs,
        "y": ys,
    }


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
    failed_episodes: list[dict] = []
    failed_friction: list[float] = []
    failed_force: list[float] = []
    survived_friction: list[float] = []
    survived_force: list[float] = []
    cfg = control_from_factor(factor)
    delay = int(cfg["delay"])
    noise_q = float(cfg["noise_q"])
    noise_v = float(cfg["noise_v"])
    clip_ctrl = bool(cfg["clip_ctrl"])
    hold = max(1, int(cfg["hold"]))
    bias_q = float(cfg["bias_q"])
    drift_q = float(cfg["drift_q"])
    limits = np.array(model.actuator_ctrlrange, copy=True)

    for episode, friction, lateral_force in specs:
        reset_to_stance(model, data)
        set_sliding_friction(model, friction_geoms, friction)
        mujoco.mj_forward(model, data)

        window: deque[dict] | None = deque(maxlen=WINDOW_SIZE) if keep_windows else None
        episode_failed = False
        ctrl_buf: deque[np.ndarray] = deque(maxlen=max(0, delay) + 1)
        last_desired: np.ndarray | None = None
        obs_rng = (
            np.random.default_rng(SEED + int(episode))
            if noise_q > 0.0 or noise_v > 0.0
            else None
        )
        params = {
            "episode": episode,
            "friction": friction,
            "lateral_force": lateral_force,
            "impulse_start": IMPULSE_START,
            "impulse_end": IMPULSE_END,
            "torso_body_id": int(torso_body_id),
            "floor_geom_id": int(floor_id),
            "kp": kp,
            "kd": kd,
        }

        for step in range(1, MAX_STEPS + 1):
            if last_desired is None or (step - 1) % hold == 0:
                apply_pd(
                    model,
                    data,
                    target_qpos,
                    kp=kp,
                    kd=kd,
                    rng=obs_rng,
                    obs_noise_q=noise_q,
                    obs_noise_v=noise_v,
                    obs_bias_q=bias_q + drift_q * (step - 1),
                )
                last_desired = np.array(data.ctrl, copy=True)
            ctrl_buf.append(np.array(last_desired, copy=True))
            data.ctrl[:] = ctrl_buf[0]
            if clip_ctrl:
                np.clip(data.ctrl, limits[:, 0], limits[:, 1], out=data.ctrl)
            apply_torso_impulse(data, torso_body_id, step, lateral_force)
            mujoco.mj_step(model, data)
            z = float(data.qpos[2])
            if window is not None:
                window.append(snapshot(step, data, z))

            if z < FALL_HEIGHT:
                record = {
                    **params,
                    "fail_step": step,
                    "fail_time": float(data.time),
                    "final_z": z,
                    "impact_phase": impact_phase(step),
                }
                if window is not None:
                    record["window"] = list(window)
                failed_episodes.append(record)
                failed_friction.append(friction)
                failed_force.append(lateral_force)
                episode_failed = True
                break

        if not episode_failed:
            survived_friction.append(friction)
            survived_force.append(lateral_force)

    n_episodes = len(specs)
    n_failures = len(failed_episodes)
    n_before = sum(
        1
        for ep in failed_episodes
        if ep.get("impact_phase") == "before_impact" or int(ep["fail_step"]) < IMPULSE_START
    )
    n_at_or_after = n_failures - n_before
    return {
        "kp": kp,
        "kd": kd,
        "n_episodes": n_episodes,
        "n_failures": n_failures,
        "n_survived": n_episodes - n_failures,
        "failure_rate": n_failures / n_episodes if n_episodes else 0.0,
        "n_before_impact": n_before,
        "n_at_or_after_impact": n_at_or_after,
        "failed_episodes": failed_episodes,
        "failed_friction": failed_friction,
        "failed_force": failed_force,
        "survived_friction": survived_friction,
        "survived_force": survived_force,
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
    failed_episodes: list[dict] = []
    failed_friction: list[float] = []
    failed_force: list[float] = []
    survived_friction: list[float] = []
    survived_force: list[float] = []
    n_episodes = 0
    for part in parts:
        n_episodes += int(part["n_episodes"])
        failed_episodes.extend(part["failed_episodes"])
        failed_friction.extend(part["failed_friction"])
        failed_force.extend(part["failed_force"])
        survived_friction.extend(part["survived_friction"])
        survived_force.extend(part["survived_force"])
    failed_episodes.sort(key=lambda ep: int(ep["episode"]))
    n_failures = len(failed_episodes)
    n_before = sum(
        1
        for ep in failed_episodes
        if ep.get("impact_phase") == "before_impact" or int(ep["fail_step"]) < IMPULSE_START
    )
    return {
        "kp": kp,
        "kd": kd,
        "n_episodes": n_episodes,
        "n_failures": n_failures,
        "n_survived": n_episodes - n_failures,
        "failure_rate": n_failures / n_episodes if n_episodes else 0.0,
        "n_before_impact": n_before,
        "n_at_or_after_impact": n_failures - n_before,
        "failed_episodes": failed_episodes,
        "failed_friction": failed_friction,
        "failed_force": failed_force,
        "survived_friction": survived_friction,
        "survived_force": survived_force,
    }


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
        "n_before_impact": int(batch["n_before_impact"]),
        "n_at_or_after_impact": int(batch["n_at_or_after_impact"]),
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
            print(
                f"    {row['label']}: сбои={row['failure_rate'] * 100:.1f}% "
                f"({row['n_failures']}/{row['n_episodes']})  "
                f"Δ vs норма={delta * 100:+.1f} п.п.  "
                f"до 450: {row['n_before_impact']}  "
                f"при/после 450: {row['n_at_or_after_impact']}  "
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
        json.dumps(merged, ensure_ascii=False, indent=2),
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
    for name, gains in PD_MODES.items():
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
    print(f"упало: {batch['n_failures']}")
    print(f"устояло: {batch['n_survived']}")
    print(f"процент сбоев: {batch['failure_rate'] * 100:.1f}%")
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
    if run_mode in ("full", "ofat"):
        sweeps = run_parameter_sweeps(target_qpos, main_kp, main_kd, scenario)

    report: dict = {
        "run_mode": run_mode,
        "scenario": scenario["id"],
        "scenario_label": scenario["label"],
        "friction_range": list(scenario["friction_range"]),
        "lateral_force_range": list(scenario["lateral_force_range"]),
        "mode": pd_name,
        "pd": pd_name,
        "seed": SEED,
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
    save_report(report)


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
