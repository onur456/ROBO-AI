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
CHUNK_SIZE = 10
MAX_STEPS = 1500
WINDOW_SIZE = 200
FALL_HEIGHT = 0.20
FRICTION_RANGE = (0.05, 1.0)
LATERAL_FORCE_RANGE = (-350.0, 350.0)
IMPULSE_START = 450
IMPULSE_END = 500
PROBE_FRICTIONS = (0.05, 1.0)
PROBE_LATERAL_FORCE = 180.0
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
        choices=tuple(PD_MODES),
        default=MAIN_MODE,
        help="режим основного прогона и JSON (сравнение всегда baseline vs stiff)",
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
) -> None:
    if model.nu != len(target_qpos):
        return
    data.ctrl[:] = kp * (target_qpos - data.qpos[7:]) - kd * data.qvel[6:]
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
    for step in range(1, MAX_STEPS + 1):
        apply_pd(model, data, target_qpos, kp=kp, kd=kd, clip_ctrl=False)
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
) -> None:
    print(
        f"\nпроверка трения: Fy={PROBE_LATERAL_FORCE:.1f} Н на шагах "
        f"{IMPULSE_START}..{IMPULSE_END}, friction={list(PROBE_FRICTIONS)}"
    )
    results = [
        run_controlled_episode(
            model,
            data,
            target_qpos,
            friction_geoms,
            mu,
            PROBE_LATERAL_FORCE,
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
) -> None:
    origin = read_body_ipos(model, torso_body_id)
    print(
        f"\nпроверка смещения COM корпуса: body_id={torso_body_id} "
        f"body_ipos0={origin.tolist()}  μ={COM_PROBE_FRICTION:g} "
        f"Fy={COM_PROBE_FORCE:.1f} Н на шагах {IMPULSE_START}..{IMPULSE_END}"
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
                COM_PROBE_FORCE,
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
) -> None:
    origin = read_ctrlrange(model)
    print(
        f"\nпроверка ограничения момента: μ={COM_PROBE_FRICTION:g} "
        f"Fy={COM_PROBE_FORCE:.1f} Н на шагах {IMPULSE_START}..{IMPULSE_END}  "
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
                COM_PROBE_FORCE,
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
) -> None:
    print(
        f"\nпроверка задержки команд (#6): μ={COM_PROBE_FRICTION:g} "
        f"Fy={COM_PROBE_FORCE:.1f} Н на шагах {IMPULSE_START}..{IMPULSE_END}"
    )
    results: list[dict] = []
    for delay in DELAY_PROBE_STEPS:
        result = run_controlled_episode(
            model,
            data,
            target_qpos,
            friction_geoms,
            COM_PROBE_FRICTION,
            COM_PROBE_FORCE,
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
) -> None:
    origin = read_dof_damping(model)
    legs = leg_dof_ids(model)
    print(
        f"\nпроверка демпфирования суставов (#7): μ={COM_PROBE_FRICTION:g} "
        f"Fy={COM_PROBE_FORCE:.1f} Н на шагах {IMPULSE_START}..{IMPULSE_END}  "
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
                COM_PROBE_FORCE,
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
) -> dict:
    failed_episodes: list[dict] = []
    failed_friction: list[float] = []
    failed_force: list[float] = []
    survived_friction: list[float] = []
    survived_force: list[float] = []

    for episode, friction, lateral_force in specs:
        reset_to_stance(model, data)
        set_sliding_friction(model, friction_geoms, friction)
        mujoco.mj_forward(model, data)

        window: deque[dict] | None = deque(maxlen=WINDOW_SIZE) if keep_windows else None
        episode_failed = False
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
            apply_pd(model, data, target_qpos, kp=kp, kd=kd)
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


def generate_episode_specs(n_episodes: int, seed: int) -> list[tuple[int, float, float]]:
    rng = np.random.default_rng(seed)
    return [
        (
            episode,
            float(rng.uniform(*FRICTION_RANGE)),
            float(rng.uniform(*LATERAL_FORCE_RANGE)),
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


def main() -> None:
    args = parse_args()
    mode_name = args.mode
    mode = PD_MODES[mode_name]
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
    print(f"основной режим: {mode_name} kp={main_kp:g} kd={main_kd:g}")
    print(f"seed={SEED}  N_ENVS={N_ENVS}  CHUNK_SIZE={CHUNK_SIZE}")

    reset_to_stance(model, data)
    target_qpos = np.array(data.qpos[7:], copy=True)
    if model.nu != len(target_qpos):
        raise RuntimeError(
            f"число актуаторов ({model.nu}) не совпадает с числом суставов ({len(target_qpos)})"
        )

    probe_friction_effect(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=main_kp, kd=main_kd
    )
    probe_com_offset(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=main_kp, kd=main_kd
    )
    probe_torque_limit(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=main_kp, kd=main_kd
    )
    probe_action_delay(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=main_kp, kd=main_kd
    )
    probe_joint_damping(
        model, data, target_qpos, friction_geoms, torso_body_id, kp=main_kp, kd=main_kd
    )

    specs = generate_episode_specs(N_ENVS, SEED)

    print("\nсравнение PD (стойка: μ=0.8, Fy=0; батч на тех же эпизодах):")
    main_batch = None
    for name, gains in PD_MODES.items():
        kp = float(gains["kp"])
        kd = float(gains["kd"])
        stance = probe_stance_hold(
            model, data, target_qpos, friction_geoms, torso_body_id, kp=kp, kd=kd
        )
        keep_windows = name == mode_name
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

    n_failures = main_batch["n_failures"]
    n_survived = main_batch["n_survived"]
    failure_rate = main_batch["failure_rate"]
    failed_episodes = main_batch["failed_episodes"]
    failed_friction_stats = value_stats(main_batch["failed_friction"])
    failed_force_stats = value_stats(main_batch["failed_force"])
    survived_friction_stats = value_stats(main_batch["survived_friction"])
    survived_force_stats = value_stats(main_batch["survived_force"])

    print(f"\nосновной прогон ({mode_name}, kp={main_batch['kp']:g}, kd={main_batch['kd']:g}):")
    print(f"всего прогонов: {main_batch['n_episodes']}  seed={SEED}")
    print(
        f"параллельно: {main_batch['n_workers']} workers, "
        f"{main_batch['n_chunks']} chunks, время={main_batch['elapsed_s']:.2f}с"
    )
    print(f"упало: {n_failures}")
    print(f"устояло: {n_survived}")
    print(f"процент сбоев: {failure_rate * 100:.1f}%")
    print(f"упавшие, трение: {format_stats(failed_friction_stats)}")
    print(f"упавшие, боковая сила: {format_stats(failed_force_stats)}")
    print(f"устоявшие, трение: {format_stats(survived_friction_stats)}")
    print(f"устоявшие, боковая сила: {format_stats(survived_force_stats)}")

    report = {
        "n_episodes": main_batch["n_episodes"],
        "n_envs": N_ENVS,
        "n_failures": n_failures,
        "n_survived": n_survived,
        "failure_rate": failure_rate,
        "mode": mode_name,
        "seed": SEED,
        "elapsed_s": main_batch.get("elapsed_s"),
        "n_workers": main_batch.get("n_workers"),
        "n_chunks": main_batch.get("n_chunks"),
        "kp": main_batch["kp"],
        "kd": main_batch["kd"],
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
            "friction": failed_friction_stats,
            "lateral_force": failed_force_stats,
        },
        "survived": {
            "friction": survived_friction_stats,
            "lateral_force": survived_force_stats,
        },
        "failures": failed_episodes,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"результаты сохранены: {REPORT_PATH}")


if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    main()
