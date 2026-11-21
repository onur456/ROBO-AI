import json
from collections import deque
from pathlib import Path

import numpy as np
import mujoco

MODEL_PATH = r"C:\Users\HP\Desktop\mujoco_menagerie-main\mujoco_menagerie-main\unitree_go2\scene.xml"
N_EPISODES = 40
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
KP = 25.0
KD = 1.5
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


def apply_pd(model: mujoco.MjModel, data: mujoco.MjData, target_qpos: np.ndarray) -> None:
    if model.nu != len(target_qpos):
        return
    data.ctrl[:] = KP * (target_qpos - data.qpos[7:]) - KD * data.qvel[6:]


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
    for step in range(1, MAX_STEPS + 1):
        apply_pd(model, data, target_qpos)
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


def probe_stance_hold(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_qpos: np.ndarray,
    friction_geoms: list[int],
    torso_body_id: int,
) -> None:
    result = run_controlled_episode(
        model,
        data,
        target_qpos,
        friction_geoms,
        STANCE_CHECK_FRICTION,
        STANCE_CHECK_FORCE,
        torso_body_id,
    )
    status = "упал" if result["fail_step"] is not None else "устоял"
    print(
        f"контроль стойки: z_start={result['z_start']:.4f} "
        f"z_end={result['z_end']:.4f} {status}"
    )


def main() -> None:
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

    reset_to_stance(model, data)
    target_qpos = np.array(data.qpos[7:], copy=True)
    if model.nu != len(target_qpos):
        raise RuntimeError(
            f"число актуаторов ({model.nu}) не совпадает с числом суставов ({len(target_qpos)})"
        )

    probe_stance_hold(model, data, target_qpos, friction_geoms, torso_body_id)
    probe_friction_effect(model, data, target_qpos, friction_geoms, torso_body_id)

    rng = np.random.default_rng()
    failed_episodes: list[dict] = []
    failed_friction: list[float] = []
    failed_force: list[float] = []
    survived_friction: list[float] = []
    survived_force: list[float] = []

    for episode in range(1, N_EPISODES + 1):
        reset_to_stance(model, data)

        friction = float(rng.uniform(*FRICTION_RANGE))
        lateral_force = float(rng.uniform(*LATERAL_FORCE_RANGE))
        set_sliding_friction(model, friction_geoms, friction)
        mujoco.mj_forward(model, data)

        window: deque[dict] = deque(maxlen=WINDOW_SIZE)
        episode_failed = False
        params = {
            "episode": episode,
            "friction": friction,
            "lateral_force": lateral_force,
            "impulse_start": IMPULSE_START,
            "impulse_end": IMPULSE_END,
            "torso_body_id": int(torso_body_id),
            "floor_geom_id": int(floor_id),
        }

        for step in range(1, MAX_STEPS + 1):
            apply_pd(model, data, target_qpos)
            apply_torso_impulse(data, torso_body_id, step, lateral_force)
            mujoco.mj_step(model, data)
            z = float(data.qpos[2])
            window.append(snapshot(step, data, z))

            if z < FALL_HEIGHT:
                failed_episodes.append(
                    {
                        **params,
                        "fail_step": step,
                        "fail_time": float(data.time),
                        "final_z": z,
                        "impact_phase": impact_phase(step),
                        "window": list(window),
                    }
                )
                failed_friction.append(friction)
                failed_force.append(lateral_force)
                episode_failed = True
                break

        if not episode_failed:
            survived_friction.append(friction)
            survived_force.append(lateral_force)

    n_failures = len(failed_episodes)
    n_survived = N_EPISODES - n_failures
    failure_rate = n_failures / N_EPISODES
    failed_friction_stats = value_stats(failed_friction)
    failed_force_stats = value_stats(failed_force)
    survived_friction_stats = value_stats(survived_friction)
    survived_force_stats = value_stats(survived_force)

    print(f"всего прогонов: {N_EPISODES}")
    print(f"упало: {n_failures}")
    print(f"устояло: {n_survived}")
    print(f"процент сбоев: {failure_rate * 100:.1f}%")
    print(f"упавшие, трение: {format_stats(failed_friction_stats)}")
    print(f"упавшие, боковая сила: {format_stats(failed_force_stats)}")
    print(f"устоявшие, трение: {format_stats(survived_friction_stats)}")
    print(f"устоявшие, боковая сила: {format_stats(survived_force_stats)}")

    report = {
        "n_episodes": N_EPISODES,
        "n_failures": n_failures,
        "n_survived": n_survived,
        "failure_rate": failure_rate,
        "kp": KP,
        "kd": KD,
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
    main()
