import json
from collections import deque
from pathlib import Path

import numpy as np
import mujoco

MODEL_PATH = r"C:\Users\HP\Desktop\mujoco_menagerie-main\mujoco_menagerie-main\unitree_go2\scene.xml"
N_EPISODES = 40
MAX_STEPS = 1500
WINDOW_SIZE = 200
FALL_HEIGHT = 0.22
FRICTION_RANGE = (0.1, 1.0)
HEIGHT_RANGE = (0.25, 0.42)
REPORT_PATH = Path("reports") / "batch_failures.json"


def find_floor_geom_id(model: mujoco.MjModel) -> int:
    for geom_id in range(model.ngeom):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or "").lower()
        if name in {"floor", "ground", "groundplane"}:
            return geom_id
    for geom_id in range(model.ngeom):
        if int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_PLANE):
            return geom_id
    return 0


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


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    floor_id = find_floor_geom_id(model)
    rng = np.random.default_rng()

    failed_episodes: list[dict] = []
    failed_friction: list[float] = []
    failed_height: list[float] = []
    survived_friction: list[float] = []
    survived_height: list[float] = []

    for episode in range(1, N_EPISODES + 1):
        mujoco.mj_resetData(model, data)

        friction = float(rng.uniform(*FRICTION_RANGE))
        initial_height = float(rng.uniform(*HEIGHT_RANGE))
        model.geom_friction[floor_id, 0] = friction
        data.qpos[2] = initial_height
        mujoco.mj_forward(model, data)

        window: deque[dict] = deque(maxlen=WINDOW_SIZE)
        episode_failed = False
        params = {
            "episode": episode,
            "friction": friction,
            "initial_height": initial_height,
            "floor_geom_id": int(floor_id),
        }

        for step in range(1, MAX_STEPS + 1):
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
                        "window": list(window),
                    }
                )
                failed_friction.append(friction)
                failed_height.append(initial_height)
                episode_failed = True
                break

        if not episode_failed:
            survived_friction.append(friction)
            survived_height.append(initial_height)

    n_failures = len(failed_episodes)
    failure_rate = n_failures / N_EPISODES
    failed_friction_stats = value_stats(failed_friction)
    failed_height_stats = value_stats(failed_height)
    survived_friction_stats = value_stats(survived_friction)
    survived_height_stats = value_stats(survived_height)

    print(f"всего прогонов: {N_EPISODES}")
    print(f"сбоев: {n_failures}")
    print(f"процент сбоев: {failure_rate * 100:.1f}%")
    print(f"упавшие, трение: {format_stats(failed_friction_stats)}")
    print(f"упавшие, начальная высота: {format_stats(failed_height_stats)}")
    print(f"неупавшие, трение: {format_stats(survived_friction_stats)}")
    print(f"неупавшие, начальная высота: {format_stats(survived_height_stats)}")

    report = {
        "n_episodes": N_EPISODES,
        "n_failures": n_failures,
        "failure_rate": failure_rate,
        "floor_geom_id": int(floor_id),
        "failed": {
            "friction": failed_friction_stats,
            "initial_height": failed_height_stats,
        },
        "survived": {
            "friction": survived_friction_stats,
            "initial_height": survived_height_stats,
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
