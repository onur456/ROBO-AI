import json
from collections import deque
from pathlib import Path

import mujoco

MODEL_PATH = r"C:\Users\HP\Desktop\mujoco_menagerie-main\mujoco_menagerie-main\unitree_go2\scene.xml"
MAX_STEPS = 2000
WINDOW_SIZE = 200
FALL_HEIGHT = 0.22
REPORT_PATH = Path("reports") / "failure_window.json"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    window: deque[dict] = deque(maxlen=WINDOW_SIZE)
    failed = False

    for step in range(1, MAX_STEPS + 1):
        mujoco.mj_step(model, data)

        z = float(data.qpos[2])
        window.append(
            {
                "step": step,
                "time": float(data.time),
                "z": z,
                "qpos_head": data.qpos[:7].tolist(),
                "qvel_head": data.qvel[:6].tolist(),
                "ctrl": data.ctrl.tolist(),
            }
        )

        if z < FALL_HEIGHT:
            print(f"сбой на шаге {step}, time={data.time:.6f}")
            print(f"z={z:.6f}")
            print(f"записей в окне до сбоя: {len(window)}")
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text(
                json.dumps(list(window), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"буфер сохранён: {REPORT_PATH}")
            failed = True
            break

    if not failed:
        print("за 2000 шагов сбой не зафиксирован")


if __name__ == "__main__":
    main()
