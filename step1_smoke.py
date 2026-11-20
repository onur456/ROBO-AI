import numpy as np
import mujoco

MODEL_PATH = r"C:\Users\HP\Desktop\mujoco_menagerie-main\mujoco_menagerie-main\unitree_go2\scene.xml"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    for step in range(1, 501):
        mujoco.mj_step(model, data)

        if step % 100 == 0:
            qpos6 = np.asarray(data.qpos[:6])
            print(f"step={step} time={data.time:.6f} qpos[:6]={qpos6}")

            if not np.isfinite(data.qpos).all():
                raise RuntimeError(f"NaN/inf detected in qpos at step {step}")


if __name__ == "__main__":
    main()
