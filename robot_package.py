"""Versioned robot package: build a profile and reject it before a batch starts."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

from robot_profile import RobotProfile, urdf_motion_joint_names

PACKAGE_VERSION = 1
TASKS = ("stance", "track_ee")


class PackageError(ValueError):
    """The package is not ready to run. The batch must not start."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        super().__init__("\n".join(self.problems))


def build_package(
    model_path: str | Path,
    task: str | None = None,
    *,
    target_height: float | None = None,
    ee_tracking_limit_m: float | None = None,
    saturation_frac_limit: float | None = None,
    kp: float | None = None,
    kd: float | None = None,
) -> dict:
    """Profile JSON, package_version 1, after the model and the task fields check out."""
    path = Path(model_path)
    if not path.is_file():
        raise PackageError([f"Model file not found: {path}"])
    _raise_if_meshes_missing(path)
    model = _compile_model(path)
    is_urdf = path.suffix.lower() == ".urdf"
    floating = _is_floating(model)
    chosen = _choose_task(task, floating, is_urdf)
    problems: list[str] = []
    if chosen == "stance" and not floating:
        problems.append("task stance needs a floating base (a free joint). This model is fixed.")
    actuators = _actuator_names(model)
    if is_urdf:
        joints = urdf_motion_joint_names(path)
    else:
        joints = actuators
        if not joints:
            problems.append(
                f"No actuators found in {path.name}. A URDF without motors can still run; an MJCF needs actuator names."
            )
    if chosen == "stance":
        height = _stance_height(model, target_height, floating)
        if height is None:
            problems.append("task stance needs target_height in meters.")
            height = 0.0
    else:
        height = float(target_height) if target_height is not None else _body_height_guess(model)
        if ee_tracking_limit_m is None:
            ee_tracking_limit_m = 0.05
        if saturation_frac_limit is None:
            saturation_frac_limit = 0.9
        if float(ee_tracking_limit_m) <= 0:
            problems.append("task track_ee needs ee_tracking_limit_m greater than 0.")
        if not 0.0 < float(saturation_frac_limit) <= 1.0:
            problems.append("task track_ee needs saturation_frac_limit between 0 and 1.")
    _reject_gains(problems, kp, kd)
    if problems:
        raise PackageError(problems)
    payload: dict = {
        "package_version": PACKAGE_VERSION,
        "name": path.stem,
        "model_path": str(path.resolve()),
        "task": chosen,
        "robot_type": "legged" if floating else "manipulator",
        "joint_names": joints,
        "num_motors": len(joints),
        "target_height": float(height),
        "fall_height": 0.2,
        "contact_z": 0.12,
        "tilt_fail_deg": 45,
        "impulse_mode": "base" if chosen == "stance" else "none",
        "use_global_outcome_thresholds": False,
    }
    if chosen == "track_ee":
        payload["ee_tracking_limit_m"] = float(ee_tracking_limit_m)
        payload["saturation_frac_limit"] = float(saturation_frac_limit)
    if kp is not None:
        payload["kp"] = float(kp)
    if kd is not None:
        payload["kd"] = float(kd)
    return payload


def validate_profile(profile: RobotProfile) -> None:
    """Stop before the batch if the model, meshes, actuators, or task fields are wrong."""
    path = Path(profile.model_path)
    problems: list[str] = []
    if not path.is_file():
        raise PackageError([f"Model file not found: {path}"])
    problems.extend(_mesh_problems(path))
    if problems:
        raise PackageError(problems)
    model = _compile_model(path)
    is_urdf = path.suffix.lower() == ".urdf"
    floating = _is_floating(model)
    if profile.task == "stance" and not floating:
        problems.append("task stance needs a floating base (a free joint). This model is fixed.")
    if profile.package_version >= 1 and profile.task == "stance" and "target_height" not in profile.raw:
        problems.append("task stance needs target_height in meters.")
    if profile.package_version >= 1 and profile.task == "track_ee":
        if "ee_tracking_limit_m" not in profile.raw:
            problems.append("task track_ee needs ee_tracking_limit_m.")
        elif profile.ee_tracking_limit_m is None or float(profile.ee_tracking_limit_m) <= 0:
            problems.append("task track_ee needs ee_tracking_limit_m greater than 0.")
        if "saturation_frac_limit" not in profile.raw:
            problems.append("task track_ee needs saturation_frac_limit.")
        elif profile.saturation_frac_limit is None or not 0.0 < float(profile.saturation_frac_limit) <= 1.0:
            problems.append("task track_ee needs saturation_frac_limit between 0 and 1.")
    _reject_gains(problems, profile.kp, profile.kd)
    actuators = _actuator_names(model)
    if is_urdf and int(model.nu) == 0:
        if not urdf_motion_joint_names(path):
            problems.append(f"URDF {path.name} has no revolute, continuous, or prismatic joints.")
    elif not actuators:
        problems.append(f"No actuators found in {path.name}.")
    else:
        missing = [name for name in profile.joint_names if name not in actuators]
        if missing:
            shown = ", ".join(missing[:8])
            problems.append(f"Actuators listed in the profile are missing from the model: {shown}")
        if int(profile.num_motors) != len(actuators):
            problems.append(
                f"num_motors={profile.num_motors} does not match actuator count ({len(actuators)})."
            )
    if problems:
        raise PackageError(problems)


def _choose_task(task: str | None, floating: bool, is_urdf: bool) -> str:
    if task is None:
        return "stance" if floating and not is_urdf else "track_ee"
    chosen = str(task).strip().lower()
    if chosen not in TASKS:
        raise PackageError([f"task must be one of {list(TASKS)}, got {task!r}"])
    return chosen


def _reject_gains(problems: list[str], kp: float | None, kd: float | None) -> None:
    if kp is not None and float(kp) <= 0:
        problems.append("kp must be greater than 0.")
    if kd is not None and float(kd) <= 0:
        problems.append("kd must be greater than 0.")


def _is_floating(model: mujoco.MjModel) -> bool:
    return model.njnt > 0 and int(model.jnt_type[0]) == int(mujoco.mjtJoint.mjJNT_FREE)


def _actuator_names(model: mujoco.MjModel) -> list[str]:
    names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index) or ""
        for index in range(int(model.nu))
    ]
    return [name for name in names if name]


def _stance_height(model: mujoco.MjModel, explicit: float | None, floating: bool) -> float | None:
    if explicit is not None:
        height = float(explicit)
        return height if height > 0 else None
    if not floating:
        return None
    if int(model.nkey) > 0 and int(model.nq) > 2:
        height = float(model.key_qpos[2])
        if height > 1e-6:
            return height
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    if int(model.nq) > 2 and float(data.qpos[2]) > 1e-6:
        return float(data.qpos[2])
    return None


def _body_height_guess(model: mujoco.MjModel) -> float:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    if int(model.nbody) > 1:
        return float(data.xpos[int(model.nbody) - 1, 2])
    return 0.5


def _compile_model(path: Path) -> mujoco.MjModel:
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    except Exception as exc:
        raise PackageError([f"Could not compile {path.name}: {exc}"]) from exc


def _raise_if_meshes_missing(path: Path) -> None:
    problems = _mesh_problems(path)
    if problems:
        raise PackageError(problems)


def _mesh_problems(path: Path) -> list[str]:
    missing: list[str] = []
    seen: set[Path] = set()
    _walk_meshes(path.resolve(), missing, seen)
    return missing


def _walk_meshes(path: Path, missing: list[str], seen: set[Path]) -> None:
    if path in seen:
        return
    seen.add(path)
    if not path.is_file():
        missing.append(f"Included model file not found: {path}")
        return
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        missing.append(f"Could not parse {path.name}: {exc}")
        return
    meshdir = ""
    for element in root.iter():
        if _local(element.tag) == "compiler" and element.get("meshdir"):
            meshdir = element.get("meshdir") or ""
    for element in root.iter():
        tag = _local(element.tag)
        if tag == "include" and element.get("file"):
            _walk_meshes(_resolve_file(path, "", element.get("file") or ""), missing, seen)
        elif tag == "mesh":
            filename = element.get("file") or element.get("filename")
            if filename:
                _expect_file(path, meshdir, filename, missing)


def _expect_file(xml_path: Path, meshdir: str, filename: str, missing: list[str]) -> None:
    candidate = _resolve_file(xml_path, meshdir, filename)
    if not candidate.is_file():
        missing.append(f"Mesh file not found: {candidate}")


def _resolve_file(xml_path: Path, meshdir: str, filename: str) -> Path:
    name = Path(filename)
    if name.is_absolute():
        return name
    base = xml_path.parent
    if meshdir:
        return base / meshdir / name
    return base / name


def _local(tag: str) -> str:
    return tag.split("}")[-1]
