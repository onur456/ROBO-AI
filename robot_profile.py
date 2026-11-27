"""Robot-specific parameters loaded from a JSON profile."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

_URDF_MOTION_TYPES = {"revolute", "continuous", "prismatic", "hinge", "slide"}
_ROBOT_TYPES = {"legged", "manipulator"}
_IMPULSE_MODES = {"base", "none", "ee"}


def resolve_config_path(path: str | Path) -> Path:
    """Find a profile JSON from the given path or next to this project."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    rooted = Path(__file__).resolve().parent / candidate
    if rooted.is_file():
        return rooted
    return candidate


def resolve_model_path(path: str, config_path: Path) -> str:
    """Find a model file from the JSON path, the project root, or the config folder."""
    candidate = Path(path)
    if candidate.is_file():
        return str(candidate)
    rooted = Path(__file__).resolve().parent / candidate
    if rooted.is_file():
        return str(rooted)
    beside = config_path.parent / candidate
    if beside.is_file():
        return str(beside)
    return str(candidate)


def urdf_motion_joint_names(urdf_path: str | Path) -> list[str]:
    """Hinge and slide joint names from a URDF, in file order."""
    path = Path(urdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"URDF model not found: {urdf_path}")
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse URDF {urdf_path}: {exc}") from exc
    names: list[str] = []
    for element in root.iter():
        tag = element.tag.split("}")[-1]
        if tag != "joint":
            continue
        joint_type = (element.get("type") or "fixed").lower()
        if joint_type not in _URDF_MOTION_TYPES:
            continue
        name = element.get("name")
        if name:
            names.append(name)
    if not names:
        raise ValueError(f"URDF {urdf_path} has no revolute, continuous, or prismatic joints")
    return names


class RobotProfile:
    """Robot description: model, motors, stance height, fall limits."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = resolve_config_path(config_path)
        if not self.config_path.is_file():
            raise FileNotFoundError(f"robot profile not found: {config_path}")
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.model_path = resolve_model_path(str(raw["model_path"]), self.config_path)
        is_urdf = self.model_path.lower().endswith(".urdf")
        listed = raw.get("joint_names")
        if listed:
            self.joint_names = [str(name) for name in listed]
        elif is_urdf:
            self.joint_names = urdf_motion_joint_names(self.model_path)
        else:
            raise KeyError("joint_names")
        if not self.joint_names:
            raise ValueError(f"robot profile {self.config_path} has no joints")
        if raw.get("num_motors") is None:
            self.num_motors = len(self.joint_names)
        else:
            self.num_motors = int(raw["num_motors"])
        self.target_height = _float_field(raw, "target_height", 0.5 if is_urdf else None)
        self.fall_height = _float_field(raw, "fall_height", 0.2 if is_urdf else None)
        self.contact_z = _float_field(raw, "contact_z", 0.12 if is_urdf else None)
        self.tilt_fail_deg = _float_field(raw, "tilt_fail_deg", 45.0 if is_urdf else None)
        self.hip_actuator = str(raw.get("hip_actuator") or self.joint_names[0])
        calf_index = 2 if len(self.joint_names) > 2 else len(self.joint_names) - 1
        self.calf_actuator = str(raw.get("calf_actuator") or self.joint_names[calf_index])
        self.robot_type = _robot_type(raw, is_urdf)
        self.use_global_outcome_thresholds = _use_global_thresholds(raw, self.config_path)
        self.impulse_mode = _impulse_mode(raw, self.robot_type)
        self.impulse_force_scale = _float_field(raw, "impulse_force_scale", 1.0)
        self.ee_tracking_limit_m = _optional_float(
            raw, "ee_tracking_limit_m", 0.05 if self.robot_type == "manipulator" else None
        )
        self.saturation_frac_limit = _optional_float(
            raw, "saturation_frac_limit", 0.9 if self.robot_type == "manipulator" else None
        )
        self.tau_over_limit_limit = _optional_float(raw, "tau_over_limit_limit", None)
        thresholds = raw.get("outcome_thresholds") or {}
        self.outcome_thresholds = dict(thresholds)
        self.z_threshold = thresholds.get("z_threshold")
        self.slip_threshold = thresholds.get("slip_threshold")


def _robot_type(raw: dict, is_urdf: bool) -> str:
    listed = raw.get("robot_type", raw.get("kind"))
    if listed is not None and str(listed).strip():
        kind = str(listed).strip().lower()
        if kind not in _ROBOT_TYPES:
            raise ValueError(
                f"robot_type must be one of {sorted(_ROBOT_TYPES)}, got {listed!r}"
            )
        return kind
    if is_urdf:
        return "manipulator"
    return "legged"


def _use_global_thresholds(raw: dict, config_path: Path) -> bool:
    """Go2's calibrated file is the default only for the Go2 profile."""
    if "use_global_outcome_thresholds" in raw and raw["use_global_outcome_thresholds"] is not None:
        return bool(raw["use_global_outcome_thresholds"])
    return config_path.stem.lower() == "go2"


def _impulse_mode(raw: dict, robot_type: str) -> str:
    listed = raw.get("impulse_mode")
    if listed is None or not str(listed).strip():
        return "none" if robot_type == "manipulator" else "base"
    mode = str(listed).strip().lower()
    if mode not in _IMPULSE_MODES:
        raise ValueError(
            f"impulse_mode must be one of {sorted(_IMPULSE_MODES)}, got {listed!r}"
        )
    return mode


def _optional_float(raw: dict, key: str, default: float | None) -> float | None:
    if key not in raw or raw[key] is None:
        return None if default is None else float(default)
    return float(raw[key])


def _float_field(raw: dict, key: str, default: float | None) -> float:
    if key in raw and raw[key] is not None:
        return float(raw[key])
    if default is None:
        raise KeyError(key)
    return float(default)
