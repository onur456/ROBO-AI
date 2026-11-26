"""Robot-specific parameters loaded from a JSON profile."""

from __future__ import annotations

import json
from pathlib import Path


def resolve_config_path(path: str | Path) -> Path:
    """Find a profile JSON from the given path or next to this project."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    rooted = Path(__file__).resolve().parent / candidate
    if rooted.is_file():
        return rooted
    return candidate


class RobotProfile:
    """Go2-style robot description: model, motors, stance height, fall limits."""

    def __init__(self, config_path: str | Path) -> None:
        self.config_path = resolve_config_path(config_path)
        if not self.config_path.is_file():
            raise FileNotFoundError(f"robot profile not found: {config_path}")
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.model_path = str(raw["model_path"])
        self.joint_names = [str(name) for name in raw["joint_names"]]
        self.num_motors = int(raw["num_motors"])
        self.target_height = float(raw["target_height"])
        self.fall_height = float(raw["fall_height"])
        self.contact_z = float(raw["contact_z"])
        self.tilt_fail_deg = float(raw["tilt_fail_deg"])
        self.hip_actuator = str(raw.get("hip_actuator") or self.joint_names[0])
        self.calf_actuator = str(raw.get("calf_actuator") or self.joint_names[2])
        thresholds = raw.get("outcome_thresholds") or {}
        self.outcome_thresholds = dict(thresholds)
        self.z_threshold = thresholds.get("z_threshold")
        self.slip_threshold = thresholds.get("slip_threshold")
