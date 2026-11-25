"""Сменный интерфейс контроллера стойки Go2.

Команда на шаге считается здесь. Шум и смещение датчиков, удержание команды,
задержка и ограничение момента остаются снаружи, в цикле эпизода.

Спецификатор: ``pd``, ``stance_shift``, ``путь/к/файлу.py:Имя`` или ``пакет.модуль:Имя``.
В параллельные воркеры уходит строка, объект контроллера создаётся уже внутри процесса.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Protocol

import numpy as np

STANCE_SHIFT_RAD = 0.20
CONTROLLER_NAMES = ("pd", "stance_shift")
CONTROLLER_IMPORT_ROOT: str | None = None
_ATTR_CACHE: dict[tuple[str, str, int], object] = {}


class StanceController(Protocol):
    name: str

    def reset(self) -> None:
        """Сброс внутреннего состояния перед эпизодом."""

    def act(
        self,
        obs: dict,
        step: int,
        dt: float,
        target_qpos: np.ndarray,
    ) -> np.ndarray:
        """Команда актуаторам по наблюдению суставов и целевой стойке."""


class JointPDController:
    """Текущий суставной PD: kp * (q_target - q) - kd * v."""

    def __init__(self, kp: float, kd: float, name: str = "pd", q_shift: float = 0.0):
        self.kp = float(kp)
        self.kd = float(kd)
        self.name = name
        self.q_shift = float(q_shift)

    def reset(self) -> None:
        return None

    def act(
        self,
        obs: dict,
        step: int,
        dt: float,
        target_qpos: np.ndarray,
    ) -> np.ndarray:
        del step, dt
        q = np.asarray(obs["q"], dtype=float)
        v = np.asarray(obs["v"], dtype=float)
        target = np.asarray(target_qpos, dtype=float) + self.q_shift
        return self.kp * (target - q) - self.kd * v


class StanceShiftController(JointPDController):
    """Тот же PD, но целевая стойка сдвинута на постоянный угол."""

    def __init__(self, kp: float, kd: float, shift: float = STANCE_SHIFT_RAD):
        super().__init__(kp, kd, name="stance_shift", q_shift=float(shift))


class CallableController:
    """Обёртка функции compute(obs, step, dt, target_qpos)."""

    def __init__(self, fn, name: str, kp: float, kd: float):
        self.fn = fn
        self.name = name
        self.kp = float(kp)
        self.kd = float(kd)

    def reset(self) -> None:
        return None

    def act(
        self,
        obs: dict,
        step: int,
        dt: float,
        target_qpos: np.ndarray,
    ) -> np.ndarray:
        try:
            return self.fn(obs, step, dt, target_qpos)
        except TypeError as exc:
            raise TypeError(
                f"controller function {self.name!r} was not called as "
                f"compute(obs, step, dt, target_qpos): {exc}"
            ) from exc


def _spec_parts(spec: str) -> tuple[str, str | None]:
    text = str(spec).strip()
    if not text:
        raise ValueError(
            "empty controller spec: expected pd, stance_shift, "
            "path.py:Name, or module:Name"
        )
    if text in CONTROLLER_NAMES:
        return text, None
    if ":" not in text:
        known = ", ".join(CONTROLLER_NAMES)
        raise ValueError(
            f"invalid controller spec {text!r}: missing ':'. "
            f"Expected {known}, path.py:ClassOrFunction, or module:Attr"
        )
    location, attr = text.rsplit(":", 1)
    location = location.strip().strip('"').strip("'")
    attr = attr.strip()
    if not location or not attr:
        raise ValueError(
            f"invalid controller spec {text!r}: "
            "both a location and an attribute name are required (for example file.py:MyController)"
        )
    if not attr.isidentifier():
        raise ValueError(
            f"invalid controller attribute name {attr!r} in spec {text!r}"
        )
    return location, attr


def _looks_like_file(location: str) -> bool:
    if location.endswith(".py") or "/" in location or "\\" in location:
        return True
    return Path(location).expanduser().is_file()


def canonical_controller_spec(spec: str) -> str:
    """Встроенное имя, абсолютный путь.py:Атрибут или модуль:Атрибут."""
    location, attr = _spec_parts(spec)
    if attr is None:
        return location
    if _looks_like_file(location):
        path = Path(location).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.is_file():
            raise ValueError(
                f"controller file not found: {path} "
                f"(spec {location}:{attr})"
            )
        return f"{path.resolve()}:{attr}"
    return f"{location}:{attr}"


def _path_entry(entry: str) -> Path:
    if entry in ("", "."):
        return Path.cwd()
    return Path(entry)


def locate_module_root(module_name: str) -> str:
    """Каталог из sys.path, из которого импортируется модуль."""
    try:
        found = importlib.util.find_spec(module_name)
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"controller module not found: {module_name!r} ({exc}). "
            "Add the module directory to sys.path or pass path.py:Name"
        ) from exc
    origin = None if found is None else getattr(found, "origin", None)
    if found is None or not origin or origin == "namespace":
        raise ValueError(
            f"controller module not found: {module_name!r}. "
            "Add the module directory to sys.path or pass path.py:Name"
        )
    origin_path = Path(origin).resolve()
    matches: list[str] = []
    for entry in sys.path:
        try:
            entry_path = _path_entry(entry).resolve()
        except OSError:
            continue
        try:
            origin_path.relative_to(entry_path)
        except ValueError:
            continue
        matches.append(str(entry_path))
    if matches:
        matches.sort(key=len, reverse=True)
        return matches[0]
    parts = module_name.split(".")
    up = len(parts) if origin_path.name == "__init__.py" else len(parts) - 1
    return str(origin_path.parents[up])


def apply_import_root(root: str | None) -> None:
    """Добавляет каталог модуля в sys.path текущего процесса (в том числе воркера)."""
    global CONTROLLER_IMPORT_ROOT
    if not root:
        return
    folder = str(Path(root).resolve())
    CONTROLLER_IMPORT_ROOT = folder
    if folder not in sys.path:
        sys.path.insert(0, folder)


def prepare_controller_spec(spec: str) -> tuple[str, str | None]:
    """Канонический спецификатор и каталог импорта для воркеров."""
    global CONTROLLER_IMPORT_ROOT
    canonical = canonical_controller_spec(spec)
    location, attr = _spec_parts(canonical)
    if attr is None or _looks_like_file(location):
        CONTROLLER_IMPORT_ROOT = None
        return canonical, None
    root = locate_module_root(location)
    apply_import_root(root)
    module = importlib.import_module(location)
    if not hasattr(module, attr):
        raise ValueError(f"module {location!r} has no attribute {attr!r}")
    return canonical, root


def _load_attribute(location: str, attr: str):
    if _looks_like_file(location):
        path = Path(location).expanduser()
        if not path.is_file():
            raise ValueError(
                f"controller file not found: {path} "
                f"(spec {location}:{attr})"
            )
        resolved = path.resolve()
        mtime = int(resolved.stat().st_mtime_ns)
        key = (str(resolved), attr, mtime)
        cached = _ATTR_CACHE.get(key)
        if cached is not None:
            return cached
        module_name = f"external_controller_{abs(hash(key))}"
        loaded = importlib.util.spec_from_file_location(module_name, resolved)
        if loaded is None or loaded.loader is None:
            raise ValueError(f"could not read controller file: {resolved}")
        module = importlib.util.module_from_spec(loaded)
        try:
            loaded.loader.exec_module(module)
        except Exception as exc:
            raise ValueError(
                f"could not load controller file {resolved}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not hasattr(module, attr):
            raise ValueError(
                f"file {resolved} has no attribute {attr!r}"
            )
        obj = getattr(module, attr)
        _ATTR_CACHE[key] = obj
        return obj
    key = (location, attr, 0)
    cached = _ATTR_CACHE.get(key)
    if cached is not None:
        return cached
    apply_import_root(CONTROLLER_IMPORT_ROOT)
    try:
        module = importlib.import_module(location)
    except ModuleNotFoundError as exc:
        raise ValueError(
            f"controller module not found: {location!r} ({exc}). "
            "For a file, pass path.py:Name"
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"could not import controller module {location!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if not hasattr(module, attr):
        raise ValueError(f"module {location!r} has no attribute {attr!r}")
    obj = getattr(module, attr)
    _ATTR_CACHE[key] = obj
    return obj


def _is_policy_callable(fn) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return True
    params = [p for name, p in sig.parameters.items() if name != "self"]
    names = [
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if names and names[0] in {"obs", "observation", "state"}:
        return True
    required = [
        p
        for p in params
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(required) >= 4:
        return True
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params) and len(names) >= 4:
        return True
    return False


def _call_with_gains(fn, kp: float, kd: float, label: str):
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(kp, kd)
    params = [p for name, p in sig.parameters.items() if name != "self"]
    names = {p.name for p in params}
    required = [
        p.name
        for p in params
        if p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if "kp" in names and "kd" in names:
        return fn(kp=float(kp), kd=float(kd))
    if not required:
        return fn()
    unknown = [name for name in required if name not in {"kp", "kd"}]
    if unknown:
        raise ValueError(
            f"cannot construct controller {label}: required arguments {unknown}. "
            "A class or factory may accept only kp and kd"
        )
    kwargs = {}
    if "kp" in names:
        kwargs["kp"] = float(kp)
    if "kd" in names:
        kwargs["kd"] = float(kd)
    return fn(**kwargs)


def _build_external(obj, kp: float, kd: float, label: str):
    if inspect.isclass(obj):
        try:
            inst = _call_with_gains(obj, kp, kd, label)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"could not construct controller class {label}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not callable(getattr(inst, "act", None)):
            raise ValueError(
                f"class {label} was constructed, but the instance has no method "
                "act(obs, step, dt, target_qpos)"
            )
        if not getattr(inst, "name", None):
            inst.name = label
        return inst
    if callable(obj):
        if _is_policy_callable(obj):
            return CallableController(obj, label, kp, kd)
        try:
            produced = _call_with_gains(obj, kp, kd, label)
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(
                f"controller factory {label} did not return an object: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if callable(getattr(produced, "act", None)):
            if not getattr(produced, "name", None):
                try:
                    produced.name = label
                except Exception:
                    pass
            return produced
        raise ValueError(
            f"attribute {label} is a function, but it is not compute(obs, step, dt, target_qpos) "
            "and not a factory that returns an object with an act method"
        )
    if callable(getattr(obj, "act", None)):
        return obj
    raise ValueError(
        f"attribute {label} cannot be used as a controller: "
        "need a class with act, a compute(obs, step, dt, target_qpos) function, "
        "or a factory (kp, kd)"
    )


def make_controller(name: str, kp: float, kd: float):
    spec = canonical_controller_spec(name)
    if spec == "pd":
        return JointPDController(kp, kd, name="pd")
    if spec == "stance_shift":
        return StanceShiftController(kp, kd)
    location, attr = _spec_parts(spec)
    if attr is None:
        known = ", ".join(CONTROLLER_NAMES)
        raise ValueError(f"unknown controller: {spec!r} (expected {known})")
    if not _looks_like_file(location):
        apply_import_root(CONTROLLER_IMPORT_ROOT)
    obj = _load_attribute(location, attr)
    return _build_external(obj, kp, kd, f"{location}:{attr}")
