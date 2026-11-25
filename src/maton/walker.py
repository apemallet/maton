"""Validated planar body trees and their MuJoCo assembly."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import tomllib
from xml.etree import ElementTree as ET


def _vector(value, size: int, label: str) -> None:
    if len(value) != size or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in value):
        raise ValueError(f"{label} must contain {size} finite numbers")


def _name(value: str) -> None:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", value) or value.startswith("root_"):
        raise ValueError(f"Invalid or reserved walker name: {value!r}")


@dataclass(frozen=True)
class Motor:
    torque_limit: float = 20.0
    kp: float = 30.0
    kd: float = 2.0
    target_range: tuple[float, float] = (-1.0, 1.0)
    neutral: float | None = None

    def __post_init__(self):
        _vector(self.target_range, 2, "target_range")
        if not all(math.isfinite(x) for x in (self.torque_limit, self.kp, self.kd)):
            raise ValueError("Motor parameters must be finite")
        if self.torque_limit <= 0 or self.kp <= 0 or self.kd < 0:
            raise ValueError("Invalid motor torque limit or gains")
        low, high = self.target_range
        if low >= high or (self.neutral is not None and not low <= self.neutral <= high):
            raise ValueError("Invalid target range or neutral")


@dataclass(frozen=True)
class Joint:
    name: str
    range: tuple[float, float]
    damping: float = 0.5
    armature: float = 0.02
    motor: Motor | None = None

    def __post_init__(self):
        _name(self.name)
        _vector(self.range, 2, "joint range")
        if not self.range[0] < self.range[1] or not self.range[0] <= 0 <= self.range[1]:
            raise ValueError("Joint range must be ordered and contain the initial angle zero")
        if not all(math.isfinite(x) and x >= 0 for x in (self.damping, self.armature)):
            raise ValueError("Joint damping and armature must be finite and nonnegative")
        if self.motor and not self.range[0] <= self.motor.target_range[0] < self.motor.target_range[1] <= self.range[1]:
            raise ValueError("Motor target range must lie inside physical joint range")


@dataclass(frozen=True)
class Geom:
    type: str
    size: tuple[float, ...]
    mass: float
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    fromto: tuple[float, ...] | None = None
    contact: bool = False
    rgba: tuple[float, float, float, float] = (0.2, 0.35, 0.8, 1.0)

    def __post_init__(self):
        dimensions = {"sphere": 1, "capsule": 1, "box": 3}
        if self.type not in dimensions:
            raise ValueError("Geometry type must be sphere, capsule, or box")
        _vector(self.size, dimensions[self.type], "geom size")
        _vector(self.pos, 3, "geom pos")
        _vector(self.rgba, 4, "geom rgba")
        if any(x <= 0 for x in self.size) or not math.isfinite(self.mass) or self.mass <= 0:
            raise ValueError("Geometry sizes and mass must be positive")
        if any(not 0 <= x <= 1 for x in self.rgba) or type(self.contact) is not bool:
            raise ValueError("Invalid geometry color or contact flag")
        if self.type == "capsule":
            if self.fromto is None:
                raise ValueError("Capsules require fromto endpoints")
            _vector(self.fromto, 6, "capsule fromto")
            if self.fromto[:3] == self.fromto[3:]:
                raise ValueError("Capsule endpoints must differ")
        elif self.fromto is not None:
            raise ValueError("Only capsules accept fromto")


@dataclass(frozen=True)
class Body:
    name: str
    geoms: tuple[Geom, ...]
    parent: str | None = None
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    joint: Joint | None = None

    def __post_init__(self):
        _name(self.name)
        _vector(self.pos, 3, "body pos")
        if not self.geoms:
            raise ValueError("Each body requires at least one mass-bearing geom")


@dataclass(frozen=True)
class WalkerConfig:
    name: str
    bodies: tuple[Body, ...]
    spawn_height: float = 1.0

    def __post_init__(self):
        _name(self.name)
        if not math.isfinite(self.spawn_height) or self.spawn_height <= 0:
            raise ValueError("spawn_height must be finite and positive")
        names = [b.name for b in self.bodies]
        joints = [b.joint.name for b in self.bodies if b.joint]
        if len(set(names)) != len(names) or len(set(joints)) != len(joints):
            raise ValueError("Body and joint names must be unique")
        roots = [b for b in self.bodies if b.parent is None]
        if len(roots) != 1 or roots[0].joint is not None or any(roots[0].pos):
            raise ValueError("Require one root body with zero pos and no joint; use spawn_height")
        by_name = {b.name: b for b in self.bodies}
        for body in self.bodies:
            visited = {body.name}
            parent = body.parent
            while parent is not None:
                if parent not in by_name or parent in visited:
                    raise ValueError("Body parents must form a connected, acyclic tree")
                visited.add(parent)
                parent = by_name[parent].parent
        if not self.motors:
            raise ValueError("Walker requires at least one motor")

    @classmethod
    def load(cls, path: str | Path) -> "WalkerConfig":
        with Path(path).open("rb") as handle:
            return cls.from_dict(tomllib.load(handle))

    @classmethod
    def from_dict(cls, values: dict) -> "WalkerConfig":
        values = dict(values)
        bodies = []
        for entry in values.pop("bodies"):
            entry = dict(entry)
            entry["geoms"] = tuple(Geom(**g) for g in entry["geoms"])
            if entry.get("joint") is not None:
                joint = dict(entry["joint"])
                if joint.get("motor") is not None:
                    joint["motor"] = Motor(**joint["motor"])
                entry["joint"] = Joint(**joint)
            bodies.append(Body(**entry))
        return cls(bodies=tuple(bodies), **values)

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self)))

    @property
    def ordered_bodies(self) -> tuple[Body, ...]:
        def visit(parent):
            return tuple(child for body in self.bodies if body.parent == parent
                         for child in (body, *visit(body.name)))
        return visit(None)

    @property
    def motors(self) -> tuple[Joint, ...]:
        return tuple(b.joint for b in self.ordered_bodies if b.joint and b.joint.motor)

    def assemble(self, world: ET.Element, actuators: ET.Element, sensors: ET.Element) -> None:
        """Emit bodies in depth-first order; motor/action order follows this traversal."""
        elements = {}
        for body in self.ordered_bodies:
            pos = body.pos if body.parent else (0, 0, self.spawn_height)
            element = ET.SubElement(world if body.parent is None else elements[body.parent], "body",
                                    name=body.name, pos=_numbers(pos))
            elements[body.name] = element
            if body.parent is None:
                for name, kind, axis, damping in [("x", "slide", "1 0 0", "0"),
                                                  ("z", "slide", "0 0 1", "0"),
                                                  ("pitch", "hinge", "0 1 0", "0.1")]:
                    ET.SubElement(element, "joint", name=f"root_{name}", type=kind, axis=axis,
                                  damping=damping, armature="0.02", limited="false")
            if body.joint:
                joint = body.joint
                ET.SubElement(element, "joint", name=joint.name, type="hinge", axis="0 1 0",
                              range=_numbers(joint.range), damping=str(joint.damping), armature=str(joint.armature))
                if joint.motor:
                    ET.SubElement(actuators, "motor", joint=joint.name, gear=str(joint.motor.torque_limit),
                                  ctrllimited="true", ctrlrange="-1 1")
            for index, geom in enumerate(body.geoms):
                shape = dict(type=geom.type, size=_numbers(geom.size))
                if geom.fromto is not None:
                    shape['fromto'] = _numbers(geom.fromto)
                else:
                    shape['pos'] = _numbers(geom.pos)
                ET.SubElement(element, "geom", {**shape, "mass": str(geom.mass), "rgba": _numbers(geom.rgba)})
                if geom.contact:
                    name = f"touch_{body.name}_{index}"
                    ET.SubElement(element, "site", {**shape, "name": name, "size": _numbers(tuple(s * 1.13 for s in geom.size))})
                    ET.SubElement(sensors, "touch", name=f"contact_{body.name}_{index}", site=name)


def _numbers(values) -> str:
    return " ".join(str(x) for x in values)


def default_walker() -> WalkerConfig:
    return WalkerConfig.load(Path(__file__).resolve().parents[2] / "configs" / "walkers" / "basic.toml")
