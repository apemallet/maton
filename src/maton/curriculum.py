"""Course definitions and success-rate-gated level progression."""

from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import tomllib
from typing import Any

from maton.terrain import TerrainConfig


@dataclass(frozen=True)
class Level:
    """Promote after `successes_required` successes among the last `window` episodes here."""
    name: str
    successes_required: int
    terrain: TerrainConfig
    window: int = 10
    efficiency_penalty: float = 0.0  # reward per J/(body mass * gravity), active throughout this level

    def __post_init__(self):
        if not math.isfinite(self.efficiency_penalty) or self.efficiency_penalty < 0:
            raise ValueError("efficiency_penalty must be finite and nonnegative")
        if not self.name:
            raise ValueError("Curriculum levels need names")
        if type(self.successes_required) is not int or self.successes_required < 1:
            raise ValueError("successes_required must be a positive integer")
        if type(self.window) is not int or self.window < self.successes_required:
            raise ValueError("window must be an integer of at least successes_required")


@dataclass(frozen=True)
class CurriculumConfig:
    levels: tuple[Level, ...]

    def __post_init__(self):
        if not self.levels:
            raise ValueError("Curriculum needs at least one level")
        topology = {(x.terrain.slots, x.terrain.spacing, x.terrain.start_length, x.terrain.run_out)
                    for x in self.levels}
        if len(topology) != 1:
            raise ValueError("Levels must share slots, spacing, start_length, and run_out (fixed MJX shapes)")
        if len({x.name for x in self.levels}) != len(self.levels):
            raise ValueError("Curriculum level names must be unique")

    @classmethod
    def load(cls, path: Path) -> "CurriculumConfig":
        with path.open("rb") as handle:
            return cls.from_dict(tomllib.load(handle))

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CurriculumConfig":
        if values.keys() - {"levels", "terrain", "successes_required", "window"}:
            raise ValueError("Unknown curriculum settings")
        shared = values.get("terrain", {})
        required, window = values.get("successes_required", 4), values.get("window", 10)
        levels = []
        for entry in values.get("levels", []):
            if entry.keys() - {"name", "successes_required", "window", "terrain", "efficiency_penalty"}:
                raise ValueError("Unknown curriculum level settings")
            levels.append(Level(entry["name"], entry.get("successes_required", required),
                                TerrainConfig(**(shared | entry.get("terrain", {}))),
                                entry.get("window", window), entry.get("efficiency_penalty", 0.0)))
        return cls(tuple(levels))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CurriculumProgress:
    config: CurriculumConfig
    level: int = 0
    recent: list[bool] = field(default_factory=list)  # outcomes at this level, oldest first
    complete: bool = False

    @property
    def efficiency_weight(self) -> float:
        """The active level defines its reward, independently of curriculum completion."""
        return self.config.levels[self.level].efficiency_penalty

    @property
    def successes(self) -> int:
        return sum(self.recent)

    def record(self, success: bool, episode_level: int) -> bool:
        """Count an episode toward promotion; only episodes from the current level count."""
        if self.complete or episode_level != self.level:
            return False
        level = self.config.levels[self.level]
        self.recent = (self.recent + [bool(success)])[-level.window:]
        if self.successes < level.successes_required:
            return False
        if self.level == len(self.config.levels) - 1:
            self.complete = True
            return False
        self.level += 1
        self.recent = []
        return True

    def state(self) -> dict[str, Any]:
        """Exact progress for resumable training."""
        return dict(level=self.level, recent=list(self.recent), complete=self.complete)

    def restore(self, state: dict[str, Any]) -> None:
        self.level, self.recent, self.complete = state["level"], list(state["recent"]), state["complete"]

    def snapshot(self) -> dict[str, Any]:
        level = self.config.levels[self.level]
        return dict(level=self.level, name=level.name, successes=self.successes,
                    successes_required=level.successes_required, window=level.window,
                    complete=self.complete, efficiency_weight=self.efficiency_weight)
