"""Course definitions and cumulative, success-gated level progression."""

from dataclasses import asdict, dataclass
from pathlib import Path
import tomllib
from typing import Any

from maton.terrain import TerrainConfig


@dataclass(frozen=True)
class Level:
    name: str
    successes_required: int
    terrain: TerrainConfig

    def __post_init__(self):
        if not self.name:
            raise ValueError("Curriculum levels need names")
        if type(self.successes_required) is not int or self.successes_required < 1:
            raise ValueError("successes_required must be a positive integer")


@dataclass(frozen=True)
class CurriculumConfig:
    levels: tuple[Level, ...]

    def __post_init__(self):
        if not self.levels:
            raise ValueError("Curriculum needs at least one level")
        topology = {(x.terrain.slots, x.terrain.spacing, x.terrain.start_length) for x in self.levels}
        if len(topology) != 1:
            raise ValueError("Levels must share slots, spacing, and start_length for fixed MJX shapes")
        if len({x.name for x in self.levels}) != len(self.levels):
            raise ValueError("Curriculum level names must be unique")
        first = self.levels[0].terrain
        if len(self.levels) > 1 and (first.block_frequency or first.gap_frequency):
            raise ValueError("The first curriculum level must be flat")

    @classmethod
    def load(cls, path: Path) -> "CurriculumConfig":
        with path.open("rb") as handle:
            return cls.from_dict(tomllib.load(handle))

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "CurriculumConfig":
        if values.keys() - {"levels", "terrain", "successes_required"}:
            raise ValueError("Unknown curriculum settings")
        shared = values.get("terrain", {})
        required = values.get("successes_required", 5)
        levels = []
        for entry in values.get("levels", []):
            if entry.keys() - {"name", "successes_required", "terrain"}:
                raise ValueError("Unknown curriculum level settings")
            levels.append(Level(entry["name"], entry.get("successes_required", required),
                                TerrainConfig(**(shared | entry.get("terrain", {})))))
        return cls(tuple(levels))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CurriculumProgress:
    config: CurriculumConfig
    level: int = 0
    successes: int = 0
    complete: bool = False

    def record(self, success: bool, episode_level: int) -> bool:
        """Only completions from the currently assigned level count toward promotion."""
        if self.complete or not success or episode_level != self.level:
            return False
        self.successes += 1
        if self.successes < self.config.levels[self.level].successes_required:
            return False
        if self.level == len(self.config.levels) - 1:
            self.complete = True
            return False
        self.level += 1
        self.successes = 0
        return True

    def snapshot(self) -> dict[str, Any]:
        return dict(level=self.level, name=self.config.levels[self.level].name,
                    successes=self.successes,
                    successes_required=self.config.levels[self.level].successes_required,
                    complete=self.complete)
