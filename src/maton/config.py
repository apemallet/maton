"""Strict, deliberately small TOML configuration surface."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from maton.algorithms import ALGORITHMS, algorithm as lookup
from maton.walker import WalkerConfig, default_walker
from maton.curriculum import CurriculumConfig


@dataclass
class Config:
    algorithm: str
    seed: int = 0
    training: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    walker: dict[str, Any] = field(default_factory=lambda: default_walker().to_dict())
    curriculum: dict[str, Any] = field(default_factory=dict)
    trainer: dict[str, Any] = field(default_factory=dict)  # the selected algorithm's table

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        from maton.jobs import Job
        return Job.load(path).config()

    @classmethod
    def from_values(cls, algorithm: str, settings: dict[str, Any], path: Path) -> "Config":
        values = dict(settings)
        trainer = values.pop(algorithm, {})
        walker_reference = values.get("walker")
        if not isinstance(walker_reference, str):
            raise ValueError("Job TOML must reference a walker file")
        values["walker"] = WalkerConfig.load(Path(path).resolve().parent / walker_reference).to_dict()
        reference = values.get("curriculum")
        if not isinstance(reference, str):
            raise ValueError("Job TOML must reference a curriculum file")
        values["curriculum"] = CurriculumConfig.load(Path(path).resolve().parent / reference).to_dict()
        config = cls(algorithm, trainer=trainer, **values)
        config.resolved()  # Reject misspelled and invalid settings before training.
        return config

    def trainer_config(self):
        return lookup(self.algorithm).load().Config(**self.trainer)

    def resolved(self) -> dict[str, Any]:
        from maton.env import EnvConfig
        from maton.training import TrainingConfig

        return {"algorithm": self.algorithm, "seed": self.seed,
                "training": asdict(TrainingConfig(**self.training)),
                "env": asdict(EnvConfig(**self.env)),
                "curriculum": CurriculumConfig.from_dict(self.curriculum).to_dict(),
                "walker": WalkerConfig.from_dict(self.walker).to_dict(),
                self.algorithm: asdict(self.trainer_config())}

    def environment(self):
        from maton.env import EnvConfig, WalkerEnv

        return WalkerEnv(EnvConfig(**self.env), curriculum=CurriculumConfig.from_dict(self.curriculum),
                         walker=WalkerConfig.from_dict(self.walker))


def save_run(config: Config, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(json.dumps(config.resolved(), indent=2) + "\n")


def load_run(output: Path) -> Config:
    values = json.loads((output / "run.json").read_text())
    algorithm = values.pop("algorithm")
    trainer = values.pop(algorithm, {})
    for name in ALGORITHMS:  # Older snapshots stored every trainer's defaults.
        values.pop(name, None)
    return Config(algorithm, trainer=trainer, **values)
