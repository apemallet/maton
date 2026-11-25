"""Strict, deliberately small TOML configuration surface."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from maton.walker import WalkerConfig, default_walker
from maton.curriculum import CurriculumConfig


@dataclass
class Config:
    seed: int = 0
    training: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    walker: dict[str, Any] = field(default_factory=lambda: default_walker().to_dict())
    curriculum: dict[str, Any] = field(default_factory=dict)
    ppo: dict[str, Any] = field(default_factory=dict)
    dreamer: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        from maton.jobs import Job
        return Job.load(path).config()

    @classmethod
    def from_values(cls, settings: dict[str, Any], path: Path) -> "Config":
        values = dict(settings)
        walker_reference = values.get("walker")
        if not isinstance(walker_reference, str):
            raise ValueError("Job TOML must reference a walker file")
        values["walker"] = WalkerConfig.load(Path(path).resolve().parent / walker_reference).to_dict()
        reference = values.get("curriculum")
        if not isinstance(reference, str):
            raise ValueError("Job TOML must reference a curriculum file")
        values["curriculum"] = CurriculumConfig.load(Path(path).resolve().parent / reference).to_dict()
        config = cls(**values)
        config.resolved()  # Reject misspelled and invalid settings before training.
        return config

    def resolved(self) -> dict[str, Any]:
        from maton.env import EnvConfig
        from maton.ppo import PPOConfig
        from maton.dreamer import DreamerConfig
        from maton.training import TrainingConfig

        training = TrainingConfig(**self.training)
        ppo = PPOConfig(**self.ppo)
        return dict(seed=self.seed, training=asdict(training), env=asdict(EnvConfig(**self.env)),
                    curriculum=CurriculumConfig.from_dict(self.curriculum).to_dict(),
                    walker=WalkerConfig.from_dict(self.walker).to_dict(),
                    ppo=asdict(ppo),
                    dreamer=asdict(DreamerConfig(**self.dreamer)))

    def environment(self):
        from maton.env import EnvConfig, WalkerEnv

        return WalkerEnv(EnvConfig(**self.env), curriculum=CurriculumConfig.from_dict(self.curriculum),
                         walker=WalkerConfig.from_dict(self.walker))


def save_run(config: Config, algorithm: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(json.dumps(
        dict(algorithm=algorithm, **config.resolved()), indent=2) + "\n")


def load_run(output: Path) -> tuple[Config, str]:
    values = json.loads((output / "run.json").read_text())
    algorithm = values.pop("algorithm")
    return Config(**values), algorithm
