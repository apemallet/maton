"""Strict, deliberately small TOML configuration surface."""

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import tomllib
from typing import Any


@dataclass
class Config:
    seed: int = 0
    training: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    terrain: dict[str, Any] = field(default_factory=dict)
    ppo: dict[str, Any] = field(default_factory=dict)
    dreamer: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with Path(path).open("rb") as handle:
            values = tomllib.load(handle)
        unknown = values.keys() - {"seed", "training", "env", "terrain", "ppo", "dreamer"}
        if unknown:
            raise ValueError(f"Unknown config sections: {sorted(unknown)}")
        config = cls(**values)
        config.resolved()  # Reject misspelled and invalid settings before training.
        return config

    def resolved(self) -> dict[str, Any]:
        from maton.env import EnvConfig
        from maton.terrain import TerrainConfig
        from maton.ppo import PPOConfig
        from maton.dreamer import DreamerConfig
        from maton.training import TrainingConfig

        training = TrainingConfig(**self.training)
        ppo = PPOConfig(**self.ppo)
        return dict(seed=self.seed, training=asdict(training), env=asdict(EnvConfig(**self.env)),
                    terrain=asdict(TerrainConfig(**self.terrain)),
                    ppo=asdict(ppo),
                    dreamer=asdict(DreamerConfig(**self.dreamer)))

    def environment(self):
        from maton.env import EnvConfig, WalkerEnv
        from maton.terrain import TerrainConfig

        return WalkerEnv(EnvConfig(**self.env), TerrainConfig(**self.terrain))


def save_run(config: Config, algorithm: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(json.dumps(
        dict(algorithm=algorithm, **config.resolved()), indent=2) + "\n")


def load_run(output: Path) -> tuple[Config, str]:
    values = json.loads((output / "run.json").read_text())
    algorithm = values.pop("algorithm")
    return Config(**values), algorithm
