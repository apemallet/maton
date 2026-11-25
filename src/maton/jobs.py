"""Named experiments and stage-specific configuration, without importing JAX."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import tomllib


NAME = r"[a-z0-9]+(?:-[a-z0-9]+)*"


def fingerprint(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read_toml(path: Path) -> dict[str, Any]:
    with path.open('rb') as stream:
        return tomllib.load(stream)


@dataclass(frozen=True)
class Evaluation:
    checkpoint: str = 'final'
    level: int | None = None
    seed: int = 1000
    episodes: int = 1
    render: tuple[int, ...] = ()

    def __post_init__(self):
        if not isinstance(self.checkpoint, str) or not re.fullmatch(r'final|step-[0-9]{8,}', self.checkpoint):
            raise ValueError('checkpoint must be final or step-XXXXXXXX')
        if self.level is not None and (type(self.level) is not int or self.level < 1):
            raise ValueError('evaluation level must be a positive integer')
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError('evaluation seed must be a nonnegative integer')
        if type(self.episodes) is not int or self.episodes < 1:
            raise ValueError('evaluation episodes must be a positive integer')
        if not isinstance(self.render, (list, tuple)) or any(type(i) is not int or not 0 <= i < self.episodes for i in self.render):
            raise ValueError('render must contain episode indices within this evaluation')
        if len(set(self.render)) != len(self.render):
            raise ValueError('render indices must be unique')
        object.__setattr__(self, 'render', tuple(self.render))

    def spec(self) -> dict[str, Any]:
        """Rendering selection does not affect recorded trajectories."""
        return {k: v for k, v in asdict(self).items() if k != 'render'}


@dataclass(frozen=True)
class Job:
    path: Path
    algorithm: str
    settings: dict[str, Any]
    evaluations: dict[str, Evaluation]

    @classmethod
    def load(cls, path: str | Path) -> 'Job':
        path = Path(path)
        if not re.fullmatch(NAME, path.stem):
            raise ValueError('Job names must use lowercase kebab-case')
        values = read_toml(path)
        algorithm = values.pop('algorithm', None)
        if algorithm not in ('ppo', 'dreamer'):
            raise ValueError('Job algorithm must be ppo or dreamer')
        protocols = values.pop('evaluations', {})
        if not isinstance(protocols, dict):
            raise ValueError('evaluations must be a table of named protocols')
        evaluations = {}
        for name, options in protocols.items():
            if not re.fullmatch(NAME, name) or not isinstance(options, dict):
                raise ValueError('Evaluations must be named tables using lowercase kebab-case')
            evaluations[name] = Evaluation(**options)
        unknown = values.keys() - {'seed', 'walker', 'curriculum', 'env', 'training', algorithm}
        if unknown:
            raise ValueError(f'Unknown job settings: {sorted(unknown)}')
        if type(values.get('seed', 0)) is not int or values.get('seed', 0) < 0:
            raise ValueError('Job seed must be a nonnegative integer')
        for reference in ('walker', 'curriculum'):
            if not isinstance(values.get(reference), str):
                raise ValueError(f'Job TOML must reference a {reference} file')
        job = cls(path, algorithm, values, evaluations)
        levels = read_toml(job.reference('curriculum')).get('levels', [])
        if any(e.level is not None and e.level > len(levels) for e in evaluations.values()):
            raise ValueError('Evaluation level is outside this curriculum')
        return job

    def reference(self, name: str) -> Path:
        return self.path.parent / self.settings[name]

    def training_spec(self) -> dict[str, Any]:
        values = dict(self.settings, algorithm=self.algorithm)
        for name in ('walker', 'curriculum'):
            values[name] = read_toml(self.reference(name))
        return values

    def config(self):
        from maton.config import Config
        return Config.from_values(self.settings, self.path)

    def checkpoint(self, train: Path, evaluation: Evaluation) -> Path:
        if evaluation.checkpoint == 'final':
            return train / ('policy.eqx' if self.algorithm == 'ppo' else 'dreamer.pkl')
        step = evaluation.checkpoint.removeprefix('step-')
        return (train / 'checkpoints' / f'step-{step}.eqx' if self.algorithm == 'ppo'
                else train / f'dreamer-{step}.pkl')

    def assert_training_identity(self, train: Path) -> None:
        provenance = train / 'provenance.json'
        if train.exists():
            if not provenance.exists() or json.loads(provenance.read_text())['training_fingerprint'] != fingerprint(self.training_spec()):
                raise ValueError(f'{train}: training settings changed; create a new job name or explicitly remove the old training artifact')
