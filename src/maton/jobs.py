"""Named experiments and stage-specific configuration, without importing JAX."""

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import tomllib

from maton.algorithms import algorithm as lookup


NAME = r"[a-z0-9]+(?:-[a-z0-9]+)*"
JOB_SEGMENT = r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"
JOB_NAME = rf"{JOB_SEGMENT}(?:/{JOB_SEGMENT})*"


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


def merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Tables merge recursively; scalar values and arrays replace their parents."""
    result = deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge(result[key], value) if isinstance(result.get(key), dict) and isinstance(value, dict) else deepcopy(value)
    return result


def job_root(path: Path) -> Path:
    return next((parent for parent in path.parents if parent.name == 'jobs'), path.parent)


def job_reference(path: Path, name: Any) -> Path:
    if not isinstance(name, str) or not re.fullmatch(JOB_NAME, name):
        raise ValueError('Job references must use alphanumeric hyphen-separated names')
    return job_root(path) / f'{name}.toml'


def inherited(path: Path, stack: tuple[Path, ...]) -> tuple[dict[str, Any], tuple[Path, ...]]:
    if path.resolve() in stack:
        raise ValueError(f'Job dependency cycle: {" -> ".join(p.stem for p in (*stack, path))}')
    values = read_toml(path)
    parent = values.pop('extends', None)
    base, inputs = {}, ()
    if parent is not None:
        parent_path = job_reference(path, parent)
        base, inputs = inherited(parent_path, (*stack, path.resolve()))
        # Rebase inherited references; child overrides stay relative to the child.
        for name in ('walker', 'curriculum'):
            if isinstance(base.get(name), str):
                base[name] = str((parent_path.parent / base[name]).resolve())
    inherit_evaluations = values.pop('inherit_evaluations', True)
    if type(inherit_evaluations) is not bool:
        raise ValueError('inherit_evaluations must be a boolean')
    if not inherit_evaluations:
        base.pop('evaluations', None)
    return merge(base, values), (*inputs, path)


@dataclass(frozen=True)
class Job:
    path: Path
    algorithm: str
    settings: dict[str, Any]
    evaluations: dict[str, Evaluation]
    config_paths: tuple[Path, ...] = ()
    source: "Job | None" = None
    source_checkpoint: str = "final"

    @property
    def name(self) -> str:
        return self.path.relative_to(job_root(self.path)).with_suffix('').as_posix()

    @classmethod
    def load(cls, path: str | Path, _stack: tuple[Path, ...] = ()) -> 'Job':
        path = Path(path)
        if not re.fullmatch(JOB_NAME, path.relative_to(job_root(path)).with_suffix('').as_posix()):
            raise ValueError('Job names must use alphanumeric hyphen-separated segments')
        values, inputs = inherited(path, _stack)
        continuation = values.pop("continue_from", None)
        source, checkpoint = None, "final"
        if continuation is not None:
            if not isinstance(continuation, dict) or "job" not in continuation or continuation.keys() - {"job", "checkpoint"}:
                raise ValueError("continue_from must contain job and optional checkpoint")
            checkpoint = Evaluation(checkpoint=continuation.get("checkpoint", "final")).checkpoint
            source = cls.load(job_reference(path, continuation["job"]), (*_stack, path.resolve()))
        algorithm = values.pop('algorithm', None)
        lookup(algorithm)
        if source is not None and source.algorithm != algorithm:
            raise ValueError("continue_from must use the same algorithm")
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
        job = cls(path, algorithm, values, evaluations, inputs, source, checkpoint)
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
        if self.source is not None:
            values['continue_from'] = dict(job=self.source.name, checkpoint=self.source_checkpoint,
                                           training=self.source.training_spec())
        return values

    def source_train(self) -> Path | None:
        return None if self.source is None else Path('data/jobs') / self.source.name / 'train'

    def initial_checkpoint(self) -> Path | None:
        train = self.source_train()
        return None if train is None else self.checkpoint(train, Evaluation(checkpoint=self.source_checkpoint))

    def config(self):
        from maton.config import Config
        return Config.from_values(self.algorithm, self.settings, self.path)

    def checkpoint(self, train: Path, evaluation: Evaluation) -> Path:
        step = None if evaluation.checkpoint == 'final' else evaluation.checkpoint.removeprefix('step-')
        return lookup(self.algorithm).checkpoint(train, step)

    def assert_training_identity(self, train: Path) -> None:
        provenance = train / 'provenance.json'
        if train.exists():
            if not provenance.exists() or json.loads(provenance.read_text())['training_fingerprint'] != fingerprint(self.training_spec()):
                raise ValueError(f'{train}: training settings changed; create a new job name or explicitly remove the old training artifact')
