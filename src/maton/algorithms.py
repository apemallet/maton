"""Trainer registry: names, modules, checkpoints, and sources, without importing JAX.

Each trainer module `maton.<name>` exposes:

- `Config`: frozen dataclass of the job's `[<name>]` table.
- `train(env, config, training, output, seed) -> dict`: writes checkpoints under `output`.
- `load_agent(checkpoint) -> Agent`: a seeded evaluation controller.
"""

from dataclasses import dataclass
import importlib
from pathlib import Path
from types import ModuleType
from typing import Protocol

import numpy as np


class Agent(Protocol):
    """Evaluation controller: deterministic for a fixed checkpoint and reset."""
    curriculum_level: int

    def reset(self) -> None: ...

    def __call__(self, observation: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Algorithm:
    name: str
    final: str  # final checkpoint, relative to the training directory
    snapshot: str  # periodic checkpoint; `{step}` is the zero-padded transition count
    modules: tuple[str, ...] = ()  # shared maton modules this trainer imports
    assets: tuple[str, ...] = ()  # non-Python training inputs

    def load(self) -> ModuleType:
        return importlib.import_module(f'maton.{self.name}')

    def checkpoint(self, train: Path, step: str | None = None) -> Path:
        return train / (self.final if step is None else self.snapshot.format(step=step))

    def sources(self) -> list[str]:
        return [f'src/maton/{module}.py' for module in (self.name, *self.modules)] + list(self.assets)


_OFF_POLICY = ('offpolicy', 'layers')

ALGORITHMS = {algorithm.name: algorithm for algorithm in (
    Algorithm('ppo', 'policy.eqx', 'checkpoints/step-{step}.eqx'),
    Algorithm('dreamer', 'dreamer.pkl', 'dreamer-{step}.pkl', assets=('src/maton/assets/dreamer_defaults.yaml',)),
    Algorithm('tdmpc2', 'tdmpc2.eqx', 'checkpoints/step-{step}.eqx', _OFF_POLICY),
    Algorithm('sac', 'sac.eqx', 'checkpoints/step-{step}.eqx', _OFF_POLICY),
)}


def algorithm(name: str) -> Algorithm:
    if name not in ALGORITHMS:
        raise ValueError(f'Job algorithm must be one of {sorted(ALGORITHMS)}')
    return ALGORITHMS[name]
