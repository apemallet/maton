"""Pausable, resumable training: trainers keep exact state under `<output>/resume/`.

A trainer called with an existing `resume/state.json` continues from it. `Pause` stops a
session after a step count, a wall-clock budget, or the first Ctrl-C; the trainer then saves
its state and returns a summary with `complete = False`.
"""

from contextlib import contextmanager
import json
from pathlib import Path
import signal
import time
from typing import Any, Iterator

import equinox as eqx
import jax

RESUME = "resume"


class Pause:
    def __init__(self, hours: float | None = None, steps: int | None = None):
        self.deadline = None if hours is None else time.monotonic() + hours * 3600
        self.steps = steps
        self.interrupted = False

    def __call__(self, steps: int) -> bool:
        return (self.interrupted or (self.steps is not None and steps >= self.steps)
                or (self.deadline is not None and time.monotonic() >= self.deadline))

    @contextmanager
    def catching_interrupt(self) -> Iterator["Pause"]:
        """First Ctrl-C requests a pause at the next safe point; a second one aborts."""
        def handler(signum, frame):
            if self.interrupted:
                raise KeyboardInterrupt
            self.interrupted = True
            print("\nPausing after this step; press Ctrl-C again to abort without saving.", flush=True)

        previous = signal.signal(signal.SIGINT, handler)
        try:
            yield self
        finally:
            signal.signal(signal.SIGINT, previous)


def never(steps: int) -> bool:
    return False


def directory(output: Path) -> Path:
    return Path(output) / RESUME


def resuming(output: Path) -> bool:
    return (directory(output) / "state.json").is_file()


def keys(key: jax.Array) -> jax.Array:
    return jax.random.key_data(key)


def unkeys(data: jax.Array) -> jax.Array:
    return jax.random.wrap_key_data(data)


def _replace(path: Path, write) -> None:
    temporary = path.with_name(path.name + ".tmp")
    write(temporary)
    temporary.replace(path)


def save(output: Path, tree: Any, host: dict[str, Any], name: str = "state") -> None:
    """Atomically write a pytree (`<name>.eqx`) and its host bookkeeping (`<name>.json`).
    The JSON lands last: a readable JSON always describes a complete pytree."""
    folder = directory(output)
    folder.mkdir(parents=True, exist_ok=True)
    _replace(folder / f"{name}.eqx", lambda path: eqx.tree_serialise_leaves(path, tree))
    _replace(folder / f"{name}.json", lambda path: path.write_text(json.dumps(host, indent=2) + "\n"))


def load(output: Path, skeleton: Any, name: str = "state") -> tuple[Any, dict[str, Any]] | None:
    folder = directory(output)
    if not (folder / f"{name}.json").is_file():
        return None
    host = json.loads((folder / f"{name}.json").read_text())
    return eqx.tree_deserialise_leaves(folder / f"{name}.eqx", skeleton), host
