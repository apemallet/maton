"""Shared off-policy training: parallel MJX collection, on-device replay, jitted updates.

A learner supplies networks, acting, and one gradient update; this module owns
environments, seeding, replay, update-to-data scheduling, curriculum, metrics, and
checkpoints. Collection and each burst of updates are single compiled calls.
"""

from dataclasses import asdict, dataclass
from functools import partial
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
import numpy as np

from maton import sessions
from maton.curriculum import CurriculumProgress
from maton.training import TrainingConfig


@dataclass(frozen=True)
class OffPolicyConfig:
    num_envs: int = 4
    batch_size: int = 256
    buffer_size: int = 1_000_000
    seed_steps: int | None = None  # None: max(1000, 5 × episode_length), as in TD-MPC2
    update_ratio: float = 1.0  # gradient updates per environment transition
    log_every: int = 2048

    def __post_init__(self):
        for name in ("num_envs", "batch_size", "buffer_size", "log_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.seed_steps is not None and (type(self.seed_steps) is not int or self.seed_steps < 1):
            raise ValueError("seed_steps must be a positive integer")
        if not 0 < self.update_ratio < float("inf"):
            raise ValueError("update_ratio must be finite and positive")


class Batch(eqx.Module):
    """Time-major windows `[horizon, batch, ...]` of consecutive transitions."""
    observation: Array
    action: Array
    reward: Array
    next_observation: Array
    terminated: Array
    mask: Array  # 1 while every earlier transition in the window continued its episode


class Replay(eqx.Module):
    """Ring buffer `[rows, envs, ...]`; each environment column is one time series."""
    observation: Array
    action: Array
    reward: Array
    next_observation: Array
    terminated: Array
    done: Array
    pointer: Array
    rows: Array

    @classmethod
    def create(cls, rows: int, envs: int, observation_size: int, action_size: int) -> "Replay":
        return cls(jnp.zeros((rows, envs, observation_size)), jnp.zeros((rows, envs, action_size)),
                   jnp.zeros((rows, envs)), jnp.zeros((rows, envs, observation_size)),
                   jnp.zeros((rows, envs), bool), jnp.zeros((rows, envs), bool),
                   jnp.array(0, jnp.int32), jnp.array(0, jnp.int32))

    @property
    def capacity(self) -> int:
        return self.reward.shape[0]

    def add(self, observation, action, reward, next_observation, terminated, done) -> "Replay":
        """Store one transition per environment. `next_observation` is the true successor,
        even when the environment resets, so time limits bootstrap correctly."""
        row = self.pointer

        def put(buffer, value):
            return jax.lax.dynamic_update_index_in_dim(buffer, value.astype(buffer.dtype), row, 0)

        return Replay(put(self.observation, observation), put(self.action, action),
                      put(self.reward, reward), put(self.next_observation, next_observation),
                      put(self.terminated, terminated), put(self.done, done),
                      (row + 1) % self.capacity, jnp.minimum(self.rows + 1, self.capacity))

    def sample(self, key: Array, batch: int, horizon: int) -> Batch:
        """Uniform windows that never wrap past the write pointer; needs `rows >= horizon`."""
        offset_key, column_key = jax.random.split(key)
        offset = jax.random.randint(offset_key, (batch,), 0, self.rows - horizon + 1)
        start = (self.pointer - self.rows + offset) % self.capacity
        rows = (start[None] + jnp.arange(horizon)[:, None]) % self.capacity
        columns = jax.random.randint(column_key, (1, batch), 0, self.done.shape[1])
        take = lambda buffer: buffer[rows, columns]
        done = take(self.done)
        continuing = jnp.cumprod(1.0 - done, 0)
        mask = jnp.concatenate((jnp.ones((1, batch)), continuing[:-1]), 0)
        return Batch(take(self.observation), take(self.action), take(self.reward),
                     take(self.next_observation), take(self.terminated).astype(jnp.float32), mask)


class Learner(Protocol):
    """One off-policy algorithm. `init(key).model` is the saved inference pytree."""
    config: Any
    horizon: int

    def init_model(self, key: Array) -> Any: ...

    def init(self, key: Array) -> Any: ...

    def init_carry(self, envs: int) -> Any: ...

    def act(self, model, carry, observation: Array, first: Array, key: Array,
            explore: bool) -> tuple[Array, Any]: ...

    def update(self, state, batch: Batch, key: Array) -> tuple[Any, dict[str, Array]]: ...


MakeLearner = Callable[[Any, int, int, int, TrainingConfig], Learner]


def discount(value: float | None, episode_length: int) -> float:
    """Explicit value, or TD-MPC2's horizon heuristic `clip((L/5 - 1) / (L/5), 0.95, 0.995)`."""
    if value is not None:
        return value
    fraction = episode_length / 5
    return min(max((fraction - 1) / fraction, 0.95), 0.995)


def seed_steps(config: OffPolicyConfig, episode_length: int, steps: int) -> int:
    seeded = config.seed_steps if config.seed_steps is not None else max(1000, 5 * episode_length)
    return min(math.ceil(seeded / config.num_envs) * config.num_envs, steps)


def _select(done: Array, fresh, old):
    return jax.tree.map(lambda new, previous: jnp.where(done.reshape(done.shape + (1,) * (new.ndim - 1)),
                                                        new, previous), fresh, old)


def _save(path: Path, name: str, model, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, model)
    path.with_suffix(".json").write_text(json.dumps(dict(algorithm=name, **metadata), indent=2) + "\n")


def train(env: Any, name: str, make: MakeLearner, config: OffPolicyConfig, training: TrainingConfig,
          output: str | Path, seed: int = 0, pause: Callable[[int], bool] = sessions.never) -> dict[str, Any]:
    """Train to `training.steps`, resuming from `output/resume` when present; `pause(steps)`
    ends the session early with resumable state saved."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    envs = config.num_envs
    if training.steps % envs:
        raise ValueError("training.steps must be divisible by num_envs")
    episode_length = env.config.episode_length
    seeding = seed_steps(config, episode_length, training.steps)
    curriculum = CurriculumProgress(env.curriculum)
    reset = jax.vmap(env.reset, in_axes=(0, None))
    step = jax.vmap(env.step)
    observe = jax.vmap(env.observation)
    key = jax.random.key(seed)
    key, reset_key, init_key = jax.random.split(key, 3)
    states = jax.jit(reset)(jax.random.split(reset_key, envs), curriculum.level)
    observation = jax.jit(observe)(states)
    learner = make(config, observation.shape[-1], env.action_size, episode_length, training)
    state = learner.init(init_key)
    rows = max(learner.horizon, math.ceil(min(config.buffer_size, training.steps) / envs))
    replay = Replay.create(rows, envs, observation.shape[-1], env.action_size)
    carry = learner.init_carry(envs)
    first = jnp.ones(envs, bool)

    host: dict[str, Any] = dict(transitions=0, updates=0, episodes=0, debt=0.0, pretrained=False,
                                refill_until=0, stored=0, episode_return=[0.0] * envs, seconds=0.0)
    restored = sessions.load(output, (state, carry, states, observation, first, sessions.keys(key)))
    if restored is not None:
        (state, carry, states, observation, first, key_data), host = restored
        key = sessions.unkeys(key_data)
        curriculum.level, curriculum.successes, curriculum.complete = (
            host["curriculum"]["level"], host["curriculum"]["successes"], host["curriculum"]["complete"])
        saved = sessions.load(output, replay, "replay")
        if saved is not None and saved[1]["step"] == host["transitions"]:
            replay = saved[0]
        else:  # New machine or crash: refill replay with the current policy before updating.
            host["stored"], host["refill_until"] = 0, host["transitions"] + seeding
        print(f"{name}: resuming at {host['transitions']}/{training.steps} transitions", flush=True)

    @partial(jax.jit, static_argnames="random", donate_argnames="replay")
    def collect(model, carry, states, observation, first, replay, key, level, random):
        act_key, reset_key = jax.random.split(key)
        if random:
            action = jax.random.uniform(act_key, (envs, env.action_size), minval=-1, maxval=1)
        else:
            action, carry = learner.act(model, carry, observation, first, act_key, True)
        following = step(states, action)
        successor = observe(following)
        done = following.terminated | following.truncated
        replay = replay.add(observation, action, following.reward, successor, following.terminated, done)
        states = jax.lax.cond(done.any(), lambda: _select(done, reset(jax.random.split(reset_key, envs), level),
                                                             following), lambda: following)
        observation = jnp.where(done[:, None], observe(states), successor)
        info = (following.reward, done, following.success, following.level)
        return carry, states, observation, done, replay, info

    @partial(jax.jit, donate_argnames="state")
    def update(state, replay, key, count):
        def body(_, carry):
            state, key, total = carry
            key, sample_key, update_key = jax.random.split(key, 3)
            batch = replay.sample(sample_key, config.batch_size, learner.horizon)
            state, metrics = learner.update(state, batch, update_key)
            return state, key, jax.tree.map(jnp.add, total, metrics)

        shapes = jax.eval_shape(learner.update, state, replay.sample(key, config.batch_size, learner.horizon), key)[1]
        zeros = jax.tree.map(lambda shape: jnp.zeros(shape.shape, shape.dtype), shapes)
        state, _, total = jax.lax.fori_loop(0, count, body, (state, key, zeros))
        return state, jax.tree.map(lambda value: value / jnp.maximum(count, 1), total)

    def metadata(step: int) -> dict[str, Any]:
        return dict(step=step, observation_size=int(observation.shape[-1]), action_size=env.action_size,
                    episode_length=episode_length, config=asdict(config), training=asdict(training),
                    curriculum=curriculum.snapshot())

    def checkpoint(with_replay: bool) -> None:
        host.update(curriculum=dict(level=curriculum.level, successes=curriculum.successes,
                                    complete=curriculum.complete),
                    episode_return=episode_return.tolist(), seconds=prior_seconds + time.perf_counter() - started)
        if with_replay:
            sessions.save(output, replay, dict(step=host["transitions"]), "replay")
        sessions.save(output, (state, carry, states, observation, first, sessions.keys(key)), host)

    episode_return = np.asarray(host["episode_return"], float)
    window: dict[str, Any] = dict(returns=[], successes=0, metrics=[])
    record: dict[str, Any] = {}
    prior_seconds, session_start, started = host["seconds"], host["transitions"], time.perf_counter()
    complete = host["transitions"] >= training.steps
    with (output / "metrics.jsonl").open("a" if restored else "w") as log:
        while not complete:
            transitions = host["transitions"]
            key, collect_key, update_key = jax.random.split(key, 3)
            carry, states, observation, first, replay, info = collect(
                state.model, carry, states, observation, first, replay, collect_key,
                jnp.asarray(curriculum.level, jnp.int32), random=transitions < seeding)
            previous, transitions = transitions, transitions + envs
            host["transitions"], host["stored"] = transitions, host["stored"] + 1
            reward, done, success, level = jax.device_get(info)
            episode_return += reward
            for index in np.flatnonzero(done):
                host["episodes"] += 1
                window["returns"].append(float(episode_return[index]))
                window["successes"] += int(success[index])
                curriculum.record(bool(success[index]), int(level[index]))
                episode_return[index] = 0.0
            ready = transitions >= max(seeding, host["refill_until"]) and host["stored"] >= learner.horizon
            if ready:
                if not host["pretrained"]:  # One burst over the seed data, then the steady ratio.
                    count, host["pretrained"] = max(1, round(seeding * config.update_ratio)), True
                else:
                    host["debt"] += envs * config.update_ratio
                    count = int(host["debt"])
                    host["debt"] -= count
                if count:
                    state, metrics = update(state, replay, update_key, jnp.asarray(count, jnp.int32))
                    host["updates"] += count
                    window["metrics"].append((count, jax.device_get(metrics)))
            complete = transitions >= training.steps
            if transitions // config.log_every > previous // config.log_every or complete:
                elapsed = time.perf_counter() - started
                record = dict(steps=transitions, updates=host["updates"], episodes=host["episodes"],
                              episode_return=float(np.mean(window["returns"])) if window["returns"] else None,
                              successes=window["successes"],
                              steps_per_second=(transitions - session_start) / elapsed,
                              curriculum=curriculum.snapshot())
                if window["metrics"]:
                    weights = np.array([count for count, _ in window["metrics"]], float)
                    for metric in window["metrics"][0][1]:
                        values = np.array([float(row[metric]) for _, row in window["metrics"]])
                        if not np.isfinite(values).all():
                            raise FloatingPointError(f"Nonfinite {name} metric {metric} at step {transitions}")
                        record[metric] = float(np.average(values, weights=weights))
                log.write(json.dumps(record) + "\n")
                log.flush()
                window = dict(returns=[], successes=0, metrics=[])
                print(f"{name} {transitions}/{training.steps}: updates={host['updates']} "
                      f"episodes={host['episodes']} level={curriculum.level + 1} "
                      f"{record['steps_per_second']:.0f} steps/s", flush=True)
            if training.checkpoint_due(previous, transitions) or complete:
                model = jax.device_get(state.model)
                _save(output / "checkpoints" / f"step-{transitions:08d}.eqx", name, model, metadata(transitions))
                _save(output / f"{name}.eqx", name, model, metadata(transitions))
                checkpoint(with_replay=False)  # crash safety; resuming then refills replay
            if not complete and pause(transitions):
                checkpoint(with_replay=True)
                break
    seconds = prior_seconds + time.perf_counter() - started
    summary = dict(algorithm=name, steps=host["transitions"], complete=complete, updates=host["updates"],
                   episodes=host["episodes"], seed_steps=seeding, seconds=seconds,
                   session_steps_per_second=(host["transitions"] - session_start) / max(time.perf_counter() - started, 1e-9),
                   last=record, curriculum=curriculum.snapshot())
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


class OffPolicyAgent:
    """Evaluation controller; `reset()` restores the planner memory and random stream."""

    def __init__(self, learner: Learner, model, curriculum_level: int, seed: int = 0):
        self.learner, self.model, self.seed = learner, model, seed
        self.curriculum_level = curriculum_level
        self._act = jax.jit(partial(learner.act, explore=False))
        self.reset()

    def reset(self) -> None:
        self.carry = self.learner.init_carry(1)
        self.first = True
        self.key = jax.random.key(self.seed)

    def __call__(self, observation) -> np.ndarray:
        self.key, key = jax.random.split(self.key)
        action, self.carry = self._act(self.model, self.carry, jnp.asarray(observation, jnp.float32)[None],
                                       jnp.array([self.first]), key)
        self.first = False
        return np.asarray(action[0])


def load_agent(path: str | Path, final: str, make: MakeLearner, config_type: type) -> OffPolicyAgent:
    path = Path(path)
    if path.is_dir():
        path = path / final
    metadata = json.loads(path.with_suffix(".json").read_text())
    learner = make(config_type(**metadata["config"]), metadata["observation_size"], metadata["action_size"],
                   metadata["episode_length"], TrainingConfig(**metadata["training"]))
    model = eqx.tree_deserialise_leaves(path, learner.init_model(jax.random.key(0)))
    return OffPolicyAgent(learner, model, metadata["curriculum"]["level"])
