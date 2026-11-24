"""Small training harness around the pinned, unmodified DreamerV3 agent."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from importlib.resources import files
import json
from pathlib import Path
import pickle
import sys
import types
from typing import Any

import jax
import numpy as np

from maton.training import TrainingConfig

UPSTREAM_COMMIT = "e01491fad6434b2245a3b8ca201dd7faedcc458c"


@dataclass(frozen=True)
class DreamerConfig:
    batch_size: int = 1
    batch_length: int = 8
    train_every: int = 16
    replay_size: int = 2048
    hidden: int = 32
    deter: int = 64
    stoch: int = 4
    classes: int = 4
    imag_length: int = 5
    platform: str = "cpu"

    def __post_init__(self):
        for name in ("batch_size", "batch_length", "train_every", "replay_size", "hidden", "deter", "stoch", "classes", "imag_length"):
            if getattr(self, name) < 1:
                raise ValueError(f"dreamer.{name} must be positive")
        if self.batch_length < 2 or self.replay_size < self.batch_length * self.batch_size:
            raise ValueError("Dreamer needs batch_length >= 2 and enough replay capacity")
        if self.deter % 4 or self.hidden % 4:
            raise ValueError("Dreamer deter and hidden must be divisible by 4 RSSM blocks")


class _JaxCompat:
    def __getattr__(self, name):
        return getattr(jax, name)

    @staticmethod
    def jit(fun, *args, **kwargs):
        names = ("in_shardings", "out_shardings", "static_argnums",
                 "static_argnames", "donate_argnums", "donate_argnames")
        return jax.jit(fun, **dict(zip(names, args)), **kwargs)


def _runtime() -> tuple[Any, Any, Any]:
    # Upstream find_packages() omits its envs namespace. We supply our own env;
    # register only this unused namespace so the wheel's embodied import works.
    if "embodied.envs" not in sys.modules:
        sys.modules["embodied.envs"] = types.ModuleType("embodied.envs")
    import elements
    import embodied
    from dreamerv3.agent import Agent
    import embodied.jax.agent as agent_module
    import embodied.jax.transform as transform_module
    import embodied.jax.internal as internal_module
    # JAX 0.8 made jit options keyword-only; upstream still passes these
    # positionally. Scope this adapter to the upstream modules, never global JAX.
    proxy = _JaxCompat()
    agent_module.jax = proxy
    transform_module.jax = proxy
    internal_module.jax = proxy
    return elements, embodied, Agent


def agent_config(config: DreamerConfig, training: TrainingConfig, output: Path, seed: int):
    elements, _, _ = _runtime()
    from ruamel.yaml import YAML
    defaults = YAML(typ="safe").load(files("maton").joinpath("assets/dreamer_defaults.yaml").read_text())["defaults"]
    defaults["jax"].update(transfer_guard=False, precompile=False, profiler=False)
    cfg = elements.Config(defaults).update({
        "logdir": str(output), "seed": seed, "batch_size": config.batch_size,
        "batch_length": config.batch_length, "report_length": config.batch_length,
        "replay_context": 0,
        "jax.platform": config.platform, "jax.compute_dtype": "float32",
        "jax.prealloc": False, "jax.transfer_guard": False,
        "jax.precompile": False, "jax.profiler": False,
        "agent.dyn.rssm.deter": config.deter,
        "agent.dyn.rssm.hidden": config.hidden,
        "agent.dyn.rssm.stoch": config.stoch,
        "agent.dyn.rssm.classes": config.classes,
        "agent.dyn.rssm.blocks": 4,
        "agent.*.units": config.hidden,
        "agent.enc.simple.units": config.hidden,
        "agent.dec.simple.units": config.hidden,
        "agent.enc.simple.layers": 1, "agent.dec.simple.layers": 1,
        "agent.policy.layers": 1, "agent.value.layers": 1,
        "agent.imag_length": config.imag_length,
        "agent.opt.lr": training.learning_rate, "agent.opt.warmup": 0,
        "agent.report": False,
    })
    return elements.Config(**cfg.agent, logdir=cfg.logdir, seed=cfg.seed, jax=cfg.jax,
                           batch_size=cfg.batch_size, batch_length=cfg.batch_length,
                           replay_context=0, report_length=cfg.report_length,
                           replica=0, replicas=1)


def _spaces(obs_size: int, action_size: int):
    elements, _, _ = _runtime()
    obs = {"vector": elements.Space(np.float32, (obs_size,)),
           "reward": elements.Space(np.float32),
           **{k: elements.Space(bool) for k in ("is_first", "is_last", "is_terminal")}}
    return obs, {"action": elements.Space(np.float32, (action_size,), -1, 1)}


def _observation(vector, *, reward=0.0, first=False, last=False, terminal=False):
    return {"vector": np.asarray(vector, np.float32), "reward": np.asarray(reward, np.float32),
            "is_first": np.asarray(first), "is_last": np.asarray(last),
            "is_terminal": np.asarray(terminal)}


def _save(agent, path: Path, config: DreamerConfig, training: TrainingConfig, obs_size: int, action_size: int, seed: int, step: int):
    agent_state = agent.save()
    for name, value in agent_state["params"].items():
        if not np.isfinite(np.asarray(value)).all():
            raise FloatingPointError(f"Nonfinite Dreamer checkpoint parameter {name}")
    payload = {"agent": agent_state, "config": asdict(config), "obs_size": obs_size,
               "training": asdict(training),
               "action_size": action_size, "seed": seed, "step": step,
               "upstream_commit": UPSTREAM_COMMIT}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as file:
        pickle.dump(payload, file, protocol=5)
    temporary.replace(path)


class DreamerPolicy:
    """Recurrent seeded evaluation policy; call reset() between episodes."""
    def __init__(self, agent, action_counter: int = 0):
        self.agent = agent
        self.action_counter = action_counter
        self.reset()

    def reset(self):
        self.carry = self.agent.init_policy(1)
        self.first = True
        # Upstream samples actions even in eval mode. Reset the action RNG counter
        # to make retrospective evaluation repeatable without changing the model.
        self.agent.n_actions.value = self.action_counter

    def __call__(self, observation):
        obs = _observation(observation, first=self.first)
        batch = {k: v[None] for k, v in obs.items()}
        self.carry, action, _ = self.agent.policy(self.carry, batch, mode="eval")
        self.first = False
        return np.clip(np.asarray(action["action"][0], np.float32), -1, 1)


def load_policy(checkpoint: Path | str) -> DreamerPolicy:
    """Load a trusted local checkpoint (pickle); use reset() for each episode."""
    checkpoint = Path(checkpoint)
    with checkpoint.open("rb") as file:
        payload = pickle.load(file)
    if payload["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError("Checkpoint belongs to another DreamerV3 revision")
    config = DreamerConfig(**payload["config"])
    training = TrainingConfig(**payload["training"])
    _, _, Agent = _runtime()
    obs, act = _spaces(payload["obs_size"], payload["action_size"])
    agent = Agent(obs, act, agent_config(config, training, checkpoint.parent, payload["seed"]))
    agent.load(payload["agent"])
    return DreamerPolicy(agent)


def train(env: Any, config: DreamerConfig, training: TrainingConfig, output: Path, seed: int = 0):
    """Collect MJX transitions and optimize upstream world model and actor/value."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _, embodied, Agent = _runtime()
    key = jax.random.key(seed)
    key, reset_key = jax.random.split(key)
    reset_fn = jax.jit(env.reset)
    observation_fn = jax.jit(env.observation)
    state = reset_fn(reset_key)
    vector = np.asarray(observation_fn(state), np.float32)
    obs_space, act_space = _spaces(vector.size, 4)
    agent = Agent(obs_space, act_space, agent_config(config, training, output, seed))
    replay = embodied.Replay(length=config.batch_length, capacity=config.replay_size, seed=seed)
    stream = embodied.streams.Consec(
        embodied.streams.Stateless(lambda: replay.sample(config.batch_size)),
        length=config.batch_length, consec=1, prefix=0, contiguous=True)
    batches = iter(agent.stream(stream))
    train_carry = agent.init_train(config.batch_size)
    policy_carry = agent.init_policy(1)
    obs = _observation(vector, first=True)
    episode_return, episodes, updates = 0.0, 0, 0
    step_fn = jax.jit(env.step)
    metrics_file = output / "metrics.jsonl"
    with metrics_file.open("w") as metrics:
        step = 0
        while step < training.steps:
            policy_carry, actions, extras = agent.policy(
                policy_carry, {k: v[None] for k, v in obs.items()}, mode="train")
            action = np.asarray(actions["action"][0], np.float32)
            replay.add({**obs, "action": action,
                        **{k: np.asarray(v[0]) for k, v in extras.items()}})
            if bool(obs["is_last"]):
                key, reset_key = jax.random.split(key)
                state = reset_fn(reset_key)
                obs = _observation(observation_fn(state), first=True)
                continue  # Replay needs episode boundaries; these are not env transitions.
            else:
                state = step_fn(state, action)
                step += 1
                obs = _observation(observation_fn(state), reward=state.reward,
                                   last=state.done, terminal=state.terminated)
                episode_return += float(state.reward)
                if bool(state.done):
                    episodes += 1
                    metrics.write(json.dumps({"step": step, "episode_return": episode_return, "episodes": episodes}) + "\n")
                    episode_return = 0.0
            if len(replay) >= config.batch_size and step % config.train_every == 0:
                train_carry, _, mets = agent.train(train_carry, next(batches))
                updates += 1
                record: dict[str, int | float] = {"step": step, "updates": updates}
                for name, value in mets.items():
                    array = np.asarray(value)
                    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                        raise FloatingPointError(f"Nonfinite Dreamer metric {name}")
                    if array.ndim == 0 and np.issubdtype(array.dtype, np.number):
                        record[name] = float(array)
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
            if training.checkpoint_due(step - 1, step):
                _save(agent, output / f"dreamer-{step:08d}.pkl", config, training, vector.size, 4, seed, step)
        _save(agent, output / "dreamer.pkl", config, training, vector.size, 4, seed, training.steps)
    summary = {"steps": training.steps, "updates": updates, "episodes": episodes,
               "checkpoint": str(output / "dreamer.pkl"), "upstream_commit": UPSTREAM_COMMIT}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
