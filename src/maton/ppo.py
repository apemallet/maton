"""Small continuous-action PPO baseline with Equinox policies."""
from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
import time
from typing import Any, Callable

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from maton.training import TrainingConfig
from maton import sessions
from maton.curriculum import CurriculumProgress


@dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 2
    rollout_steps: int = 32
    epochs: int = 2
    minibatch_size: int = 32
    hidden_size: int = 64
    initial_std: float = math.exp(-0.5)
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.001
    max_grad_norm: float = 0.5

    def __post_init__(self):
        for name in ("num_envs", "rollout_steps", "epochs", "minibatch_size", "hidden_size"):
            if getattr(self, name) <= 0:
                raise ValueError(f"ppo.{name} must be positive")
        if not math.exp(-5) <= self.initial_std <= math.exp(2):
            raise ValueError("ppo.initial_std must lie in [exp(-5), exp(2)]")
        if self.minibatch_size > self.num_envs * self.rollout_steps:
            raise ValueError("minibatch_size exceeds rollout batch")
        if self.num_envs * self.rollout_steps % self.minibatch_size:
            raise ValueError("rollout batch must be divisible by minibatch_size")
        if not 0 <= self.gamma <= 1 or not 0 <= self.gae_lambda <= 1:
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if any(getattr(self, name) <= 0 for name in ("clip_ratio", "max_grad_norm")):
            raise ValueError("Clip ratio and gradient norm must be positive")


class Policy(eqx.Module):
    actor: eqx.nn.MLP
    critic: eqx.nn.MLP
    log_std: jax.Array
    observation_size: int = eqx.field(static=True)
    action_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    curriculum_level: int = eqx.field(static=True)

    def __init__(self, observation_size: int, action_size: int, hidden_size: int, key: jax.Array,
                 curriculum_level: int = 0, initial_std: float = math.exp(-0.5)):
        actor_key, critic_key = jax.random.split(key)
        self.actor = eqx.nn.MLP(observation_size, action_size, hidden_size, 2,
                               activation=jax.nn.tanh, key=actor_key)
        self.actor = eqx.tree_at(lambda net: (net.layers[-1].weight, net.layers[-1].bias),
                                self.actor, (self.actor.layers[-1].weight * 0.01,
                                             jnp.zeros(action_size)))
        self.critic = eqx.nn.MLP(observation_size, 1, hidden_size, 2,
                                activation=jax.nn.tanh, key=critic_key)
        self.log_std = jnp.full((action_size,), math.log(initial_std))
        self.observation_size = observation_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.curriculum_level = curriculum_level

    def __call__(self, observation: jax.Array):
        return self.actor(observation), self.critic(observation)[0]


def log_probability(mean, log_std, raw_action):
    """Density of tanh-normal actions, evaluated using the unsquashed sample."""
    log_std = jnp.clip(log_std, -5.0, 2.0)
    normal = -0.5 * (((raw_action - mean) / jnp.exp(log_std)) ** 2
                     + 2 * log_std + jnp.log(2 * jnp.pi))
    # Stable log(1 - tanh(x)^2), including near-saturated actions.
    correction = 2 * (jnp.log(2.0) - raw_action - jax.nn.softplus(-2 * raw_action))
    return einx.sum("... a -> ...", normal - correction)  # pyright: ignore[reportPrivateImportUsage]


@eqx.filter_jit
def action(policy: Policy, observation: jax.Array):
    return jnp.tanh(policy.actor(jnp.asarray(observation, dtype=jnp.float32)))


@eqx.filter_jit
def _sample(policy: Policy, observations, key):
    mean, value = jax.vmap(policy)(observations)
    raw = mean + jnp.exp(jnp.clip(policy.log_std, -5, 2)) * jax.random.normal(key, mean.shape)
    return jnp.tanh(raw), raw, log_probability(mean, policy.log_std, raw), value


@eqx.filter_jit
def _values(policy: Policy, observations):
    return jax.vmap(policy)(observations)[1]


def advantages(rewards, values, next_values, terminated, truncated, gamma, gae_lambda):
    """Time-limit transitions bootstrap; both kinds of boundary stop GAE."""
    deltas = rewards + gamma * (1 - terminated) * next_values - values
    continuation = (1 - terminated) * (1 - truncated)

    def backward(carry, transition):
        delta, mask = transition
        carry = delta + gamma * gae_lambda * mask * carry
        return carry, carry

    _, result = jax.lax.scan(backward, jnp.zeros_like(rewards[0]),
                             (deltas, continuation), reverse=True)
    return result, result + values


def _loss(policy, batch, config):
    obs, raw_actions, old_logprob, targets, advantage = batch
    mean, value = jax.vmap(policy)(obs)
    logprob = log_probability(mean, policy.log_std, raw_actions)
    ratio = jnp.exp(logprob - old_logprob)
    clipped = jnp.clip(ratio, 1 - config.clip_ratio, 1 + config.clip_ratio)
    actor = -jnp.mean(jnp.minimum(ratio * advantage, clipped * advantage))
    critic = jnp.mean((value - targets) ** 2)
    # Normal entropy is a stable exploration regularizer before the tanh transform.
    entropy = jnp.sum(jnp.clip(policy.log_std, -5, 2) + 0.5 * jnp.log(2 * jnp.pi * jnp.e))
    loss = actor + config.value_coefficient * critic - config.entropy_coefficient * entropy
    return loss, (actor, critic, entropy, jnp.mean(old_logprob - logprob))


def _save(policy: Policy, path: Path, step: int, config: PPOConfig, training: TrainingConfig,
          curriculum: CurriculumProgress):
    path.parent.mkdir(parents=True, exist_ok=True)
    eqx.tree_serialise_leaves(path, policy)
    path.with_suffix(".json").write_text(json.dumps({
        "algorithm": "ppo", "step": step, "observation_size": policy.observation_size,
        "action_size": policy.action_size, "hidden_size": policy.hidden_size,
        "ppo": asdict(config),
        "training": asdict(training),
        "curriculum": curriculum.snapshot(),
    }, indent=2) + "\n")


def load_policy(path: str | Path) -> Policy:
    path = Path(path)
    if path.is_dir():
        path = path / "policy.eqx"
    metadata = json.loads(path.with_suffix(".json").read_text())
    policy = Policy(metadata["observation_size"], metadata["action_size"],
                    metadata["hidden_size"], jax.random.PRNGKey(0), metadata["curriculum"]["level"])
    return eqx.tree_deserialise_leaves(path, policy)


class PPOAgent:
    """Deterministic evaluation controller: the tanh of the actor mean."""
    def __init__(self, policy: Policy):
        self.policy = policy
        self.curriculum_level = policy.curriculum_level

    def reset(self) -> None:
        pass

    def __call__(self, observation) -> np.ndarray:
        return np.asarray(action(self.policy, observation))


def load_agent(path: str | Path) -> PPOAgent:
    return PPOAgent(load_policy(path))


def train(env: Any, config: PPOConfig, training: TrainingConfig, output: str | Path, seed: int = 0,
          pause: Callable[[int], bool] = sessions.never):
    """Train PPO, resuming from `output/resume` when present; `pause(steps)` ends the
    session after an update with resumable state saved."""
    if training.steps % config.num_envs:
        raise ValueError("training.steps must be divisible by ppo.num_envs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(seed)
    key, reset_key, policy_key = jax.random.split(key, 3)
    curriculum = CurriculumProgress(env.curriculum)
    reset_one = jax.jit(env.reset)
    reset = jax.jit(jax.vmap(env.reset, in_axes=(0, None)))
    step = jax.jit(jax.vmap(env.step))
    observe = jax.jit(jax.vmap(env.observation))
    states = reset(jax.random.split(reset_key, config.num_envs), curriculum.level)
    observations = observe(states)
    policy = Policy(observations.shape[-1], env.action_size, config.hidden_size, policy_key,
                    initial_std=config.initial_std)
    optimizer = optax.chain(optax.clip_by_global_norm(config.max_grad_norm),
                           optax.adam(training.learning_rate, eps=1e-5))
    optimizer_state = optimizer.init(eqx.filter(policy, eqx.is_inexact_array))
    host: dict[str, Any] = dict(transitions=0, iteration=0, episode_returns=[0.0] * config.num_envs,
                                completed_returns=[], last=None, seconds=0.0)
    restored = sessions.load(output, (policy, optimizer_state, states, observations, key))
    if restored is not None:
        (policy, optimizer_state, states, observations, key), host = restored
        curriculum.level, curriculum.successes, curriculum.complete = (
            host["curriculum"]["level"], host["curriculum"]["successes"], host["curriculum"]["complete"])
        print(f"PPO: resuming at {host['transitions']}/{training.steps} transitions", flush=True)

    @eqx.filter_jit
    def update(policy, optimizer_state, batch):
        (loss, metrics), grads = eqx.filter_value_and_grad(_loss, has_aux=True)(policy, batch, config)
        updates, optimizer_state = optimizer.update(grads, optimizer_state, policy)
        policy = eqx.apply_updates(policy, updates)
        return policy, optimizer_state, jnp.array((loss, *metrics, optax.tree.norm(grads)))

    # Python drives the rollout: MJX compiles one batched step, not a large unrolled graph.
    transitions = host["transitions"]
    episode_returns = np.asarray(host["episode_returns"], float)
    completed_returns: list[float] = host["completed_returns"]
    record: dict[str, Any] | None = host["last"]
    prior_seconds, started = host["seconds"], time.perf_counter()

    def checkpoint(iteration: int) -> None:
        host.update(transitions=transitions, iteration=iteration, episode_returns=episode_returns.tolist(),
                    completed_returns=completed_returns, last=record,
                    seconds=prior_seconds + time.perf_counter() - started,
                    curriculum=dict(level=curriculum.level, successes=curriculum.successes,
                                    complete=curriculum.complete))
        sessions.save(output, (policy, optimizer_state, states, observations, key), host)

    with (output / "metrics.csv").open("a" if restored else "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["update", "steps", "reward_mean", "episodes",
                                                    "episode_return_mean", "loss", "actor_loss",
                                                    "value_loss", "entropy", "approx_kl", "grad_norm",
                                                    "curriculum_level", "curriculum_successes", "curriculum_complete"])
        if not restored:
            writer.writeheader()
        total_updates = (training.steps + config.num_envs * config.rollout_steps - 1) // (config.num_envs * config.rollout_steps)
        for iteration in range(host["iteration"] + 1, total_updates + 1):
            rollout = []
            rollout_steps = min(config.rollout_steps, (training.steps - transitions) // config.num_envs)
            for _ in range(rollout_steps):
                key, sample_key = jax.random.split(key)
                actions, raw, logprob, value = _sample(policy, observations, sample_key)
                next_states = step(states, actions)
                next_observations = observe(next_states)
                next_value = _values(policy, next_observations)
                rollout.append((observations, raw, logprob, value, next_states.reward,
                                next_value, next_states.terminated, next_states.truncated))
                episode_returns += np.asarray(next_states.reward)
                done = np.asarray(next_states.terminated | next_states.truncated)
                states = next_states
                # Reset only completed slots, retaining terminal observations above for bootstrap.
                if done.any():
                    for index in np.flatnonzero(done):
                        completed_returns.append(float(episode_returns[index]))
                        curriculum.record(bool(next_states.success[index]), int(next_states.level[index]))
                        episode_returns[index] = 0
                        key, reset_key = jax.random.split(key)
                        fresh = reset_one(reset_key, curriculum.level)
                        states = jax.tree.map(lambda batch, item: batch.at[index].set(item), states, fresh)
                    observations = observe(states)
                else:
                    observations = next_observations
            obs, raw, old_logprob, values, rewards, next_values, terminated, truncated = (
                jnp.stack([row[column] for row in rollout]) for column in range(8))
            advantage, targets = advantages(rewards, values, next_values, terminated, truncated,
                                             config.gamma, config.gae_lambda)
            advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)
            batch = tuple(einx.id("t n ... -> (t n) ...", x)  # pyright: ignore[reportPrivateImportUsage]
                          for x in (obs, raw, old_logprob, targets, advantage))
            update_metrics = []
            for _ in range(config.epochs):
                key, shuffle_key = jax.random.split(key)
                order = jax.random.permutation(shuffle_key, config.num_envs * rollout_steps)
                for offset in range(0, len(order), config.minibatch_size):
                    indices = order[offset:offset + config.minibatch_size]
                    policy, optimizer_state, metrics = update(policy, optimizer_state,
                                                               tuple(x[indices] for x in batch))
                    update_metrics.append(np.asarray(metrics))
            metrics = np.mean(update_metrics, axis=0)
            if not np.isfinite(metrics).all():
                raise FloatingPointError(f"Nonfinite PPO update {iteration}: {metrics}")
            previous_transitions = transitions
            transitions += config.num_envs * rollout_steps
            record = dict(zip(("loss", "actor_loss", "value_loss", "entropy", "approx_kl", "grad_norm"),
                              map(float, metrics), strict=True))
            record.update(update=iteration, steps=transitions, reward_mean=float(jnp.mean(rewards)),
                          curriculum_level=curriculum.level, curriculum_successes=curriculum.successes,
                          curriculum_complete=curriculum.complete,
                          episodes=len(completed_returns),
                          episode_return_mean=float(np.mean(completed_returns)) if completed_returns else 0.0)
            writer.writerow(record)
            stream.flush()
            print(f"PPO {iteration}/{total_updates}: steps={transitions} loss={record['loss']:.4f}", flush=True)
            if training.checkpoint_due(previous_transitions, transitions) or transitions == training.steps:
                _save(policy, output / "checkpoints" / f"step-{transitions:08d}.eqx", transitions, config, training, curriculum)
                _save(policy, output / "policy.eqx", transitions, config, training, curriculum)
                checkpoint(iteration)
            if transitions < training.steps and pause(transitions):
                checkpoint(iteration)
                break
    summary = {"algorithm": "ppo", "steps": transitions, "complete": transitions >= training.steps,
               "updates": host["iteration"] if transitions < training.steps else total_updates,
               "episodes": len(completed_returns), "seconds": prior_seconds + time.perf_counter() - started,
               "last": record, "curriculum": curriculum.snapshot()}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


Config = PPOConfig
