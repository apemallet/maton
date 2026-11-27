"""Soft Actor-Critic (Haarnoja et al., 2018) on the shared off-policy driver.

Model-free off-policy baseline for TD-MPC2: tanh-Gaussian actor, a LayerNorm critic
ensemble with clipped double-Q targets, and automatic temperature tuning.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import einx
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array
import optax

from maton import offpolicy, sessions
from maton.layers import MLP
from maton.offpolicy import Batch, OffPolicyConfig
from maton.training import TrainingConfig


@dataclass(frozen=True)
class SACConfig(OffPolicyConfig):
    hidden_size: int = 256
    hidden_layers: int = 2
    num_q: int = 2
    critic_norm: bool = True  # LayerNorm in critics: stable at high update ratios
    initial_alpha: float = 1.0
    target_entropy: float | None = None  # None: -action_size
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    tau: float = 0.005
    discount: float | None = None  # None: TD-MPC2's episode-length heuristic (0.99 at 512 steps)

    def __post_init__(self):
        super().__post_init__()
        if self.hidden_size < 1 or self.hidden_layers < 1 or self.num_q < 2:
            raise ValueError("sac needs positive hidden_size/hidden_layers and num_q >= 2")
        if self.initial_alpha <= 0 or self.log_std_min >= self.log_std_max or not 0 < self.tau <= 1:
            raise ValueError("sac initial_alpha, log-std bounds, or tau out of range")
        if self.discount is not None and not 0 < self.discount <= 1:
            raise ValueError("sac.discount must be in (0, 1]")


Config = SACConfig


class Model(eqx.Module):
    actor: MLP
    critics: MLP  # ensemble


class TrainState(eqx.Module):
    model: Model
    target_critics: MLP
    actor_optimizer: Any
    critic_optimizer: Any
    log_alpha: Array
    alpha_optimizer: Any


class SAC:
    horizon = 1

    def __init__(self, config: SACConfig, observation_size: int, action_size: int, episode_length: int,
                 training: TrainingConfig):
        self.config = config
        self.observation_size, self.action_size = observation_size, action_size
        self.discount = offpolicy.discount(config.discount, episode_length)
        self.target_entropy = -action_size if config.target_entropy is None else config.target_entropy
        self.optimizer = optax.adam(training.learning_rate)

    def init_model(self, key: Array) -> Model:
        c = self.config
        actor_key, critic_key = jax.random.split(key)
        hidden = (c.hidden_size,) * c.hidden_layers
        return Model(MLP(actor_key, self.observation_size, hidden, 2 * self.action_size,
                         hidden_activation="relu", norm=False),
                     MLP(critic_key, self.observation_size + self.action_size, hidden, 1, ensemble=c.num_q,
                         hidden_activation="relu", norm=c.critic_norm))

    def init(self, key: Array) -> TrainState:
        model = self.init_model(key)
        log_alpha = jnp.log(jnp.asarray(self.config.initial_alpha))
        return TrainState(model, jax.tree.map(jnp.copy, model.critics),
                          self.optimizer.init(eqx.filter(model.actor, eqx.is_array)),
                          self.optimizer.init(eqx.filter(model.critics, eqx.is_array)),
                          log_alpha, self.optimizer.init(log_alpha))

    def init_carry(self, envs: int) -> Array:
        return jnp.zeros((envs, 0))

    def sample(self, actor: MLP, observation: Array, key: Array) -> tuple[Array, Array]:
        c = self.config
        mean, raw = jnp.split(actor(observation), 2, -1)
        log_std = c.log_std_min + 0.5 * (c.log_std_max - c.log_std_min) * (jnp.tanh(raw) + 1)
        pre = mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)
        # Stable log(1 - tanh(u)^2) correction for the squashed Gaussian.
        log_prob = (-0.5 * ((pre - mean) / jnp.exp(log_std)) ** 2 - log_std - 0.5 * jnp.log(2 * jnp.pi)
                    - 2 * (jnp.log(2.0) - pre - jax.nn.softplus(-2 * pre)))
        return jnp.tanh(pre), einx.sum("... a -> ...", log_prob)  # pyright: ignore[reportPrivateImportUsage]

    @staticmethod
    def values(critics: MLP, observation: Array, action: Array) -> Array:
        return critics(jnp.concatenate((observation, action), -1))[..., 0]  # [q, ...]

    def act(self, model: Model, carry: Array, observation: Array, first: Array, key: Array,
            explore: bool) -> tuple[Array, Array]:
        if explore:
            return self.sample(model.actor, observation, key)[0], carry
        return jnp.tanh(jnp.split(model.actor(observation), 2, -1)[0]), carry

    def update(self, state: TrainState, batch: Batch, key: Array) -> tuple[TrainState, dict[str, Array]]:
        c = self.config
        target_key, actor_key = jax.random.split(key)
        observation, action, reward = batch.observation[0], batch.action[0], batch.reward[0]
        next_observation, terminated = batch.next_observation[0], batch.terminated[0]
        alpha = jnp.exp(state.log_alpha)
        model = state.model

        next_action, next_log_prob = self.sample(model.actor, next_observation, target_key)
        next_value = self.values(state.target_critics, next_observation, next_action).min(0) - alpha * next_log_prob
        target = jax.lax.stop_gradient(reward + self.discount * (1 - terminated) * next_value)

        def critic_loss(critics: MLP):
            return jnp.sum(jnp.mean((self.values(critics, observation, action) - target) ** 2, -1))

        q_loss, critic_grads = eqx.filter_value_and_grad(critic_loss)(model.critics)
        updates, critic_optimizer = self.optimizer.update(critic_grads, state.critic_optimizer)
        critics = eqx.apply_updates(model.critics, updates)
        frozen = jax.lax.stop_gradient(critics)

        def actor_loss(actor: MLP):
            sampled, log_prob = self.sample(actor, observation, actor_key)
            return jnp.mean(alpha * log_prob - self.values(frozen, observation, sampled).min(0)), log_prob

        (pi_loss, log_prob), actor_grads = eqx.filter_value_and_grad(actor_loss, has_aux=True)(model.actor)
        updates, actor_optimizer = self.optimizer.update(actor_grads, state.actor_optimizer)
        actor = eqx.apply_updates(model.actor, updates)

        entropy_gap = jax.lax.stop_gradient(log_prob + self.target_entropy)
        alpha_grad = -jnp.mean(entropy_gap)  # d/d(log_alpha) of -mean(log_alpha · gap)
        alpha_update, alpha_optimizer = self.optimizer.update(alpha_grad, state.alpha_optimizer)
        log_alpha = state.log_alpha + jnp.asarray(alpha_update)
        target_critics = jax.tree.map(lambda old, new: old + c.tau * (new - old), state.target_critics, critics)
        metrics = dict(q_loss=q_loss, pi_loss=pi_loss, alpha=alpha, entropy=-jnp.mean(log_prob),
                       q_target=jnp.mean(target))
        return TrainState(Model(actor, critics), target_critics, actor_optimizer, critic_optimizer,
                          log_alpha, alpha_optimizer), metrics


def train(env: Any, config: SACConfig, training: TrainingConfig, output: str | Path, seed: int = 0,
          pause=sessions.never):
    return offpolicy.train(env, "sac", SAC, config, training, output, seed, pause)


def load_agent(path: str | Path) -> offpolicy.OffPolicyAgent:
    return offpolicy.load_agent(path, "sac.eqx", SAC, SACConfig)
