"""TD-MPC2 (Hansen, Su & Wang, ICLR 2024) in Equinox, on the shared off-policy driver.

Decoder-free latent world model (SimNorm latents, two-hot reward/value heads, Q
ensemble) with MPPI planning over a learned policy prior. Follows the official
single-task implementation; departures are listed in docs/tdmpc2.md.
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

from maton import offpolicy
from maton.layers import MLP
from maton.offpolicy import Batch, OffPolicyConfig
from maton.training import TrainingConfig


@dataclass(frozen=True)
class TDMPC2Config(OffPolicyConfig):
    # World model (official 5M-parameter defaults).
    latent_dim: int = 512
    mlp_dim: int = 512
    enc_dim: int = 256
    num_enc_layers: int = 2
    num_q: int = 5
    simnorm_dim: int = 8
    dropout: float = 0.01
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0
    episodic: bool = True  # learn a termination head and stop value at predicted terminals
    # Planning.
    mpc: bool = True
    horizon: int = 3
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5
    # Optimization.
    rho: float = 0.5
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    termination_coef: float = 1.0
    entropy_coef: float = 1e-4
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    enc_lr_scale: float = 0.3
    grad_clip_norm: float = 20.0
    tau: float = 0.01
    discount: float | None = None  # None: clip((L/5 - 1) / (L/5), 0.95, 0.995) for episode length L

    def __post_init__(self):
        super().__post_init__()
        for name in ("latent_dim", "mlp_dim", "enc_dim", "num_enc_layers", "simnorm_dim", "horizon",
                     "iterations", "num_samples", "num_elites"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"tdmpc2.{name} must be a positive integer")
        if self.latent_dim % self.simnorm_dim:
            raise ValueError("tdmpc2.latent_dim must be divisible by simnorm_dim")
        if self.num_q < 2 or self.num_bins < 2 or not self.vmin < self.vmax:
            raise ValueError("tdmpc2 needs num_q >= 2, num_bins >= 2, and vmin < vmax")
        if not 0 <= self.num_pi_trajs < self.num_samples or not self.num_elites <= self.num_samples:
            raise ValueError("tdmpc2 needs num_pi_trajs < num_samples and num_elites <= num_samples")
        if not 0 < self.min_std <= self.max_std or self.log_std_min >= self.log_std_max:
            raise ValueError("tdmpc2 standard deviation bounds must be ordered")
        if not 0 <= self.dropout < 1 or not 0 < self.rho <= 1 or not 0 < self.tau <= 1:
            raise ValueError("tdmpc2 dropout, rho, or tau out of range")
        if self.discount is not None and not 0 < self.discount <= 1:
            raise ValueError("tdmpc2.discount must be in (0, 1]")


Config = TDMPC2Config


def symlog(x: Array) -> Array:
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: Array) -> Array:
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


class World(eqx.Module):
    encoder: MLP
    dynamics: MLP
    reward: MLP
    termination: MLP
    qs: MLP  # ensemble


class Model(eqx.Module):
    world: World
    policy: MLP


class TrainState(eqx.Module):
    model: Model
    target_qs: MLP
    world_optimizer: Any
    policy_optimizer: Any
    scale: Array  # running 5–95 percentile range of Q, for scale-free policy loss


class TDMPC2:
    def __init__(self, config: TDMPC2Config, observation_size: int, action_size: int,
                 episode_length: int, training: TrainingConfig):
        self.config = config
        self.observation_size, self.action_size = observation_size, action_size
        self.horizon = config.horizon
        self.discount = offpolicy.discount(config.discount, episode_length)
        self.iterations = config.iterations + 2 * (action_size >= 20)  # official large-action heuristic
        self.bins = jnp.linspace(config.vmin, config.vmax, config.num_bins)
        clip = optax.clip_by_global_norm(config.grad_clip_norm)
        self.world_optimizer = optax.chain(clip, optax.adam(training.learning_rate))
        self.policy_optimizer = optax.chain(clip, optax.adam(training.learning_rate, eps=1e-5))

    # ----- networks -----------------------------------------------------------------
    def init_model(self, key: Array) -> Model:
        c, latent, action = self.config, self.config.latent_dim, self.action_size
        keys = jax.random.split(key, 6)
        hidden = (c.mlp_dim, c.mlp_dim)
        group = c.simnorm_dim
        world = World(
            encoder=MLP(keys[0], self.observation_size, (c.enc_dim,) * max(c.num_enc_layers - 1, 1), latent,
                        output="simnorm", group=group),
            dynamics=MLP(keys[1], latent + action, hidden, latent, output="simnorm", group=group),
            reward=MLP(keys[2], latent + action, hidden, c.num_bins, zero=True),
            termination=MLP(keys[3], latent, hidden, 1),
            qs=MLP(keys[4], latent + action, hidden, c.num_bins, ensemble=c.num_q, dropout=c.dropout, zero=True))
        return Model(world, MLP(keys[5], latent, hidden, 2 * action))

    def init(self, key: Array) -> TrainState:
        model = self.init_model(key)
        target_qs = jax.tree.map(jnp.copy, model.world.qs)  # distinct buffers: training donates state
        return TrainState(model, target_qs, self.world_optimizer.init(eqx.filter(model.world, eqx.is_array)),
                          self.policy_optimizer.init(eqx.filter(model.policy, eqx.is_array)), jnp.ones(()))

    def init_carry(self, envs: int) -> Array:
        return jnp.zeros((envs, self.horizon, self.action_size))  # previous MPPI mean per env

    @staticmethod
    def next(world: World, z: Array, action: Array) -> Array:
        return world.dynamics(jnp.concatenate((z, action), -1))

    def decode(self, logits: Array) -> Array:
        return symexp(einx.dot("... k, k -> ...", jax.nn.softmax(logits, -1), self.bins))  # pyright: ignore[reportPrivateImportUsage]

    def two_hot(self, x: Array) -> Array:
        c = self.config
        position = (jnp.clip(symlog(x), c.vmin, c.vmax) - c.vmin) / ((c.vmax - c.vmin) / (c.num_bins - 1))
        lower = jnp.floor(position)
        upper_weight = position - lower
        lower = lower.astype(jnp.int32)
        upper = jnp.minimum(lower + 1, c.num_bins - 1)
        return (jax.nn.one_hot(lower, c.num_bins) * (1 - upper_weight)[..., None]
                + jax.nn.one_hot(upper, c.num_bins) * upper_weight[..., None])

    def soft_ce(self, logits: Array, target: Array) -> Array:
        return -jnp.sum(self.two_hot(target) * jax.nn.log_softmax(logits, -1), -1)

    def pi(self, policy: MLP, z: Array, key: Array) -> tuple[Array, Array]:
        """Reparameterized tanh-Gaussian action and TD-MPC2's scaled entropy, which is
        `-action_size * log N(eps)` (the official ratio form reduces to this)."""
        c = self.config
        mean, raw = jnp.split(policy(z), 2, -1)
        log_std = c.log_std_min + 0.5 * (c.log_std_max - c.log_std_min) * (jnp.tanh(raw) + 1)
        noise = jax.random.normal(key, mean.shape)
        log_prob = jnp.sum(-0.5 * noise ** 2 - log_std, -1) - 0.5 * jnp.log(2 * jnp.pi) * self.action_size
        return jnp.tanh(mean + noise * jnp.exp(log_std)), -self.action_size * log_prob

    def q(self, qs: MLP, z: Array, action: Array, key: Array, reduce: str, dropout: Array | None = None) -> Array:
        """Min or mean of two random heads; evaluates only those two members."""
        members = jax.random.choice(key, self.config.num_q, (2,), replace=False)
        values = self.decode(qs.select(members)(jnp.concatenate((z, action), -1), dropout))
        return values.min(0) if reduce == "min" else values.mean(0)

    # ----- planning -----------------------------------------------------------------
    def estimate_value(self, model: Model, z: Array, actions: Array, key: Array) -> Array:
        world, total, scale = model.world, 0.0, 1.0
        terminated = jnp.zeros(z.shape[:-1])
        for t in range(self.horizon):
            reward = self.decode(world.reward(jnp.concatenate((z, actions[t]), -1)))
            z = self.next(world, z, actions[t])
            total = total + scale * (1 - terminated) * reward
            scale *= self.discount
            if self.config.episodic:
                terminated = jnp.maximum(terminated, (world.termination(z)[..., 0] > 0).astype(z.dtype))
        policy_key, q_key = jax.random.split(key)
        action, _ = self.pi(model.policy, z, policy_key)
        return total + scale * (1 - terminated) * self.q(world.qs, z, action, q_key, "avg")

    def plan(self, model: Model, z: Array, previous: Array, first: Array, key: Array,
             explore: bool) -> tuple[Array, Array]:
        """MPPI for one environment from latent `z`; returns the action and new mean."""
        c, horizon, size = self.config, self.horizon, self.action_size
        prior_key, iterate_key, choice_key, noise_key = jax.random.split(key, 4)
        prior = jnp.zeros((horizon, 0, size))
        if c.num_pi_trajs:
            latent, actions = jnp.broadcast_to(z, (c.num_pi_trajs, z.shape[-1])), []
            for t, step_key in enumerate(jax.random.split(prior_key, horizon)):
                action, _ = self.pi(model.policy, latent, step_key)
                actions.append(action)
                if t < horizon - 1:
                    latent = self.next(model.world, latent, action)
            prior = jnp.stack(actions)
        latents = jnp.broadcast_to(z, (c.num_samples, z.shape[-1]))
        shifted = jnp.concatenate((previous[1:], jnp.zeros((1, size))))
        mean = jnp.where(first, jnp.zeros_like(shifted), shifted)
        std = jnp.full((horizon, size), c.max_std)

        def iterate(carry, key):
            mean, std, _, _ = carry
            sample_key, value_key = jax.random.split(key)
            noise = jax.random.normal(sample_key, (horizon, c.num_samples - c.num_pi_trajs, size))
            actions = jnp.concatenate((prior, jnp.clip(mean[:, None] + std[:, None] * noise, -1, 1)), 1)
            value = jnp.nan_to_num(self.estimate_value(model, latents, actions, value_key), nan=0.0)
            elite_value, index = jax.lax.top_k(value, c.num_elites)
            elites = actions[:, index]
            score = jax.nn.softmax(c.temperature * (elite_value - elite_value.max()))
            mean = einx.dot("e, h e a -> h a", score, elites)  # pyright: ignore[reportPrivateImportUsage]
            spread = einx.dot("e, h e a -> h a", score, (elites - mean[:, None]) ** 2)  # pyright: ignore[reportPrivateImportUsage]
            return (mean, jnp.clip(jnp.sqrt(spread), c.min_std, c.max_std), score, elites), None

        initial = (mean, std, jnp.zeros(c.num_elites), jnp.zeros((horizon, c.num_elites, size)))
        (mean, std, score, elites), _ = jax.lax.scan(iterate, initial, jax.random.split(iterate_key, self.iterations))
        action = elites[0, jax.random.categorical(choice_key, jnp.log(score))]
        if explore:
            action = action + std[0] * jax.random.normal(noise_key, (size,))
        return jnp.clip(action, -1, 1), mean

    def act(self, model: Model, carry: Array, observation: Array, first: Array, key: Array,
            explore: bool) -> tuple[Array, Array]:
        z = model.world.encoder(observation)
        if not self.config.mpc:
            action, _ = self.pi(model.policy, z, key)
            if not explore:
                action = jnp.tanh(jnp.split(model.policy(z), 2, -1)[0])
            return action, carry
        keys = jax.random.split(key, observation.shape[0])
        return jax.vmap(lambda *args: self.plan(model, *args, explore=explore))(z, carry, first, keys)

    # ----- learning -----------------------------------------------------------------
    def update(self, state: TrainState, batch: Batch, key: Array) -> tuple[TrainState, dict[str, Array]]:
        c, horizon = self.config, self.horizon
        target_key, dropout_key, policy_key = jax.random.split(key, 3)
        world, policy = state.model.world, state.model.policy
        size = batch.reward.shape[1]

        # Targets come from the online encoder and the slow target critics.
        next_z = jax.lax.stop_gradient(world.encoder(batch.next_observation))
        action_key, q_key = jax.random.split(target_key)
        next_action, _ = self.pi(policy, next_z, action_key)
        td_target = jax.lax.stop_gradient(batch.reward + self.discount * (1 - batch.terminated)
                                          * self.q(state.target_qs, next_z, next_action, q_key, "min"))
        weight = (c.rho ** jnp.arange(horizon))[:, None] * batch.mask

        def world_loss(world: World):
            z = world.encoder(batch.observation[0])
            latents = [z]
            for t in range(horizon):
                z = self.next(world, z, batch.action[t])
                latents.append(z)
            zs = jnp.stack(latents)
            inputs = jnp.concatenate((zs[:-1], batch.action), -1)
            consistency = jnp.sum(jnp.mean((zs[1:] - next_z) ** 2, -1) * weight) / (horizon * size)
            reward = jnp.sum(self.soft_ce(world.reward(inputs), batch.reward) * weight) / (horizon * size)
            value = jnp.sum(self.soft_ce(world.qs(inputs, dropout_key), td_target) * weight) / (horizon * size * c.num_q)
            termination = jnp.zeros(())
            if c.episodic:
                logits = world.termination(zs[1:])[..., 0]
                termination = jnp.sum(optax.sigmoid_binary_cross_entropy(logits, batch.terminated)
                                      * batch.mask) / (horizon * size)
            total = (c.consistency_coef * consistency + c.reward_coef * reward
                     + c.value_coef * value + c.termination_coef * termination)
            return total, (zs, dict(consistency_loss=consistency, reward_loss=reward, value_loss=value,
                                    termination_loss=termination, total_loss=total))

        (_, (zs, metrics)), grads = eqx.filter_value_and_grad(world_loss, has_aux=True)(world)
        updates, world_optimizer = self.world_optimizer.update(grads, state.world_optimizer)
        updates = eqx.tree_at(lambda tree: tree.encoder, updates,
                              replace_fn=lambda encoder: jax.tree.map(lambda u: u * c.enc_lr_scale, encoder))
        world = eqx.apply_updates(world, updates)

        # Policy: maximize scale-normalized Q from the updated critics along detached latents.
        zs = jax.lax.stop_gradient(zs)
        critics = jax.lax.stop_gradient(world.qs)
        state_weight = (c.rho ** jnp.arange(horizon + 1))[:, None] * \
            jnp.concatenate((jnp.ones((1, size)), batch.mask))

        def policy_loss(policy: MLP):
            action_key, member_key, drop_key = jax.random.split(policy_key, 3)
            action, entropy = self.pi(policy, zs, action_key)
            q = self.q(critics, zs, action, member_key, "avg", drop_key)
            low, high = jnp.percentile(jax.lax.stop_gradient(q[0]), jnp.array([5.0, 95.0]))
            scale = state.scale + c.tau * (jnp.maximum(high - low, 1.0) - state.scale)
            objective = c.entropy_coef * entropy + q / scale
            return -jnp.sum(objective * state_weight) / ((horizon + 1) * size), (scale, jnp.mean(entropy))

        (pi_loss, (scale, entropy)), pi_grads = eqx.filter_value_and_grad(policy_loss, has_aux=True)(policy)
        pi_updates, policy_optimizer = self.policy_optimizer.update(pi_grads, state.policy_optimizer)
        policy = eqx.apply_updates(policy, pi_updates)
        target_qs = jax.tree.map(lambda target, online: target + c.tau * (online - target), state.target_qs, world.qs)
        metrics.update(grad_norm=optax.tree.norm(grads), pi_loss=pi_loss, pi_scaled_entropy=entropy,
                       pi_scale=scale, td_target=jnp.mean(td_target))
        return TrainState(Model(world, policy), target_qs, world_optimizer, policy_optimizer, scale), metrics


def train(env: Any, config: TDMPC2Config, training: TrainingConfig, output: str | Path, seed: int = 0):
    return offpolicy.train(env, "tdmpc2", TDMPC2, config, training, output, seed)


def load_agent(path: str | Path) -> offpolicy.OffPolicyAgent:
    return offpolicy.load_agent(path, "tdmpc2.eqx", TDMPC2, TDMPC2Config)
