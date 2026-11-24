import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maton.ppo import PPOConfig, Policy, action, advantages, load_policy, log_probability, train


def test_advantages_bootstrap_timeout_but_stop_at_reset():
    advantage, target = advantages(
        jnp.array([[1.], [9.]]), jnp.zeros((2, 1)), jnp.array([[2.], [3.]]),
        jnp.array([[False], [True]]), jnp.array([[True], [False]]), 0.9, 0.95)
    np.testing.assert_allclose(advantage[:, 0], [2.8, 9.])
    np.testing.assert_allclose(target, advantage)


def test_policy_log_probability_is_finite_at_saturation():
    values = log_probability(jnp.zeros((2, 4)), jnp.zeros(4), jnp.full((2, 4), 40.))
    assert np.isfinite(values).all()
    policy = Policy(3, 4, 8, jax.random.PRNGKey(0))
    assert action(policy, jnp.zeros(3)).shape == (4,)
    assert np.all(np.abs(action(policy, jnp.ones(3))) <= 1)


class ToyState(eqx.Module):
    x: jax.Array
    steps: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array


class ToyEnv:
    action_size = 1

    def reset(self, key):
        return ToyState(jax.random.uniform(key, (), minval=-0.1, maxval=0.1),
                        jnp.array(0), jnp.array(0.), jnp.array(False), jnp.array(False))

    def observation(self, state):
        return jnp.array([state.x, state.steps / 3])

    def step(self, state, action):
        return ToyState(state.x + action[0] * 0.1, state.steps + 1,
                        1. - (action[0] - 0.5) ** 2, jnp.array(False), state.steps >= 2)


def test_training_updates_and_checkpoint_roundtrip(tmp_path):
    config = PPOConfig(updates=2, num_envs=2, rollout_steps=4, epochs=1,
                       minibatch_size=4, hidden_size=8)
    result = train(ToyEnv(), config, tmp_path, seed=12)
    assert result['steps'] == 16
    assert result['episodes'] >= 2
    assert result['last']['grad_norm'] > 0
    assert np.isfinite(list(result['last'].values())).all()
    loaded = load_policy(tmp_path)
    checkpoint = load_policy(tmp_path / 'checkpoints/update-000002.eqx')
    np.testing.assert_array_equal(action(loaded, jnp.ones(2)), action(checkpoint, jnp.ones(2)))
    key = jax.random.PRNGKey(12)
    _, _, policy_key = jax.random.split(key, 3)
    initial = Policy(2, 1, 8, policy_key)
    assert not np.array_equal(initial.actor.layers[0].weight, loaded.actor.layers[0].weight)
    assert json.loads((tmp_path / 'policy.json').read_text())['step'] == 16


def test_config_rejects_partial_minibatch():
    with pytest.raises(ValueError, match='divisible'):
        PPOConfig(num_envs=1, rollout_steps=7, minibatch_size=3)
