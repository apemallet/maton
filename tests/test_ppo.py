import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maton.ppo import PPOConfig, Policy, action, advantages, load_policy, log_probability, train
from maton.training import TrainingConfig
from maton.curriculum import CurriculumConfig, Level
from maton.terrain import TerrainConfig


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
    level: jax.Array
    success: jax.Array


class ToyEnv:
    action_size = 1
    curriculum = CurriculumConfig((Level("toy", 1, TerrainConfig()),))

    def reset(self, key, level=0):
        return ToyState(jax.random.uniform(key, (), minval=-0.1, maxval=0.1),
                        jnp.array(0), jnp.array(0.), jnp.array(False), jnp.array(False),
                        jnp.asarray(level), jnp.array(False))

    def observation(self, state):
        return jnp.array([state.x, state.steps / 3])

    def step(self, state, action):
        return ToyState(state.x + action[0] * 0.1, state.steps + 1,
                        1. - (action[0] - 0.5) ** 2, jnp.array(False), state.steps >= 2,
                        state.level, jnp.array(False))


def test_training_updates_and_checkpoint_roundtrip(tmp_path):
    config = PPOConfig(num_envs=2, rollout_steps=4, epochs=1,
                       minibatch_size=4, hidden_size=8)
    training = TrainingConfig(steps=16, checkpoint_every=8)
    result = train(ToyEnv(), config, training, tmp_path, seed=12)
    assert result['steps'] == 16
    assert result['episodes'] >= 2
    assert result['last']['grad_norm'] > 0
    assert np.isfinite(list(result['last'].values())).all()
    loaded = load_policy(tmp_path)
    checkpoint = load_policy(tmp_path / 'checkpoints/step-00000016.eqx')
    np.testing.assert_array_equal(action(loaded, jnp.ones(2)), action(checkpoint, jnp.ones(2)))
    key = jax.random.PRNGKey(12)
    _, _, policy_key = jax.random.split(key, 3)
    initial = Policy(2, 1, 8, policy_key)
    assert not np.array_equal(initial.actor.layers[0].weight, loaded.actor.layers[0].weight)
    assert json.loads((tmp_path / 'policy.json').read_text())['step'] == 16


def test_exact_partial_rollout_budget_and_crossed_checkpoint_thresholds(tmp_path):
    config = PPOConfig(num_envs=2, rollout_steps=4, epochs=1,
                       minibatch_size=4, hidden_size=8)
    result = train(ToyEnv(), config, TrainingConfig(steps=18, checkpoint_every=5), tmp_path)
    assert result['steps'] == 18
    assert result['updates'] == 3
    assert sorted(p.name for p in (tmp_path / 'checkpoints').glob('*.eqx')) == [
        'step-00000008.eqx', 'step-00000016.eqx', 'step-00000018.eqx']
    assert json.loads((tmp_path / 'policy.json').read_text())['training']['steps'] == 18


def test_budget_must_fit_parallel_environment_count(tmp_path):
    with pytest.raises(ValueError, match='divisible'):
        train(ToyEnv(), PPOConfig(), TrainingConfig(steps=3), tmp_path)


def test_config_rejects_partial_minibatch():
    with pytest.raises(ValueError, match='divisible'):
        PPOConfig(num_envs=1, rollout_steps=7, minibatch_size=3)


def test_batched_training_advances_only_on_successes(tmp_path):
    flat = TerrainConfig(block_frequency=0, gap_frequency=0)

    class SuccessfulEnv(ToyEnv):
        curriculum = CurriculumConfig((Level("one", 2, flat), Level("two", 2, flat)))

        def step(self, state, action):
            result = super().step(state, action)
            return eqx.tree_at(lambda s: (s.terminated, s.truncated, s.success), result,
                               (jnp.array(True), jnp.array(False), jnp.array(True)))

    config = PPOConfig(num_envs=2, rollout_steps=4, epochs=1, minibatch_size=4, hidden_size=8)
    result = train(SuccessfulEnv(), config, TrainingConfig(steps=8), tmp_path)
    assert result['curriculum']['level'] == 1
    assert result['curriculum']['complete']
    assert load_policy(tmp_path).curriculum_level == 1


def test_configurable_initial_exploration():
    policy = Policy(3, 4, 8, jax.random.PRNGKey(0), initial_std=0.2)
    np.testing.assert_allclose(jnp.exp(policy.log_std), 0.2)
    with pytest.raises(ValueError, match='initial_std'):
        PPOConfig(initial_std=0.0)


def test_pause_and_resume_matches_uninterrupted_training(tmp_path):
    config = PPOConfig(num_envs=2, rollout_steps=4, epochs=1, minibatch_size=4, hidden_size=8)
    training = TrainingConfig(steps=32, checkpoint_every=16)
    straight = train(ToyEnv(), config, training, tmp_path / "straight", seed=4)
    paused = train(ToyEnv(), config, training, tmp_path / "paused", seed=4, pause=lambda steps: steps >= 8)
    assert not paused["complete"] and paused["steps"] == 8
    resumed = train(ToyEnv(), config, training, tmp_path / "paused", seed=4)
    assert resumed["complete"] and resumed["episodes"] == straight["episodes"]
    np.testing.assert_array_equal(load_policy(tmp_path / "straight").actor.layers[0].weight,
                                  load_policy(tmp_path / "paused").actor.layers[0].weight)
    assert (tmp_path / "paused/metrics.csv").read_text() .count("update") == 1  # one header
    assert len((tmp_path / "paused/metrics.csv").read_text().splitlines()) == 5
