from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maton.curriculum import CurriculumConfig, Level
from maton.offpolicy import OffPolicyConfig, Replay, seed_steps
from maton.terrain import TerrainConfig
from maton import tdmpc2
from maton.training import TrainingConfig


class ToyState(eqx.Module):
    x: jax.Array
    steps: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    level: jax.Array
    success: jax.Array


class ToyEnv:
    """Reward peaks at action 0.5; episodes last three actions."""
    action_size = 1
    config = SimpleNamespace(episode_length=3)
    curriculum = CurriculumConfig((Level("toy", 1, TerrainConfig()),))

    def reset(self, key, level=0):
        return ToyState(jax.random.uniform(key, (), minval=-0.1, maxval=0.1), jnp.array(0), jnp.array(0.),
                        jnp.array(False), jnp.array(False), jnp.asarray(level), jnp.array(False))

    def observation(self, state):
        return jnp.array([state.x, state.steps / 3])

    def step(self, state, action):
        return ToyState(state.x + action[0] * 0.1, state.steps + 1, 1. - (action[0] - 0.5) ** 2,
                        jnp.array(False), state.steps >= 2, state.level, jnp.array(False))


class FinishingEnv(ToyEnv):
    flat = TerrainConfig(block_frequency=0, gap_frequency=0)
    curriculum = CurriculumConfig((Level("one", 2, flat), Level("two", 2, flat)))

    def step(self, state, action):
        result = super().step(state, action)
        return eqx.tree_at(lambda s: (s.terminated, s.truncated, s.success), result,
                           (jnp.array(True), jnp.array(False), jnp.array(True)))


TINY = dict(num_envs=2, batch_size=8, seed_steps=8, log_every=8, latent_dim=16, mlp_dim=32, enc_dim=16,
            num_q=2, num_samples=32, num_elites=4, num_pi_trajs=4, iterations=2)


def test_replay_windows_stay_in_time_order_and_mask_after_episode_ends():
    replay = Replay.create(4, 1, 1, 1)
    for t in range(6):  # Capacity 4: rows for t = 2..5 survive; t = 3 ends an episode.
        replay = replay.add(jnp.full((1, 1), t), jnp.zeros((1, 1)), jnp.full(1, t), jnp.full((1, 1), t + 1),
                            jnp.array([t == 3]), jnp.array([t == 3]))
    batch = replay.sample(jax.random.key(0), 256, 2)
    start = np.asarray(batch.observation[0, :, 0])
    assert set(start) == {2, 3, 4}
    np.testing.assert_array_equal(batch.observation[1, :, 0], start + 1)
    np.testing.assert_array_equal(batch.mask[1], start != 3)
    np.testing.assert_array_equal(batch.terminated[0], start == 3)


def test_seed_steps_default_and_rounding():
    assert seed_steps(OffPolicyConfig(num_envs=4), 512, 10_000) == 2560
    assert seed_steps(OffPolicyConfig(num_envs=3, seed_steps=10), 3, 100) == 12
    assert seed_steps(OffPolicyConfig(), 512, 100) == 100
    with pytest.raises(ValueError, match="update_ratio"):
        OffPolicyConfig(update_ratio=0)


def test_training_budget_checkpoints_and_deterministic_evaluation(tmp_path):
    config = tdmpc2.TDMPC2Config(**TINY)
    result = tdmpc2.train(ToyEnv(), config, TrainingConfig(steps=24, checkpoint_every=10), tmp_path, seed=3)
    assert result["steps"] == 24 and result["updates"] == 24  # 8 pretraining + 2 per later step pair
    assert np.isfinite(result["last"]["total_loss"])
    assert sorted(p.name for p in (tmp_path / "checkpoints").glob("*.eqx")) == [
        "step-00000010.eqx", "step-00000020.eqx", "step-00000024.eqx"]
    agent = tdmpc2.load_agent(tmp_path)
    first = agent(np.zeros(2, np.float32))
    agent.reset()
    assert np.array_equal(first, agent(np.zeros(2, np.float32)))
    assert first.shape == (1,) and np.all(np.abs(first) <= 1)


def test_update_ratio_scales_gradient_steps_and_budget_must_divide(tmp_path):
    config = tdmpc2.TDMPC2Config(**TINY | dict(update_ratio=0.25))
    result = tdmpc2.train(ToyEnv(), config, TrainingConfig(steps=24), tmp_path / "a")
    assert result["updates"] == 2 + 4  # round(8 × 0.25) pretraining, then 16 × 0.25
    with pytest.raises(ValueError, match="divisible"):
        tdmpc2.train(ToyEnv(), config, TrainingConfig(steps=23), tmp_path / "b")


def test_curriculum_advances_on_successes(tmp_path):
    result = tdmpc2.train(FinishingEnv(), tdmpc2.TDMPC2Config(**TINY), TrainingConfig(steps=8), tmp_path)
    assert result["curriculum"]["level"] == 1 and result["curriculum"]["complete"]
    assert tdmpc2.load_agent(tmp_path).curriculum_level == 1
