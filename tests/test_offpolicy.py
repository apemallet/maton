from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maton.curriculum import CurriculumConfig, Level
from maton.offpolicy import OffPolicyConfig, Replay, seed_steps
from maton.terrain import TerrainConfig
from maton import sac, tdmpc2
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


@pytest.mark.parametrize("module", [tdmpc2, sac])
def test_pause_and_resume_matches_uninterrupted_training(tmp_path, module):
    config = (tdmpc2.TDMPC2Config(**TINY) if module is tdmpc2 else
              sac.SACConfig(num_envs=2, batch_size=8, seed_steps=8, log_every=8, hidden_size=16))
    training = TrainingConfig(steps=24, checkpoint_every=10)
    straight = module.train(ToyEnv(), config, training, tmp_path / "straight", seed=5)
    paused = module.train(ToyEnv(), config, training, tmp_path / "paused", seed=5, pause=lambda steps: steps >= 12)
    assert not paused["complete"] and paused["steps"] == 12
    resumed = module.train(ToyEnv(), config, training, tmp_path / "paused", seed=5)
    assert resumed["complete"] and resumed["updates"] == straight["updates"]
    assert resumed["episodes"] == straight["episodes"]
    name = module.__name__.split(".")[-1]
    for left, right in zip(jax.tree.leaves(eqx.tree_deserialise_leaves(tmp_path / "straight" / f"{name}.eqx", module.load_agent(tmp_path / "straight").model)),
                           jax.tree.leaves(module.load_agent(tmp_path / "paused").model)):
        np.testing.assert_array_equal(left, right)
    rows = (tmp_path / "paused" / "metrics.jsonl").read_text().splitlines()
    assert len(rows) == len((tmp_path / "straight" / "metrics.jsonl").read_text().splitlines())


def test_resume_without_replay_refills_before_updating(tmp_path):
    config = tdmpc2.TDMPC2Config(**TINY)
    training = TrainingConfig(steps=24)
    tdmpc2.train(ToyEnv(), config, training, tmp_path, pause=lambda steps: steps >= 12)
    for path in (tmp_path / "resume").glob("replay.*"):  # e.g. resuming on another machine
        path.unlink()
    result = tdmpc2.train(ToyEnv(), config, training, tmp_path)
    assert result["complete"] and result["updates"] == 12 + 6  # no updates while 8 transitions refill (steps 14–18)


@pytest.mark.parametrize('module', [sac, tdmpc2])
def test_continuation_loads_model_with_fresh_targets_and_optimizer(tmp_path, module):
    if module is sac:
        config = sac.SACConfig(num_envs=2, batch_size=4, seed_steps=4, hidden_size=8, num_q=2)
    else:
        config = tdmpc2.TDMPC2Config(**TINY)
    source = tmp_path / 'source'
    module.train(FinishingEnv(), config, TrainingConfig(steps=8), source, seed=3)
    checkpoint = source / f'{module.__name__.split(".")[-1]}.eqx'
    original = module.load_agent(checkpoint).model
    child = tmp_path / 'child'
    result = module.train(ToyEnv(), config, TrainingConfig(steps=8, learning_rate=1e-30), child,
                          seed=99, checkpoint=checkpoint)
    assert result['steps'] == 8 and result['curriculum']['level'] == 0
    loaded = module.load_agent(child).model
    for before, after in zip(jax.tree.leaves(original), jax.tree.leaves(loaded)):
        np.testing.assert_allclose(before, after, rtol=0, atol=1e-25)


def test_efficiency_switch_clears_old_rewards_and_survives_resume(tmp_path):
    import json
    from maton import sessions

    class EfficientEnv(FinishingEnv):
        curriculum = CurriculumConfig((Level('hard', 2, TerrainConfig()),
                                       Level('efficient', 2, TerrainConfig(), efficiency_penalty=2.0)))

        def step_with_efficiency(self, state, action, weight):
            following = self.step(state, action)
            return eqx.tree_at(lambda s: s.reward, following, following.reward * 0 + 1 - weight)

    config = sac.SACConfig(num_envs=2, hidden_size=8, batch_size=4, seed_steps=2, log_every=2, update_ratio=0.5)
    training = TrainingConfig(steps=8, checkpoint_every=2)
    result = sac.train(EfficientEnv(), config, training, tmp_path, pause=lambda n: n >= 6)
    assert not result['complete'] and result['curriculum']['efficiency_weight'] == 2
    assert result['curriculum']['level'] == 1
    host = json.loads((tmp_path / 'resume/state.json').read_text())
    assert host['stored'] == 2 and host['refill_until'] == 4
    saved = sessions.load(tmp_path, Replay.create(4, 2, 2, 1), 'replay')
    assert saved is not None
    replay, _ = saved
    np.testing.assert_array_equal(replay.reward[:int(replay.rows)], -np.ones((2, 2)))
    logs = [json.loads(line) for line in (tmp_path / 'metrics.jsonl').read_text().splitlines()]
    assert logs[0]['updates'] == 0  # no update using the old objective at the switch
    assert logs[0]['episode_return'] == 1 and logs[1]['episode_return'] == -1
    final = sac.train(EfficientEnv(), config, training, tmp_path)
    assert final['complete'] and final['curriculum']['efficiency_weight'] == 2


def test_reward_uses_each_parallel_episode_level():
    from maton.training import environment_step

    class EfficientEnv(ToyEnv):
        curriculum = CurriculumConfig((Level('hard', 2, TerrainConfig()),
                                       Level('efficient', 2, TerrainConfig(), efficiency_penalty=2.0)))

        def step_with_efficiency(self, state, action, weight):
            following = self.step(state, action)
            return eqx.tree_at(lambda s: s.reward, following, jnp.float32(1) - weight)

    env = EfficientEnv()
    states = jax.vmap(env.reset)(jax.random.split(jax.random.key(0), 2), jnp.array([0, 1]))
    following = jax.jit(jax.vmap(lambda state, action: environment_step(env, state, action)))(states, jnp.zeros((2, 1)))
    np.testing.assert_array_equal(following.reward, [1, -1])
