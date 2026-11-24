import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from maton.env import EnvConfig, WalkerEnv
from maton.terrain import TerrainConfig


@pytest.fixture(scope='module')
def env():
    return WalkerEnv(EnvConfig(episode_length=8, reset_noise=0), TerrainConfig(slots=1))


def test_planar_model_senses_and_lidar(env):
    assert env.model.nq == env.model.nv == 7
    assert env.model.nu == 4
    state = jax.jit(env.reset)(jax.random.key(0))
    obs = env.observation(state)
    assert obs.shape == (env.observation_size,)
    assert np.all(np.isfinite(obs))
    assert float(env.lidar(state)[0]) == pytest.approx(1 / env.config.lidar_range, abs=1e-5)
    assert np.all(np.asarray(env.lidar(state)) >= 0)
    assert np.all(np.asarray(env.lidar(state)) <= 1)
    model = env.cpu_model(state)
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(state.data.qpos)
    mujoco.mj_forward(model, data)
    np.testing.assert_allclose(data.geom_xpos, state.data.geom_xpos, atol=1e-5)


def test_jitted_rollout_potential_reward_and_timeout(env):
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.key(0))
    for _ in range(8):
        previous_x = float(state.data.qpos[0])
        state = step(state, jnp.zeros(4))
        assert np.all(np.isfinite(env.observation(state)))
        assert float(state.reward) == pytest.approx(float(state.data.qpos[0]) - previous_x, abs=1e-6)
    assert bool(state.truncated)
    assert not bool(state.terminated)
    assert float(state.data.sensordata.sum()) > 0


def test_fall_terminates_and_action_is_clipped(env):
    state = jax.jit(env.reset)(jax.random.key(0))
    state = state.__class__(state.data.replace(qpos=state.data.qpos.at[1].set(-0.7)),
                            state.terrain, state.steps, state.reward, state.terminated,
                            state.truncated, state.action)
    state = jax.jit(env.step)(state, jnp.full(4, 100))
    assert bool(state.terminated)
    np.testing.assert_array_equal(state.action, np.ones(4))
