import jax
import equinox as eqx
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


def test_fall_allows_recovery_and_action_is_clipped(env):
    state = jax.jit(env.reset)(jax.random.key(0))
    state = eqx.tree_at(lambda s: s.data.qpos, state, state.data.qpos.at[1].set(-0.7))
    state = jax.jit(env.step)(state, jnp.full(4, 100))
    assert not bool(state.terminated)
    np.testing.assert_array_equal(state.action, np.ones(4))


def test_position_motor_control_is_neutral_at_targets_and_damped():
    env = WalkerEnv(EnvConfig(control_mode='position'), TerrainConfig(slots=1))
    target = jnp.array([-.3, .65, -.3, .65])
    data = env._data.replace(qpos=env._data.qpos.at[3:].set(target))
    np.testing.assert_allclose(env.motor_control(data, jnp.zeros(4)), 0, atol=1e-6)
    data = data.replace(qvel=data.qvel.at[3:].set(1))
    np.testing.assert_allclose(env.motor_control(data, jnp.zeros(4)), -.1, atol=1e-6)
    np.testing.assert_array_less(np.abs(env.motor_control(data, jnp.ones(4))), 1.00001)


def test_cpu_model_has_correct_collision_bvh_away_from_origin(env):
    state = jax.jit(env.reset)(jax.random.key(0))
    model = env.cpu_model(state)
    data = mujoco.MjData(model)
    data.qpos[0] = env.terrain_config.length - .2
    for _ in range(100):
        mujoco.mj_step(model, data)
    assert 1 + data.qpos[1] > .5
    assert data.ncon > 0


@pytest.mark.parametrize('kwargs', [dict(control_mode='bad'), dict(torque_limit=0),
                                    dict(position_kd=-1), dict(hip_target_min=-2),
                                    dict(knee_target_max=3), dict(upright_reward=-1)])
def test_invalid_motor_settings(kwargs):
    with pytest.raises(ValueError):
        EnvConfig(**kwargs)


def test_success_allows_recovered_full_rotation(env):
    state = jax.jit(env.reset)(jax.random.key(0))
    state = eqx.tree_at(lambda s: s.data.qpos, state,
          state.data.qpos.at[0].set(env.terrain_config.length + .01).at[2].set(2 * jnp.pi))
    state = jax.jit(env.step)(state, jnp.zeros(4))
    assert bool(state.success)


def test_fallen_beyond_finish_gets_no_runoff_reward(env):
    state = jax.jit(env.reset)(jax.random.key(0))
    state = eqx.tree_at(lambda s: s.data.qpos, state,
                        state.data.qpos.at[0].set(env.terrain_config.length + 1).at[2].set(jnp.pi / 2))
    state = eqx.tree_at(lambda s: s.data.qvel, state, state.data.qvel.at[0].set(2))
    state = jax.jit(env.step)(state, jnp.zeros(4))
    assert float(state.reward) == pytest.approx(0, abs=1e-6)
    assert not bool(state.success)


def test_capsule_foot_neutral_stance_and_target_endpoints():
    cfg = EnvConfig(control_mode='position', foot_length=.16, reset_noise=0,
                    hip_target_min=-.8, hip_target_max=.8, hip_target_neutral=0,
                    knee_target_min=-.08, knee_target_max=1.3, knee_target_neutral=0,
                    finish_reward=25)
    env = WalkerEnv(cfg, TerrainConfig(slots=1))
    state = jax.jit(env.reset)(jax.random.key(0))
    np.testing.assert_allclose(env.motor_control(state.data, jnp.zeros(4)), 0, atol=1e-6)
    model = env.cpu_model(state)
    data = mujoco.MjData(model)
    for _ in range(2000):
        data.ctrl[:] = np.clip((-30 * data.qpos[3:] - 2 * data.qvel[3:]) / 20, -1, 1)
        mujoco.mj_step(model, data)
    assert 1 + data.qpos[1] > .8
    assert abs(data.qpos[2]) < .1
    terminal = eqx.tree_at(lambda s: s.data.qpos, state,
                         state.data.qpos.at[0].set(env.terrain_config.length + .01))
    terminal = jax.jit(env.step)(terminal, jnp.zeros(4))
    assert bool(terminal.success)
    assert float(terminal.reward) == pytest.approx(25, abs=1e-5)


def test_upright_potential_loses_progress_on_fall_and_restores_on_recovery(env):
    upright_env = WalkerEnv(EnvConfig(progress_mode='upright', progress_scale=10), env.terrain_config)
    data = upright_env._data.replace(qpos=jnp.zeros(7).at[0].set(2))
    assert float(upright_env.potential(data)) == pytest.approx(20)
    fallen = data.replace(qpos=data.qpos.at[1].set(-.7))
    fall_reward = upright_env.potential(fallen) - upright_env.potential(data)
    recover_reward = upright_env.potential(data) - upright_env.potential(fallen)
    assert float(fall_reward) == pytest.approx(-20)
    assert float(recover_reward) == pytest.approx(20)
    assert float(fall_reward + recover_reward) == pytest.approx(0)
    forward = data.replace(qpos=data.qpos.at[0].set(2.5))
    assert float(upright_env.potential(forward)) > float(upright_env.potential(data))
    rolled = data.replace(qpos=data.qpos.at[2].set(2 * jnp.pi))
    assert float(upright_env.potential(rolled)) == pytest.approx(20)


def test_upright_progress_adds_position_observation_and_step_rewards_telescope(env):
    upright_env = WalkerEnv(EnvConfig(progress_mode='upright', control_cost=0), env.terrain_config)
    state = jax.jit(env.reset)(jax.random.key(0))
    assert env.observation_size == 29
    assert upright_env.observation_size == 30
    assert env.observation(state).shape == (29,)
    assert upright_env.observation(state).shape == (30,)
    assert float(upright_env.observation(state)[-1]) == pytest.approx(float(state.data.qpos[0]) / env.terrain_config.length)
    initial_potential = float(upright_env.potential(state.data))
    total = 0
    step = jax.jit(upright_env.step)
    for _ in range(16):
        state = step(state, jnp.array([.1, -.1, -.1, .1]))
        total += float(state.reward)
    assert total == pytest.approx(float(upright_env.potential(state.data)) - initial_potential, abs=1e-5)


def test_invalid_progress_mode():
    with pytest.raises(ValueError):
        EnvConfig(progress_mode='invalid')
