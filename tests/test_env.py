import dataclasses
import jax
import equinox as eqx
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest

from maton.env import EnvConfig, WalkerEnv
from maton.terrain import TerrainConfig
from maton.walker import WalkerConfig


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
    assert not bool(state.terminated) and int(state.fallen_steps) == 1
    np.testing.assert_array_equal(state.action, np.ones(4))


def test_staying_fallen_for_the_timeout_ends_the_episode():
    env = WalkerEnv(EnvConfig(reset_noise=0, fall_timeout=0.1, episode_length=100), TerrainConfig(slots=1))
    steps_allowed = round(0.1 / (env.config.timestep * env.config.frame_skip))  # 5 steps of 0.02 s
    state = jax.jit(env.reset)(jax.random.key(0))
    step = jax.jit(env.step)
    lying = state.data.qpos.at[2].set(jnp.pi / 2)  # torso horizontal
    for index in range(steps_allowed):
        state = step(eqx.tree_at(lambda s: s.data.qpos, state, lying), jnp.zeros(4))
        assert bool(state.terminated) == (index == steps_allowed - 1)
    assert not bool(state.success)
    # Getting back up resets the clock.
    upright = step(eqx.tree_at(lambda s: (s.data.qpos, s.fallen_steps), state, (lying.at[2].set(0), jnp.array(3))),
                   jnp.zeros(4))
    assert int(upright.fallen_steps) == 0


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


@pytest.mark.parametrize('kwargs', [dict(control_mode='bad'), dict(upright_reward=-1)])
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
    assert not bool(state.success) and bool(state.terminated)


def test_capsule_foot_neutral_stance_and_target_endpoints():
    cfg = EnvConfig(control_mode='position', reset_noise=0, finish_reward=25)
    walker = WalkerConfig.load('configs/walkers/biped.toml')
    env = WalkerEnv(cfg, TerrainConfig(slots=1), walker=walker)
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


def test_observation_is_fourier_encoded_and_progress_mode_independent_and_step_rewards_telescope(env):
    upright_env = WalkerEnv(EnvConfig(progress_mode='upright', control_cost=0), env.terrain_config)
    state = jax.jit(env.reset)(jax.random.key(0))
    assert env.observation_size == upright_env.observation_size == 20 * 2 + 4 + 9
    assert env.observation(state).shape == (env.observation_size,)
    np.testing.assert_allclose(env.observation(state), upright_env.observation(state))
    wide = WalkerEnv(EnvConfig(angle_harmonics=3), env.terrain_config)
    assert wide.observation_size == 20 * 3 + 4 + 9
    # Pitch wrap-around is invisible to the encoding.
    turned = dataclasses.replace(state, data=state.data.replace(qpos=state.data.qpos.at[2].add(2 * jnp.pi)))
    np.testing.assert_allclose(env.observation(turned)[:4], env.observation(state)[:4], atol=1e-5)
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


def test_work_integrates_actual_torque_and_velocity_and_shapes_only_reward(env, monkeypatch):
    from mujoco import mjx
    state = jax.jit(env.reset)(jax.random.key(0))
    state = eqx.tree_at(lambda s: s.data.qvel, state, state.data.qvel.at[3:].set(2))

    def physics(model, data):
        return data.replace(qfrc_actuator=data.qfrc_actuator.at[3:].set(3))

    monkeypatch.setattr(mjx, 'step', physics)
    transition = jax.jit(env.transition)
    base, work = transition(state, jnp.zeros(4), 0.0)
    penalized, second_work = transition(state, jnp.zeros(4), 2.0)
    expected = 4 * 3 * 2 * env.config.timestep * env.config.frame_skip
    assert float(work) == pytest.approx(expected)
    assert float(second_work) == pytest.approx(expected)
    assert float(base.reward - penalized.reward) == pytest.approx(2 * expected / env.weight)
    np.testing.assert_array_equal(base.data.qpos, penalized.data.qpos)
