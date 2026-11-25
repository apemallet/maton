from copy import deepcopy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maton.env import EnvConfig, WalkerEnv
from maton.replay import record
from maton.terrain import TerrainConfig
from maton.walker import Body, Geom, Joint, Motor, WalkerConfig, default_walker


@pytest.mark.parametrize('name,motors,contacts', [('basic', 4, 2), ('biped', 4, 2), ('quadruped', 8, 4)])
def test_example_assembly(name, motors, contacts):
    walker = WalkerConfig.load(Path('configs/walkers') / f'{name}.toml')
    env = WalkerEnv(EnvConfig(frame_skip=1, reset_noise=0), TerrainConfig(slots=1), walker=walker)
    assert env.model.nq == env.model.nv == 3 + motors
    assert env.action_size == motors
    assert env.model.nsensordata == contacts
    state = jax.jit(env.reset)(jax.random.key(0))
    state = jax.jit(env.step)(state, jnp.zeros(motors))
    assert env.observation(state).shape == (4 * (1 + motors) * 2 + 2 + contacts + 9,)
    assert np.isfinite(env.observation(state)).all()
    assert env.cpu_model(state).nu == motors


def custom_walker():
    """Out-of-order tree, a fixed link, passive joint, and distinct motor settings."""
    geom = Geom('sphere', (.06,), .5)
    return WalkerConfig('custom', (
        Body('end', (geom,), 'fixed', (0, 0, -.2),
             Joint('end_hinge', (-1, 1), motor=Motor(10, 10, 1, (-.5, .5), .1))),
        Body('root', (Geom('box', (.1, .1, .1), 2),)),
        Body('passive', (geom,), 'root', (0, 0, -.2), Joint('passive_hinge', (-1, 1))),
        Body('fixed', (geom,), 'passive', (0, 0, -.2)),
        Body('branch', (geom,), 'root', (.2, 0, -.2),
             Joint('branch_hinge', (-2, 2), motor=Motor(40, 20, 4, (-1, 1), -.2))),
    ), spawn_height=.8)


def test_fixed_passive_joints_motor_mapping_and_order(tmp_path):
    walker = custom_walker()
    env = WalkerEnv(EnvConfig(control_mode='position', frame_skip=1, episode_length=1, reset_noise=0),
                    TerrainConfig(slots=1), walker=walker)
    assert [b.name for b in walker.ordered_bodies] == ['root', 'passive', 'fixed', 'end', 'branch']
    assert [j.name for j in walker.motors] == ['end_hinge', 'branch_hinge']
    assert env.action_size == 2
    assert env.model.nq == 6
    assert env.model.nsensordata == 0
    state = jax.jit(env.reset)(jax.random.key(0))
    # Passive hinge does not consume a motor command or a PD target.
    data = state.data.replace(qpos=state.data.qpos.at[3:].set(jnp.array([.8, .1, -.2])))
    np.testing.assert_allclose(env.motor_control(data, jnp.zeros(2)), 0, atol=1e-6)
    data = data.replace(qvel=data.qvel.at[3:].set(1))
    np.testing.assert_allclose(env.motor_control(data, jnp.zeros(2)), [-.1, -.1], atol=1e-6)
    np.testing.assert_allclose(env.motor_control(state.data, jnp.ones(2)), [.5, .5], atol=1e-6)
    np.testing.assert_allclose(env.motor_control(state.data, -jnp.ones(2)), [-.5, -.5], atol=1e-6)
    assert env.observation(state).shape == (env.observation_size,)
    np.testing.assert_allclose(env.lidar(state)[0], .8 / env.config.lidar_range, atol=1e-6)
    output = record(env, lambda _: np.zeros(2), tmp_path / 'episode', seed=1)
    with np.load(output / 'trajectory.npz') as saved:
        assert saved['actions'].shape == (1, 2)
        assert saved['qpos'].shape == (2, 6)


@pytest.mark.parametrize('mutation', [
    lambda d: d.update(typo=True),
    lambda d: d['bodies'][1].update(parent='missing'),
    lambda d: d['bodies'][1].update(parent=d['bodies'][2]['name']),
    lambda d: d['bodies'][1].update(name=d['bodies'][0]['name']),
    lambda d: d['bodies'][1]['joint'].update(name='root_reserved'),
    lambda d: d['bodies'][1]['joint'].update(range=[1, 2]),
    lambda d: d['bodies'][1]['joint']['motor'].update(target_range=[-5, 5]),
    lambda d: d['bodies'][1]['joint']['motor'].update(neutral=100),
    lambda d: d['bodies'][1]['joint']['motor'].update(torque_limit=0),
    lambda d: d['bodies'][1]['joint']['motor'].update(kd=-1),
    lambda d: d['bodies'][1]['joint']['motor'].update(kp=float('nan')),
    lambda d: d['bodies'][1]['geoms'][0].update(size=[0]),
    lambda d: d['bodies'][1]['geoms'][0].update(fromto=[0]*6),
    lambda d: d['bodies'][0].update(pos=[1, 0, 0]),
    lambda d: d.update(spawn_height=float('inf')),
])
def test_invalid_structures(mutation):
    values = deepcopy(default_walker().to_dict())
    mutation(values)
    with pytest.raises((ValueError, TypeError)):
        WalkerConfig.from_dict(values)


def test_no_motors_rejected():
    with pytest.raises(ValueError, match='motor'):
        WalkerConfig('unactuated', (Body('root', (Geom('sphere', (.1,), 1),)),))
