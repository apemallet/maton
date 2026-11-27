import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maton.offpolicy import Replay
from maton.offpolicy import discount
from maton.tdmpc2 import TDMPC2, TDMPC2Config
from maton.training import TrainingConfig


def learner(**overrides):
    config = TDMPC2Config(**dict(latent_dim=16, mlp_dim=32, enc_dim=16, num_q=3, num_samples=32,
                                 num_elites=4, num_pi_trajs=4, iterations=2) | overrides)
    return TDMPC2(config, 5, 2, 100, TrainingConfig(learning_rate=3e-4))


def test_discount_heuristic_matches_official_table():
    assert [round(discount(None, length), 4) for length in (100, 500, 1000, 10_000)] == [0.95, 0.99, 0.995, 0.995]
    assert discount(0.9, 500) == 0.9


def test_two_hot_roundtrip_and_soft_ce_prefers_target():
    model = learner()
    values = jnp.array([-50.0, -1.5, 0.0, 0.3, 7.0, 1e3])
    encoded = model.two_hot(values)
    np.testing.assert_allclose(encoded.sum(-1), 1, rtol=1e-6)
    np.testing.assert_allclose(model.decode(jnp.log(encoded + 1e-12)), values, rtol=1e-3, atol=1e-3)
    logits = jnp.log(encoded[3] + 1e-6)
    assert model.soft_ce(logits, jnp.array(0.3)) < model.soft_ce(logits, jnp.array(7.0))


def test_planner_shapes_bounds_and_warm_start():
    model = learner()
    state = model.init(jax.random.key(0))
    carry = jnp.ones((3, model.horizon, 2)) * 0.5
    actions, means = model.act(state.model, carry, jnp.zeros((3, 5)), jnp.array([True, False, False]),
                               jax.random.key(1), True)
    assert actions.shape == (3, 2) and means.shape == (3, model.horizon, 2)
    assert jnp.all(jnp.abs(actions) <= 1)
    policy_only = learner(mpc=False)
    action, _ = policy_only.act(state.model, carry, jnp.zeros((3, 5)), jnp.zeros(3, bool), jax.random.key(1), False)
    assert action.shape == (3, 2)


def test_update_is_finite_and_moves_targets_slowly():
    model = learner()
    state = model.init(jax.random.key(0))
    replay = Replay.create(8, 2, 5, 2)
    for t in range(8):
        replay = replay.add(jax.random.normal(jax.random.key(t), (2, 5)), jnp.full((2, 2), 0.3), jnp.ones(2),
                            jax.random.normal(jax.random.key(t + 9), (2, 5)), jnp.array([False, t == 5]),
                            jnp.array([False, t == 5]))
    batch = replay.sample(jax.random.key(1), 16, model.horizon)
    updated, metrics = jax.jit(model.update)(state, batch, jax.random.key(2))
    assert all(np.isfinite(float(value)) for value in metrics.values())
    online = updated.model.world.qs.layers[0].weight
    old_target, target = state.target_qs.layers[0].weight, updated.target_qs.layers[0].weight
    np.testing.assert_allclose(target - old_target, 0.01 * (online - old_target), rtol=1e-4, atol=1e-7)


@pytest.mark.parametrize("bad", [dict(latent_dim=12, simnorm_dim=8), dict(num_q=1), dict(num_pi_trajs=512),
                                 dict(min_std=3.0), dict(vmin=1.0, vmax=0.0), dict(update_ratio=-1.0)])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        TDMPC2Config(**bad)
