from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp
from typing import NamedTuple

from maton import dreamer
from maton.dreamer import DreamerConfig, _JaxCompat, _observation, agent_config
from maton.training import TrainingConfig


def test_config_validation():
    for kwargs in ({"batch_length": 1}, {"hidden": 31}):
        with pytest.raises(ValueError):
            DreamerConfig(**kwargs)


def test_small_upstream_config(tmp_path: Path):
    cfg = agent_config(DreamerConfig(), TrainingConfig(learning_rate=0.0002), tmp_path, 3)
    assert cfg.dyn.rssm.deter == 64
    assert cfg.rewhead.units == 32
    assert cfg.replay_context == 0
    assert cfg.jax.transfer_guard is False
    assert cfg.opt.lr == 0.0002


def test_observation_boundary_semantics():
    obs = _observation([1, 2], reward=0.5, last=True, terminal=False)
    assert obs["vector"].dtype == np.float32
    assert obs["reward"].shape == ()
    assert obs["is_last"] and not obs["is_terminal"]


def test_legacy_jit_adapter():
    fn = _JaxCompat.jit(lambda x: x + 1, None, None, (), None, ())
    assert int(fn(2)) == 3


def test_budget_and_checkpoint_units_exclude_episode_reset_bookkeeping(tmp_path, monkeypatch):
    elements, embodied, _ = dreamer._runtime()

    class Agent:
        def __init__(self, *args):
            pass

        def init_train(self, size):
            return None

        def init_policy(self, size):
            return None

        def stream(self, stream):
            return stream

        def policy(self, carry, obs, mode):
            return carry, {'action': np.zeros((1, 4), np.float32)}, {}

        def train(self, carry, batch):
            return carry, {}, {'loss': np.array(1.)}

        def save(self):
            return {'params': {'weight': np.zeros(1)}}

    class State(NamedTuple):
        steps: object
        reward: object
        done: object
        terminated: object

    class Env:
        def reset(self, key):
            return State(jnp.array(0), jnp.array(0.), jnp.array(False), jnp.array(False))

        def observation(self, state):
            return jnp.array([state.steps], dtype=jnp.float32)

        def step(self, state, action):
            return State(state.steps + 1, jnp.array(1.), state.steps >= 1, jnp.array(False))

    monkeypatch.setattr(dreamer, '_runtime', lambda: (elements, embodied, Agent))
    result = dreamer.train(Env(), DreamerConfig(batch_length=2, train_every=2),
                           TrainingConfig(steps=8, checkpoint_every=3), tmp_path)
    assert result['steps'] == 8
    assert result['episodes'] == 4
    assert result['updates'] == 4
    assert sorted(p.name for p in tmp_path.glob('dreamer-*.pkl')) == [
        'dreamer-00000003.pkl', 'dreamer-00000006.pkl']
    import pickle
    with (tmp_path / 'dreamer.pkl').open('rb') as handle:
        payload = pickle.load(handle)
    assert payload['step'] == 8
    assert payload['training']['checkpoint_every'] == 3
