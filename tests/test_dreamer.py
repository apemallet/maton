from pathlib import Path

import numpy as np
import pytest
import jax.numpy as jnp
from typing import NamedTuple

from maton import dreamer
from maton.dreamer import DreamerConfig, _JaxCompat, _observation, agent_config
from maton.training import TrainingConfig
from maton.curriculum import CurriculumConfig, Level
from maton.terrain import TerrainConfig


def test_config_validation():
    for kwargs in ({"batch_length": 1}, {"hidden": 31}, {"optimizer_warmup": -1}, {"learning_rate": 0.0}, {"learning_rate": float("inf")}, {"policy_minstd": 0}, {"policy_maxstd": float("nan")}, {"policy_minstd": 0.5, "policy_maxstd": 0.1}):
        with pytest.raises(ValueError):
            DreamerConfig(**kwargs)


def test_small_upstream_config(tmp_path: Path):
    cfg = agent_config(DreamerConfig(), TrainingConfig(learning_rate=0.0002), tmp_path, 3)
    assert cfg.dyn.rssm.deter == 64
    assert cfg.rewhead.units == 32
    assert cfg.replay_context == 0
    assert cfg.jax.transfer_guard is False
    assert cfg.opt.lr == 0.0002
    narrow = agent_config(DreamerConfig(policy_minstd=0.1, policy_maxstd=0.3, optimizer_warmup=1000), TrainingConfig(), tmp_path, 0)
    assert narrow.policy.minstd == 0.1
    assert narrow.policy.maxstd == 0.3
    assert narrow.opt.warmup == 1000
    assert narrow.opt.lr == 0.0001
    own = agent_config(DreamerConfig(learning_rate=0.00004), TrainingConfig(learning_rate=0.0003), tmp_path, 0)
    assert own.opt.lr == 0.00004


def test_observation_boundary_semantics():
    obs = _observation([1, 2], reward=0.5, last=True, terminal=False)
    assert obs["vector"].dtype == np.float32
    assert obs["reward"].shape == ()
    assert obs["is_last"] and not obs["is_terminal"]


def test_legacy_jit_adapter():
    fn = _JaxCompat.jit(lambda x: x + 1, None, None, (), None, ())
    assert int(fn(2)) == 3


def fake_runtime(monkeypatch):
    elements, embodied, _ = dreamer._runtime()

    class Agent:
        def __init__(self, obs_space, act_space, config):
            assert act_space['action'].shape == (2,)

        def init_train(self, size):
            return None

        def init_policy(self, size):
            return None

        def stream(self, stream):
            return stream

        def policy(self, carry, obs, mode):
            return carry, {'action': np.zeros((1, 2), np.float32)}, {}

        def train(self, carry, batch):
            return carry, {}, {'loss': np.array(1.)}

        def save(self):
            return {'params': {'weight': np.zeros(1)}}

        def load(self, state):
            self.loaded = state

    class State(NamedTuple):
        steps: object
        reward: object
        done: object
        terminated: object
        level: object
        success: object

    class Env:
        action_size = 2
        flat = TerrainConfig(block_frequency=0, gap_frequency=0)
        curriculum = CurriculumConfig((Level("one", 2, flat), Level("two", 2, flat)))

        def reset(self, key, level=0):
            return State(jnp.array(0), jnp.array(0.), jnp.array(False), jnp.array(False),
                         jnp.asarray(level), jnp.array(False))

        def observation(self, state):
            return jnp.array([state.steps], dtype=jnp.float32)

        def step(self, state, action):
            return State(state.steps + 1, jnp.array(1.), state.steps >= 1, jnp.array(False),
                         state.level, state.steps >= 1)

    monkeypatch.setattr(dreamer, '_runtime', lambda: (elements, embodied, Agent))
    return Env


def test_budget_and_checkpoint_units_exclude_episode_reset_bookkeeping(tmp_path, monkeypatch):
    elements, embodied, _ = dreamer._runtime()

    class Agent:
        def __init__(self, obs_space, act_space, config):
            assert act_space['action'].shape == (2,)

        def init_train(self, size):
            return None

        def init_policy(self, size):
            return None

        def stream(self, stream):
            return stream

        def policy(self, carry, obs, mode):
            return carry, {'action': np.zeros((1, 2), np.float32)}, {}

        def train(self, carry, batch):
            return carry, {}, {'loss': np.array(1.)}

        def save(self):
            return {'params': {'weight': np.zeros(1)}}

    class State(NamedTuple):
        steps: object
        reward: object
        done: object
        terminated: object
        level: object
        success: object

    class Env:
        action_size = 2
        flat = TerrainConfig(block_frequency=0, gap_frequency=0)
        curriculum = CurriculumConfig((Level("one", 2, flat), Level("two", 2, flat)))

        def reset(self, key, level=0):
            return State(jnp.array(0), jnp.array(0.), jnp.array(False), jnp.array(False),
                         jnp.asarray(level), jnp.array(False))

        def observation(self, state):
            return jnp.array([state.steps], dtype=jnp.float32)

        def step(self, state, action):
            return State(state.steps + 1, jnp.array(1.), state.steps >= 1, jnp.array(False),
                         state.level, state.steps >= 1)

    monkeypatch.setattr(dreamer, '_runtime', lambda: (elements, embodied, Agent))
    result = dreamer.train(Env(), DreamerConfig(batch_length=2, train_every=2),
                           TrainingConfig(steps=8, checkpoint_every=3), tmp_path)
    assert result['steps'] == 8
    assert result['episodes'] == 4
    assert result['updates'] == 4
    assert result['curriculum']['complete']
    assert result['curriculum']['level'] == 1
    assert sorted(p.name for p in tmp_path.glob('dreamer-*.pkl')) == [
        'dreamer-00000003.pkl', 'dreamer-00000006.pkl']
    import pickle
    with (tmp_path / 'dreamer.pkl').open('rb') as handle:
        payload = pickle.load(handle)
    assert payload['action_size'] == 2
    assert payload['step'] == 8
    assert payload['training']['checkpoint_every'] == 3
    assert payload['curriculum']['level'] == 1


def test_evaluation_uses_policy_prediction(monkeypatch):
    from types import SimpleNamespace
    from maton.dreamer import _runtime
    from dreamerv3.agent import Agent
    _, _, EvaluationAgent = _runtime()
    carry = ({}, {"feature": 1}, {}, {"action": np.array([0.9])})
    monkeypatch.setattr(Agent, "policy", lambda self, c, o, m: (carry, carry[-1], {}))
    model = object.__new__(EvaluationAgent)
    model.feat2tensor = lambda feature: feature
    model.pol = lambda feature, bdims: {"action": SimpleNamespace(pred=lambda: np.array([0.2]))}
    updated, action, _ = model.policy(carry, {}, "eval")
    assert np.array_equal(action["action"], [0.2])
    assert updated[-1] is action
    _, sampled, _ = model.policy(carry, {}, "train")
    assert np.array_equal(sampled["action"], [0.9])


def test_pause_and_resume_keeps_budget_counters_and_curriculum(tmp_path, monkeypatch):
    Env = fake_runtime(monkeypatch)
    training = TrainingConfig(steps=8, checkpoint_every=3)
    paused = dreamer.train(Env(), DreamerConfig(batch_length=2, train_every=2), training, tmp_path,
                           pause=lambda step: step >= 4)
    assert not paused["complete"] and paused["steps"] == 4 and not (tmp_path / "dreamer.pkl").exists()
    resumed = dreamer.train(Env(), DreamerConfig(batch_length=2, train_every=2), training, tmp_path)
    assert resumed["complete"] and resumed["steps"] == 8
    assert resumed["episodes"] == 4 and resumed["curriculum"]["complete"]
    assert (tmp_path / "dreamer.pkl").exists()


def test_continuation_resets_optimizer_counters_and_curriculum(tmp_path, monkeypatch):
    import pickle
    Env = fake_runtime(monkeypatch)
    elements, embodied, Base = dreamer._runtime()
    loads = []

    class Agent(Base):
        def save(self):
            return {'params': {'weight': np.zeros(1), 'opt/momentum': np.zeros(1)},
                    'counters': {'updates': 0, 'batches': 0, 'actions': 0}}

        def load(self, state):
            loads.append(state)

    monkeypatch.setattr(dreamer, '_runtime', lambda: (elements, embodied, Agent))
    source = tmp_path / 'source'
    dreamer.train(Env(), DreamerConfig(batch_length=2, train_every=2), TrainingConfig(steps=8), source)
    checkpoint = source / 'dreamer.pkl'
    with checkpoint.open('rb') as stream:
        payload = pickle.load(stream)
    payload['agent']['params']['weight'][:] = 5
    payload['agent']['params']['opt/momentum'][:] = 9
    payload['agent']['counters'] = dict(updates=99, batches=99, actions=99)
    with checkpoint.open('wb') as stream:
        pickle.dump(payload, stream)
    result = dreamer.train(Env(), DreamerConfig(batch_length=2, train_every=2), TrainingConfig(steps=1),
                           tmp_path / 'child', checkpoint=checkpoint)
    assert result['steps'] == 1 and result['curriculum']['level'] == 0
    assert loads[0]['params']['weight'][0] == 5
    assert loads[0]['params']['opt/momentum'][0] == 0
    assert loads[0]['counters'] == dict(updates=0, batches=0, actions=0)
