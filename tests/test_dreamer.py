from pathlib import Path

import numpy as np
import pytest

from maton.dreamer import DreamerConfig, _JaxCompat, _observation, agent_config


def test_config_validation():
    for kwargs in ({"batch_length": 1}, {"steps": 0}, {"hidden": 31}, {"learning_rate": float("nan")}):
        with pytest.raises(ValueError):
            DreamerConfig(**kwargs)


def test_small_upstream_config(tmp_path: Path):
    cfg = agent_config(DreamerConfig(), tmp_path, 3)
    assert cfg.dyn.rssm.deter == 64
    assert cfg.rewhead.units == 32
    assert cfg.replay_context == 0
    assert cfg.jax.transfer_guard is False


def test_observation_boundary_semantics():
    obs = _observation([1, 2], reward=0.5, last=True, terminal=False)
    assert obs["vector"].dtype == np.float32
    assert obs["reward"].shape == ()
    assert obs["is_last"] and not obs["is_terminal"]


def test_legacy_jit_adapter():
    fn = _JaxCompat.jit(lambda x: x + 1, None, None, (), None, ())
    assert int(fn(2)) == 3
