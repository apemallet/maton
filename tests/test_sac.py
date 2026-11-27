import numpy as np
import pytest

from maton.sac import SACConfig, load_agent, train
from maton.training import TrainingConfig
from test_offpolicy import ToyEnv


def test_sac_learns_toy_optimum_and_reloads(tmp_path):
    config = SACConfig(num_envs=4, batch_size=64, seed_steps=64, hidden_size=32, log_every=1024)
    result = train(ToyEnv(), config, TrainingConfig(steps=1024, learning_rate=1e-3), tmp_path, seed=0)
    assert result["updates"] == 1024 and np.isfinite(result["last"]["q_loss"])
    agent = load_agent(tmp_path)
    agent.reset()
    assert abs(float(agent(np.zeros(2, np.float32))[0]) - 0.5) < 0.1  # reward peaks at action 0.5


@pytest.mark.parametrize("bad", [dict(num_q=1), dict(initial_alpha=0.0), dict(log_std_min=3.0), dict(tau=0.0)])
def test_config_validation(bad):
    with pytest.raises(ValueError):
        SACConfig(**bad)
